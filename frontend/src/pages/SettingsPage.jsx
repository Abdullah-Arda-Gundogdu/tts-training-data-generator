import { useState, useEffect } from 'react'
import { Key, Save, AlertCircle, Check, Eye, EyeOff, RefreshCw, Settings } from 'lucide-react'

const API_BASE_URL = `http://${window.location.hostname}:5001`

function SettingsPage() {
    const [keys, setKeys] = useState({})
    const [edits, setEdits] = useState({})
    const [showSecrets, setShowSecrets] = useState({})
    const [isSaving, setIsSaving] = useState(false)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(null)

    useEffect(() => {
        loadKeys()
    }, [])

    useEffect(() => {
        if (error) { const t = setTimeout(() => setError(null), 5000); return () => clearTimeout(t) }
    }, [error])
    useEffect(() => {
        if (success) { const t = setTimeout(() => setSuccess(null), 5000); return () => clearTimeout(t) }
    }, [success])

    const loadKeys = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/settings/keys`)
            const data = await response.json()
            if (data.success) setKeys(data.keys)
        } catch (err) {
            setError('Ayarlar yüklenemedi: ' + err.message)
        }
    }

    const handleEdit = (keyName, value) => {
        setEdits(prev => ({ ...prev, [keyName]: value }))
    }

    const saveKeys = async () => {
        if (Object.keys(edits).length === 0) {
            setError('Değişiklik yapılmadı')
            return
        }

        setIsSaving(true)
        try {
            const response = await fetch(`${API_BASE_URL}/api/settings/keys`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(edits)
            })
            const data = await response.json()
            if (data.success) {
                setSuccess(`✅ ${data.updated_keys.length} anahtar güncellendi`)
                setEdits({})
                loadKeys()
            } else {
                setError(data.error || 'Ayarlar kaydedilemedi')
            }
        } catch (err) {
            setError('Ayarlar kaydedilemedi: ' + err.message)
        } finally {
            setIsSaving(false)
        }
    }

    const toggleSecret = (keyName) => {
        setShowSecrets(prev => ({ ...prev, [keyName]: !prev[keyName] }))
    }

    const isSecret = (keyName) => keyName.includes('KEY') || keyName.includes('SECRET')

    const keyOrder = ['OPENAI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'GEMINI_API_KEY', 'OLLAMA_BASE_URL']

    return (
        <div className="page-content">
            {/* Notifications */}
            {error && <div className="notification error"><AlertCircle size={20} /><span>{error}</span></div>}
            {success && <div className="notification success"><Check size={20} /><span>{success}</span></div>}

            {/* Header */}
            <div className="page-header">
                <div className="page-header-left">
                    <Settings className="header-icon" />
                    <div>
                        <h1>Ayarlar</h1>
                        <p>API anahtarları ve yapılandırma</p>
                    </div>
                </div>
                <button className="btn btn-ghost" onClick={loadKeys}>
                    <RefreshCw size={16} /> Yenile
                </button>
            </div>

            {/* API Keys Card */}
            <div className="card">
                <h3 className="card-title"><Key size={18} /> API Anahtarları</h3>
                <p className="text-muted" style={{ marginBottom: '1.5rem' }}>
                    Servisler için gereken API anahtarlarını ve yolları burada yönetebilirsiniz.
                    Değerler <code>.env</code> dosyasında saklanır.
                </p>

                <div className="settings-keys-list">
                    {keyOrder.map(keyName => {
                        const keyInfo = keys[keyName]
                        if (!keyInfo) return null

                        const isEditing = keyName in edits
                        const currentValue = isEditing ? edits[keyName] : ''
                        const secret = isSecret(keyName)

                        return (
                            <div key={keyName} className="settings-key-row">
                                <div className="key-header">
                                    <label className="key-label">{keyInfo.label}</label>
                                    <span className={`key-status ${keyInfo.is_set ? 'key-set' : 'key-unset'}`}>
                                        {keyInfo.is_set ? '✅ Ayarlandı' : '⚠️ Boş'}
                                    </span>
                                </div>

                                <div className="key-input-row">
                                    <div className="key-input-wrapper">
                                        {isEditing ? (
                                            <input
                                                type={secret && !showSecrets[keyName] ? 'password' : 'text'}
                                                value={currentValue}
                                                onChange={e => handleEdit(keyName, e.target.value)}
                                                onKeyDown={e => { if (e.key === 'Enter') saveKeys() }}
                                                placeholder={`Yeni ${keyInfo.label} girin...`}
                                                autoFocus
                                            />
                                        ) : (
                                            <div className="key-current-value">
                                                <span>{secret && !showSecrets[keyName] ? (keyInfo.value || '(ayarlanmamış)') : (keyInfo.value || '(ayarlanmamış)')}</span>
                                            </div>
                                        )}
                                    </div>

                                    <div className="key-actions">
                                        {secret && (
                                            <button className="btn btn-ghost btn-icon" onClick={() => toggleSecret(keyName)} title={showSecrets[keyName] ? 'Gizle' : 'Göster'}>
                                                {showSecrets[keyName] ? <EyeOff size={16} /> : <Eye size={16} />}
                                            </button>
                                        )}
                                        {!isEditing ? (
                                            <button className="btn btn-ghost btn-icon" onClick={() => handleEdit(keyName, keyInfo.value?.includes('*') ? '' : keyInfo.value || '')}>
                                                Düzenle
                                            </button>
                                        ) : (
                                            <button className="btn btn-ghost btn-icon" onClick={() => setEdits(prev => { const n = { ...prev }; delete n[keyName]; return n })}>
                                                İptal
                                            </button>
                                        )}
                                    </div>
                                </div>

                                {keyName === 'OPENAI_API_KEY' && (
                                    <p className="key-hint">OpenAI GPT API anahtarınız. Cümle üretimi için kullanılır.</p>
                                )}
                                {keyName === 'GOOGLE_APPLICATION_CREDENTIALS' && (
                                    <p className="key-hint">Google Cloud servis hesap JSON dosyasının yolu. Google Cloud TTS için kullanılır.</p>
                                )}
                                {keyName === 'GEMINI_API_KEY' && (
                                    <p className="key-hint">Gemini 2.5 Flash TTS ile ses üretimi için gereklidir. <a href="https://aistudio.google.com" target="_blank" rel="noreferrer">Anahtar al →</a></p>
                                )}
                                {keyName === 'OLLAMA_BASE_URL' && (
                                    <p className="key-hint">Ollama API sunucu adresi. Varsayılan: http://localhost:11434</p>
                                )}
                            </div>
                        )
                    })}
                </div>

                {/* Save Button */}
                {Object.keys(edits).length > 0 && (
                    <div className="settings-save-bar">
                        <span className="text-muted">{Object.keys(edits).length} değişiklik yapıldı</span>
                        <button className="btn btn-primary" onClick={saveKeys} disabled={isSaving}>
                            <Save size={16} />
                            {isSaving ? 'Kaydediliyor...' : 'Değişiklikleri Kaydet'}
                        </button>
                    </div>
                )}
            </div>

            {/* Info Card */}
            <div className="card" style={{ marginTop: '1.5rem' }}>
                <h3 className="card-title">ℹ️ Bilgi</h3>
                <div className="info-list">
                    <div className="info-item">
                        <strong>OpenAI API Key</strong> — GPT ile cümle üretimi için gereklidir. <a href="https://platform.openai.com/api-keys" target="_blank" rel="noreferrer">Anahtar al →</a>
                    </div>
                    <div className="info-item">
                        <strong>Google Cloud Credentials</strong> — Google Cloud TTS ile ses dosyası oluşturma için gereklidir. JSON formatında bir servis hesap dosyası kullanılır.
                    </div>
                    <div className="info-item">
                        <strong>Gemini API Key</strong> — Gemini 2.5 Flash TTS ile doğal ses üretimi için gereklidir. <a href="https://aistudio.google.com" target="_blank" rel="noreferrer">Google AI Studio →</a>
                    </div>
                    <div className="info-item">
                        <strong>Ollama</strong> — Yerel LLM kullanmak istiyorsanız Ollama sunucusu çalışıyor olmalıdır. <a href="https://ollama.ai" target="_blank" rel="noreferrer">Ollama →</a>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SettingsPage
