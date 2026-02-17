import { useState, useEffect, useRef } from 'react'
import { Volume2, Trash2, Tag, Search, Plus, AlertCircle, Check, Clock, XCircle, Brain, X, Mic, Loader, Pause } from 'lucide-react'

const API_BASE_URL = `http://${window.location.hostname}:5001`

function ModelsPage() {
    const [models, setModels] = useState([])
    const [searchQuery, setSearchQuery] = useState('')
    const [filterStatus, setFilterStatus] = useState('all')
    const [playingId, setPlayingId] = useState(null)
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(null)
    const audioRef = useRef(null)

    // Manual model add dialog
    const [showAddDialog, setShowAddDialog] = useState(false)
    const [newModel, setNewModel] = useState({ name: '', description: '', tags: '', model_path: '' })

    // Test panel
    const [testModelId, setTestModelId] = useState(null)
    const [testText, setTestText] = useState('')
    const [testLanguage, setTestLanguage] = useState('tr')
    const [isGenerating, setIsGenerating] = useState(false)
    const [testAudioUrl, setTestAudioUrl] = useState(null)
    const [testPlayingId, setTestPlayingId] = useState(null)
    const testAudioRef = useRef(null)

    useEffect(() => {
        loadModels()
        const interval = setInterval(loadModels, 10000)
        return () => clearInterval(interval)
    }, [])

    useEffect(() => {
        if (error) { const t = setTimeout(() => setError(null), 5000); return () => clearTimeout(t) }
    }, [error])
    useEffect(() => {
        if (success) { const t = setTimeout(() => setSuccess(null), 5000); return () => clearTimeout(t) }
    }, [success])

    // Cleanup test audio URL on unmount
    useEffect(() => {
        return () => {
            if (testAudioUrl) URL.revokeObjectURL(testAudioUrl)
        }
    }, [testAudioUrl])

    const loadModels = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models`)
            const data = await response.json()
            if (data.success) setModels(data.models)
        } catch (err) {
            console.error('Models yüklenemedi:', err)
        }
    }

    const deleteModel = async (id) => {
        if (!confirm('Bu modeli silmek istediğinize emin misiniz?')) return
        try {
            const response = await fetch(`${API_BASE_URL}/api/models/${id}`, { method: 'DELETE' })
            if (response.ok) {
                setSuccess('✅ Model silindi')
                if (testModelId === id) closeTestPanel()
                loadModels()
            }
        } catch (err) {
            setError('Model silinemedi: ' + err.message)
        }
    }

    const addManualModel = async () => {
        if (!newModel.name.trim()) { setError('Model adı gerekli'); return }
        if (!newModel.model_path.trim()) { setError('Model dosya yolu gerekli'); return }
        try {
            const response = await fetch(`${API_BASE_URL}/api/models`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newModel.name.trim(),
                    description: newModel.description.trim(),
                    tags: newModel.tags.split(',').map(t => t.trim()).filter(Boolean),
                    model_path: newModel.model_path.trim(),
                    status: 'completed'
                })
            })
            if (response.ok) {
                setSuccess('✅ Model eklendi')
                setShowAddDialog(false)
                setNewModel({ name: '', description: '', tags: '', model_path: '' })
                loadModels()
            }
        } catch (err) {
            setError('Model eklenemedi: ' + err.message)
        }
    }

    // --- Test panel functions ---
    const openTestPanel = (modelId) => {
        setTestModelId(modelId)
        setTestText('')
        setTestLanguage('tr')
        setIsGenerating(false)
        if (testAudioUrl) { URL.revokeObjectURL(testAudioUrl); setTestAudioUrl(null) }
    }

    const closeTestPanel = () => {
        setTestModelId(null)
        setTestText('')
        if (testAudioUrl) { URL.revokeObjectURL(testAudioUrl); setTestAudioUrl(null) }
        if (testAudioRef.current) { testAudioRef.current.pause(); testAudioRef.current = null }
        setTestPlayingId(null)
    }

    const synthesize = async () => {
        if (!testText.trim()) { setError('Lütfen bir metin girin'); return }
        if (!testModelId) return

        setIsGenerating(true)
        if (testAudioUrl) { URL.revokeObjectURL(testAudioUrl); setTestAudioUrl(null) }
        if (testAudioRef.current) { testAudioRef.current.pause(); testAudioRef.current = null }
        setTestPlayingId(null)

        try {
            const response = await fetch(`${API_BASE_URL}/api/models/${testModelId}/synthesize`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: testText.trim(),
                    language: testLanguage
                })
            })

            if (!response.ok) {
                const errData = await response.json()
                throw new Error(errData.error || 'Sentez başarısız')
            }

            const blob = await response.blob()
            const url = URL.createObjectURL(blob)
            setTestAudioUrl(url)
            setSuccess('🔊 Ses üretildi!')

            // Auto-play
            const audio = new Audio(url)
            audio.onended = () => setTestPlayingId(null)
            testAudioRef.current = audio
            setTestPlayingId(testModelId)
            audio.play().catch(() => { })
        } catch (err) {
            setError('Sentez hatası: ' + err.message)
        } finally {
            setIsGenerating(false)
        }
    }

    const playTestResult = () => {
        if (!testAudioUrl) return
        if (testAudioRef.current && !testAudioRef.current.paused) {
            testAudioRef.current.pause()
            setTestPlayingId(null)
            return
        }
        const audio = new Audio(testAudioUrl)
        audio.onended = () => setTestPlayingId(null)
        testAudioRef.current = audio
        setTestPlayingId(testModelId)
        audio.play().catch(() => { setError('Ses oynatılamadı'); setTestPlayingId(null) })
    }

    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed': return <Check size={14} />
            case 'training': return <Clock size={14} />
            case 'failed': return <XCircle size={14} />
            default: return <Clock size={14} />
        }
    }

    const getStatusClass = (status) => {
        switch (status) {
            case 'completed': return 'status-badge status-completed'
            case 'training': return 'status-badge status-training'
            case 'failed': return 'status-badge status-failed'
            default: return 'status-badge status-pending'
        }
    }

    const filteredModels = models.filter(m => {
        const matchesSearch = !searchQuery ||
            m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            m.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
            (m.tags || []).some(t => t.toLowerCase().includes(searchQuery.toLowerCase()))
        const matchesStatus = filterStatus === 'all' || m.status === filterStatus
        return matchesSearch && matchesStatus
    })

    const testModel = testModelId ? models.find(m => m.id === testModelId) : null

    return (
        <div className="page-content">
            {/* Notifications */}
            {error && <div className="notification error"><AlertCircle size={20} /><span>{error}</span></div>}
            {success && <div className="notification success"><Check size={20} /><span>{success}</span></div>}

            {/* Header */}
            <div className="page-header">
                <div className="page-header-left">
                    <Brain className="header-icon" />
                    <div>
                        <h1>Model Listesi</h1>
                        <p>Eğitilmiş modeller ve test</p>
                    </div>
                </div>
                <button className="btn btn-primary" onClick={() => setShowAddDialog(true)}>
                    <Plus size={16} /> Manuel Ekle
                </button>
            </div>

            {/* Search & Filter */}
            <div className="card" style={{ marginBottom: '1.5rem' }}>
                <div className="models-toolbar">
                    <div className="search-box">
                        <Search size={18} />
                        <input
                            type="text"
                            placeholder="Model ara (isim, açıklama, tag)..."
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                        />
                    </div>
                    <div className="filter-buttons">
                        {['all', 'completed', 'training', 'failed'].map(s => (
                            <button
                                key={s}
                                className={`btn btn-small ${filterStatus === s ? 'btn-primary' : 'btn-ghost'}`}
                                onClick={() => setFilterStatus(s)}
                            >
                                {s === 'all' ? 'Tümü' : s === 'completed' ? '✅ Tamamlanan' : s === 'training' ? '🟡 Eğitimde' : '🔴 Başarısız'}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* Test Panel (shown when a model is selected for testing) */}
            {testModel && (
                <div className="card test-panel" style={{ marginBottom: '1.5rem' }}>
                    <div className="test-panel-header">
                        <div>
                            <h3 className="card-title" style={{ marginBottom: 0 }}>
                                <Mic size={18} /> Model Test — {testModel.name}
                            </h3>
                            <p className="text-muted" style={{ marginTop: '0.25rem' }}>
                                Metin girin ve modelin nasıl konuştuğunu test edin
                            </p>
                        </div>
                        <button className="btn btn-ghost btn-icon" onClick={closeTestPanel}>
                            <X size={18} />
                        </button>
                    </div>

                    <div className="test-panel-body">
                        <div className="test-input-area">
                            <textarea
                                value={testText}
                                onChange={e => setTestText(e.target.value)}
                                placeholder="Test edilecek metni yazın... Örn: Bu modern tasarımda üç adet kanepe ve sekiz adet sürahi bulunuyor."
                                rows={3}
                                disabled={isGenerating}
                                className="test-textarea"
                            />
                            <div className="test-controls">
                                <select
                                    value={testLanguage}
                                    onChange={e => setTestLanguage(e.target.value)}
                                    disabled={isGenerating}
                                    style={{ padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.3)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text)', fontSize: '0.85rem' }}
                                >
                                    <option value="tr">Türkçe</option>
                                    <option value="en">English</option>
                                    <option value="de">Deutsch</option>
                                    <option value="fr">Français</option>
                                    <option value="es">Español</option>
                                </select>
                                <button
                                    className="btn btn-primary"
                                    onClick={synthesize}
                                    disabled={isGenerating || !testText.trim()}
                                >
                                    {isGenerating ? (
                                        <><Loader size={16} className="spin" /> Üretiliyor...</>
                                    ) : (
                                        <><Volume2 size={16} /> Sesi Üret</>
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* Audio Result */}
                        {testAudioUrl && (
                            <div className="test-result">
                                <button
                                    className={`btn-play-test ${testPlayingId ? 'playing' : ''}`}
                                    onClick={playTestResult}
                                >
                                    {testPlayingId ? <Pause size={20} /> : <Volume2 size={20} />}
                                </button>
                                <div className="test-result-info">
                                    <span style={{ fontWeight: 500 }}>Üretilen ses</span>
                                    <span className="text-muted" style={{ fontSize: '0.75rem' }}>
                                        Tekrar dinlemek için play butonuna tıklayın
                                    </span>
                                </div>
                                <a
                                    href={testAudioUrl}
                                    download={`test_${testModel.name.replace(/\s+/g, '_')}.wav`}
                                    className="btn btn-small btn-ghost"
                                >
                                    İndir
                                </a>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Model Grid */}
            {filteredModels.length === 0 ? (
                <div className="empty-state">
                    <Brain size={48} style={{ opacity: 0.3 }} />
                    <h3>Henüz model yok</h3>
                    <p>Eğitim sayfasından yeni bir model eğitin veya manuel olarak ekleyin.</p>
                </div>
            ) : (
                <div className="model-grid">
                    {filteredModels.map(model => (
                        <div key={model.id} className={`model-card ${testModelId === model.id ? 'model-card-active' : ''}`}>
                            <div className="model-card-header">
                                <h3>{model.name}</h3>
                                <span className={getStatusClass(model.status)}>
                                    {getStatusIcon(model.status)}
                                    {model.status === 'completed' ? 'Tamamlandı' : model.status === 'training' ? 'Eğitimde' : model.status === 'failed' ? 'Başarısız' : model.status === 'cancelled' ? 'İptal Edildi' : 'Bekliyor'}
                                </span>
                            </div>

                            {model.description && (
                                <p className="model-description">{model.description}</p>
                            )}

                            {/* Tags */}
                            {(model.tags || []).length > 0 && (
                                <div className="model-tags">
                                    {model.tags.map((tag, i) => (
                                        <span key={i} className="tag-badge">
                                            <Tag size={10} /> {tag}
                                        </span>
                                    ))}
                                </div>
                            )}

                            {/* Training params */}
                            {model.training_params && Object.keys(model.training_params).length > 0 && (
                                <div className="model-params">
                                    {model.training_params.epochs && <span className="param-chip">Epoch: {model.training_params.epochs}</span>}
                                    {model.training_params.learning_rate && <span className="param-chip">LR: {model.training_params.learning_rate}</span>}
                                    {model.training_params.batch_size && <span className="param-chip">Batch: {model.training_params.batch_size}</span>}
                                </div>
                            )}

                            {/* Meta info */}
                            <div className="model-meta">
                                <span>Oluşturulma: {new Date(model.created_at + (model.created_at?.endsWith('Z') ? '' : 'Z')).toLocaleDateString('tr-TR')}</span>
                                {model.base_model && <span>Temel: {model.base_model}</span>}
                            </div>

                            {/* Actions */}
                            <div className="model-actions">
                                {model.status === 'completed' && (
                                    <button
                                        className={`btn btn-small ${testModelId === model.id ? 'btn-primary' : 'btn-success-small'}`}
                                        onClick={() => testModelId === model.id ? closeTestPanel() : openTestPanel(model.id)}
                                        title="Model ile konuşma sentezi"
                                    >
                                        <Mic size={16} />
                                        {testModelId === model.id ? 'Testi Kapat' : 'Test Et'}
                                    </button>
                                )}
                                <button
                                    className="btn btn-small btn-danger-small"
                                    onClick={() => deleteModel(model.id)}
                                    title="Modeli sil"
                                >
                                    <Trash2 size={16} /> Sil
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* Add Model Dialog */}
            {showAddDialog && (
                <div className="modal-overlay" onClick={() => setShowAddDialog(false)}>
                    <div className="modal-content" onClick={e => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>Manuel Model Ekle</h3>
                            <button className="btn btn-ghost" onClick={() => setShowAddDialog(false)}><X size={20} /></button>
                        </div>
                        <div className="modal-body">
                            <div className="form-group">
                                <label>Model Adı *</label>
                                <input
                                    type="text"
                                    value={newModel.name}
                                    onChange={e => setNewModel({ ...newModel, name: e.target.value })}
                                    placeholder="Örn: XTTS Turkish v1"
                                />
                            </div>
                            <div className="form-group">
                                <label>Açıklama</label>
                                <textarea
                                    value={newModel.description}
                                    onChange={e => setNewModel({ ...newModel, description: e.target.value })}
                                    placeholder="Modelin amacı ve detayları..."
                                    rows={3}
                                />
                            </div>
                            <div className="form-group">
                                <label>Model Dosya Yolu *</label>
                                <input
                                    type="text"
                                    value={newModel.model_path}
                                    onChange={e => setNewModel({ ...newModel, model_path: e.target.value })}
                                    placeholder="Örn: C:\models\my_model\model.pth"
                                />
                                <span className="text-muted" style={{ fontSize: '0.75rem' }}>Model .pth dosyasının tam yolu</span>
                            </div>
                            <div className="form-group">
                                <label>Etiketler (virgülle ayırın)</label>
                                <input
                                    type="text"
                                    value={newModel.tags}
                                    onChange={e => setNewModel({ ...newModel, tags: e.target.value })}
                                    placeholder="Örn: turkish, pronunciation, v1"
                                />
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button className="btn btn-ghost" onClick={() => setShowAddDialog(false)}>İptal</button>
                            <button className="btn btn-primary" onClick={addManualModel}>Ekle</button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

export default ModelsPage
