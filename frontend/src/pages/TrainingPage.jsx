import { useState, useEffect, useRef } from 'react'
import { GraduationCap, Play, Square, AlertCircle, Check, Settings, Folder, Volume2, Pause, ChevronDown, ChevronUp } from 'lucide-react'

const API_BASE_URL = `http://${window.location.hostname}:5001`

function TrainingPage() {
    // Form state
    const [name, setName] = useState('')
    const [description, setDescription] = useState('')
    const [tags, setTags] = useState('')
    const [datasetPath, setDatasetPath] = useState('')
    const [metaFile, setMetaFile] = useState('metadata.csv')

    // Dataset selection
    const [folders, setFolders] = useState([])
    const [selectedFolders, setSelectedFolders] = useState(new Set())
    const [expandedFolder, setExpandedFolder] = useState(null)
    const [folderItems, setFolderItems] = useState({})
    const [playingId, setPlayingId] = useState(null)
    const audioRef = useRef(null)

    // Training params
    const [epochs, setEpochs] = useState(40)
    const [batchSize, setBatchSize] = useState(1)
    const [gradAccumSteps, setGradAccumSteps] = useState(10)
    const [learningRate, setLearningRate] = useState('5e-06')
    const [saveStep, setSaveStep] = useState(30)
    const [numSamples, setNumSamples] = useState(0)
    const [language, setLanguage] = useState('tr')
    const [speakerRef, setSpeakerRef] = useState('')

    // Training status
    const [activeModelId, setActiveModelId] = useState(null)
    const [trainingStatus, setTrainingStatus] = useState(null)
    const [trainingLog, setTrainingLog] = useState('')
    const [isStarting, setIsStarting] = useState(false)

    // History
    const [trainingHistory, setTrainingHistory] = useState([])

    // Messages
    const [error, setError] = useState(null)
    const [success, setSuccess] = useState(null)

    // Log ref for auto-scroll
    const logRef = useRef(null)

    useEffect(() => {
        loadHistory()
        loadFolders()

        // Restore active training state if a job is running (e.g. after navigating away and back)
        const checkActiveJobs = async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/training/jobs`)
                const data = await response.json()
                if (data.success && data.jobs && data.jobs.length > 0) {
                    const runningJob = data.jobs.find(j => j.is_running)
                    if (runningJob) {
                        setActiveModelId(runningJob.model_id)
                        setTrainingStatus(runningJob.status || 'training')
                        return
                    }
                }
                // Fallback: check DB for any model still in 'training' status
                const modelsRes = await fetch(`${API_BASE_URL}/api/models?status=training`)
                const modelsData = await modelsRes.json()
                if (modelsData.success && modelsData.models && modelsData.models.length > 0) {
                    const m = modelsData.models[0]
                    setActiveModelId(m.id)
                    setTrainingStatus('training')
                }
            } catch (err) {
                console.error('Active job check failed:', err)
            }
        }
        checkActiveJobs()
    }, [])

    // Poll training status when active
    useEffect(() => {
        if (!activeModelId) return
        const interval = setInterval(async () => {
            try {
                const response = await fetch(`${API_BASE_URL}/api/training/status/${activeModelId}`)
                const data = await response.json()
                if (data.success) {
                    setTrainingStatus(data.status)
                    setTrainingLog(data.training_log || '')

                    if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                        clearInterval(interval)
                        setActiveModelId(null)
                        loadHistory()
                        if (data.status === 'completed') setSuccess('🎉 Eğitim tamamlandı!')
                        if (data.status === 'failed') setError('❌ Eğitim başarısız: ' + (data.error_message || ''))
                    }
                }
            } catch (err) {
                console.error('Training status error:', err)
            }
        }, 2000)

        return () => clearInterval(interval)
    }, [activeModelId])

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight
        }
    }, [trainingLog])

    // Update numSamples when folder selection changes
    useEffect(() => {
        let total = 0
        selectedFolders.forEach(folderName => {
            const f = folders.find(x => x.name === folderName)
            if (f) total += f.file_count
        })
        setNumSamples(total)
    }, [selectedFolders, folders])

    useEffect(() => {
        if (error) { const t = setTimeout(() => setError(null), 8000); return () => clearTimeout(t) }
    }, [error])
    useEffect(() => {
        if (success) { const t = setTimeout(() => setSuccess(null), 8000); return () => clearTimeout(t) }
    }, [success])

    const loadFolders = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/folders`)
            const data = await response.json()
            if (data.success) setFolders(data.folders || [])
        } catch (err) {
            console.error('Folders yüklenemedi:', err)
        }
    }

    const loadFolderItems = async (folderName) => {
        if (folderItems[folderName]) return // already loaded
        try {
            const response = await fetch(`${API_BASE_URL}/api/items?word=${encodeURIComponent(folderName)}&status=generated&limit=200`)
            const data = await response.json()
            if (data.success) {
                setFolderItems(prev => ({ ...prev, [folderName]: data.items || [] }))
            }
        } catch (err) {
            console.error(`Items for ${folderName} yüklenemedi:`, err)
        }
    }

    const toggleFolder = (folderName) => {
        setSelectedFolders(prev => {
            const next = new Set(prev)
            if (next.has(folderName)) next.delete(folderName)
            else next.add(folderName)
            return next
        })
    }

    const toggleAllFolders = () => {
        if (selectedFolders.size === folders.length) {
            setSelectedFolders(new Set())
        } else {
            setSelectedFolders(new Set(folders.map(f => f.name)))
        }
    }

    const expandFolder = (folderName) => {
        if (expandedFolder === folderName) {
            setExpandedFolder(null)
        } else {
            setExpandedFolder(folderName)
            loadFolderItems(folderName)
        }
    }

    const playAudio = (itemId) => {
        if (playingId === itemId && audioRef.current) {
            if (audioRef.current.paused) {
                audioRef.current.play()
            } else {
                audioRef.current.pause()
                setPlayingId(null)
            }
            return
        }
        if (audioRef.current) { audioRef.current.pause(); audioRef.current = null }

        const audio = new Audio(`${API_BASE_URL}/api/audio/${itemId}/play`)
        audio.onended = () => { setPlayingId(null); audioRef.current = null }
        audio.onerror = () => { setError('Ses dosyası oynatılamadı'); setPlayingId(null); audioRef.current = null }
        audioRef.current = audio
        setPlayingId(itemId)
        audio.play().catch(() => { setError('Ses oynatılamadı'); setPlayingId(null) })
    }

    const loadHistory = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/models`)
            const data = await response.json()
            if (data.success) setTrainingHistory(data.models)
        } catch (err) {
            console.error('History yüklenemedi:', err)
        }
    }

    const startTraining = async () => {
        if (!name.trim()) { setError('Model adı gerekli'); return }
        if (selectedFolders.size === 0 && !datasetPath.trim()) {
            setError('En az bir klasör seçin veya dataset yolu girin'); return
        }

        setIsStarting(true)
        setError(null)

        try {
            // Build selected_folders array for the backend
            const foldersArray = Array.from(selectedFolders)

            const response = await fetch(`${API_BASE_URL}/api/training/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    description: description.trim(),
                    tags: tags.split(',').map(t => t.trim()).filter(Boolean),
                    dataset_path: datasetPath.trim() || 'training_output',
                    meta_file_train: metaFile.trim(),
                    selected_folders: foldersArray.length > 0 ? foldersArray : undefined,
                    training_params: {
                        epochs: parseInt(epochs),
                        batch_size: parseInt(batchSize),
                        grad_accum_steps: parseInt(gradAccumSteps),
                        learning_rate: parseFloat(learningRate),
                        save_step: parseInt(saveStep),
                        num_samples: parseInt(numSamples) || 1,
                        language,
                        speaker_reference: speakerRef || undefined
                    }
                })
            })

            const data = await response.json()
            if (data.success) {
                setActiveModelId(data.model_id)
                setTrainingStatus('training')
                setTrainingLog('')
                setSuccess(`🚀 Eğitim başlatıldı (Model ID: ${data.model_id})`)
            } else {
                setError(data.error || 'Eğitim başlatılamadı')
            }
        } catch (err) {
            setError('Eğitim başlatılamadı: ' + err.message)
        } finally {
            setIsStarting(false)
        }
    }

    const cancelTraining = async () => {
        if (!activeModelId) {
            // No active model but UI is stuck — just reset
            setTrainingStatus(null)
            setActiveModelId(null)
            setSuccess('⚠️ UI sıfırlandı')
            return
        }
        try {
            await fetch(`${API_BASE_URL}/api/training/cancel/${activeModelId}`, { method: 'POST' })
            setSuccess('⚠️ Eğitim iptal edildi')
        } catch (err) {
            setSuccess('⚠️ Eğitim iptal istendi (sunucu yeniden başlatmanız gerekebilir)')
        }
        // Always reset UI state
        setActiveModelId(null)
        setTrainingStatus(null)
    }

    const isTraining = !!activeModelId || trainingStatus === 'training' || trainingStatus === 'starting'

    return (
        <div className="page-content">
            {/* Notifications */}
            {error && <div className="notification error"><AlertCircle size={20} /><span>{error}</span></div>}
            {success && <div className="notification success"><Check size={20} /><span>{success}</span></div>}

            {/* Header */}
            <div className="page-header">
                <div className="page-header-left">
                    <GraduationCap className="header-icon" />
                    <div>
                        <h1>Model Eğitimi</h1>
                        <p>XTTS model fine-tuning parametreleri ve eğitim kontrolü</p>
                    </div>
                </div>
            </div>

            <div className="training-layout">
                {/* Left: Config Form */}
                <div className="training-form-panel">
                    <div className="card">
                        <h3 className="card-title"><Settings size={18} /> Eğitim Yapılandırması</h3>

                        {/* Model Info */}
                        <div className="form-section">
                            <h4>Model Bilgileri</h4>
                            <div className="form-group">
                                <label>Model Adı *</label>
                                <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Örn: XTTS Turkish Pronunciation v2" disabled={isTraining} />
                            </div>
                            <div className="form-group">
                                <label>Açıklama</label>
                                <textarea value={description} onChange={e => setDescription(e.target.value)} placeholder="Eğitimin amacı..." rows={2} disabled={isTraining} />
                            </div>
                            <div className="form-group">
                                <label>Etiketler (virgülle ayırın)</label>
                                <input type="text" value={tags} onChange={e => setTags(e.target.value)} placeholder="turkish, pronunciation, v2" disabled={isTraining} />
                            </div>
                        </div>

                        {/* Dataset Selection from Veri Üretimi */}
                        <div className="form-section">
                            <h4>📂 Dataset Seçimi (Veri Üretimi Klasörleri)</h4>
                            {folders.length === 0 ? (
                                <div className="dataset-empty">
                                    <p className="text-muted">Henüz oluşturulmuş veri klasörü yok.</p>
                                    <p className="text-muted" style={{ fontSize: '0.75rem' }}>
                                        Önce "Veri Üretimi" sekmesinden ses dosyaları oluşturun.
                                    </p>
                                </div>
                            ) : (
                                <>
                                    <div className="dataset-toolbar">
                                        <button className="btn btn-small btn-ghost" onClick={toggleAllFolders} disabled={isTraining}>
                                            {selectedFolders.size === folders.length ? 'Hiçbirini Seçme' : 'Tümünü Seç'}
                                        </button>
                                        <span className="text-muted" style={{ fontSize: '0.8rem' }}>
                                            {selectedFolders.size} / {folders.length} klasör seçili • {numSamples} ses dosyası
                                        </span>
                                    </div>
                                    <div className="folder-list">
                                        {folders.map(folder => (
                                            <div key={folder.name} className={`folder-item ${selectedFolders.has(folder.name) ? 'folder-selected' : ''}`}>
                                                <div className="folder-item-main">
                                                    <input
                                                        type="checkbox"
                                                        className="checkbox"
                                                        checked={selectedFolders.has(folder.name)}
                                                        onChange={() => toggleFolder(folder.name)}
                                                        disabled={isTraining}
                                                    />
                                                    <Folder size={16} style={{ color: 'var(--primary-light)', flexShrink: 0 }} />
                                                    <span className="folder-name">{folder.name}</span>
                                                    <span className="folder-count">{folder.file_count} dosya</span>
                                                    <button
                                                        className="btn btn-ghost btn-icon"
                                                        onClick={() => expandFolder(folder.name)}
                                                        title="Dosyaları göster"
                                                    >
                                                        {expandedFolder === folder.name ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                                                    </button>
                                                </div>

                                                {/* Expanded: Audio items with play buttons */}
                                                {expandedFolder === folder.name && (
                                                    <div className="folder-items-list">
                                                        {!folderItems[folder.name] ? (
                                                            <p className="text-muted" style={{ padding: '0.5rem', fontSize: '0.8rem' }}>Yükleniyor...</p>
                                                        ) : folderItems[folder.name].length === 0 ? (
                                                            <p className="text-muted" style={{ padding: '0.5rem', fontSize: '0.8rem' }}>Bu klasörde ses dosyası bulunamadı.</p>
                                                        ) : (
                                                            folderItems[folder.name].map(item => (
                                                                <div key={item.id} className="folder-audio-item">
                                                                    <button
                                                                        className={`btn-play-small ${playingId === item.id ? 'playing' : ''}`}
                                                                        onClick={() => playAudio(item.id)}
                                                                        title={playingId === item.id ? 'Durdur' : 'Dinle'}
                                                                    >
                                                                        {playingId === item.id ? <Pause size={12} /> : <Volume2 size={12} />}
                                                                    </button>
                                                                    <span className="audio-item-text" title={item.sentence}>
                                                                        {item.sentence}
                                                                    </span>
                                                                    {item.duration_seconds && (
                                                                        <span className="audio-item-duration">
                                                                            {Math.round(item.duration_seconds)}s
                                                                        </span>
                                                                    )}
                                                                </div>
                                                            ))
                                                        )}
                                                    </div>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </>
                            )}

                            {/* Manual dataset path (fallback) */}
                            <div style={{ marginTop: '0.75rem' }}>
                                <label style={{ fontSize: '0.75rem', color: 'var(--text-dim)', display: 'block', marginBottom: '0.25rem' }}>
                                    Veya manuel yol girin:
                                </label>
                                <input
                                    type="text"
                                    value={datasetPath}
                                    onChange={e => setDatasetPath(e.target.value)}
                                    placeholder="C:\path\to\dataset\"
                                    disabled={isTraining}
                                    style={{ width: '100%', padding: '0.5rem 0.75rem', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text)', fontSize: '0.85rem', outline: 'none' }}
                                />
                            </div>
                        </div>

                        {/* Training Parameters */}
                        <div className="form-section">
                            <h4>Eğitim Parametreleri</h4>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Epoch Sayısı</label>
                                    <div className="param-input-group">
                                        <input
                                            type="range" min={1} max={200} value={epochs}
                                            onChange={e => setEpochs(parseInt(e.target.value))}
                                            disabled={isTraining} className="param-slider"
                                        />
                                        <input type="number" value={epochs} onChange={e => setEpochs(e.target.value)} min={1} max={500} disabled={isTraining} className="param-number" />
                                    </div>
                                </div>
                                <div className="form-group">
                                    <label>Batch Size</label>
                                    <select value={batchSize} onChange={e => setBatchSize(e.target.value)} disabled={isTraining}>
                                        <option value={1}>1</option>
                                        <option value={2}>2</option>
                                        <option value={4}>4</option>
                                        <option value={8}>8</option>
                                    </select>
                                </div>
                            </div>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Gradient Accumulation Steps</label>
                                    <input type="number" value={gradAccumSteps} onChange={e => setGradAccumSteps(e.target.value)} min={1} disabled={isTraining} />
                                </div>
                                <div className="form-group">
                                    <label>Learning Rate</label>
                                    <input type="text" value={learningRate} onChange={e => setLearningRate(e.target.value)} disabled={isTraining} />
                                </div>
                            </div>
                            <div className="form-row">
                                <div className="form-group">
                                    <label>Checkpoint Kaydetme (her N step)</label>
                                    <input type="number" value={saveStep} onChange={e => setSaveStep(e.target.value)} min={1} disabled={isTraining} />
                                </div>
                                <div className="form-group">
                                    <label>Dil</label>
                                    <select value={language} onChange={e => setLanguage(e.target.value)} disabled={isTraining}>
                                        <option value="tr">Türkçe (tr)</option>
                                        <option value="en">English (en)</option>
                                        <option value="de">Deutsch (de)</option>
                                        <option value="fr">Français (fr)</option>
                                        <option value="es">Español (es)</option>
                                        <option value="it">Italiano (it)</option>
                                        <option value="pt">Português (pt)</option>
                                        <option value="ru">Русский (ru)</option>
                                        <option value="ar">العربية (ar)</option>
                                        <option value="zh-cn">中文 (zh-cn)</option>
                                        <option value="ja">日本語 (ja)</option>
                                        <option value="ko">한국어 (ko)</option>
                                    </select>
                                </div>
                            </div>
                            <div className="form-group">
                                <label>Speaker Reference WAV (opsiyonel)</label>
                                <input type="text" value={speakerRef} onChange={e => setSpeakerRef(e.target.value)} placeholder="C:\path\to\clone.wav" disabled={isTraining} />
                            </div>
                        </div>

                        {/* Calculated Info */}
                        <div className="training-summary-box">
                            <h4>📊 Hesaplanan Değerler</h4>
                            <div className="summary-grid">
                                <span>Seçilen Örnekler:</span>
                                <strong>{numSamples}</strong>
                                <span>Effective Batch Size:</span>
                                <strong>{batchSize * gradAccumSteps}</strong>
                                <span>Steps per Epoch:</span>
                                <strong>{Math.max(1, Math.floor((numSamples || 1) / (batchSize * gradAccumSteps)))}</strong>
                                <span>Total Steps:</span>
                                <strong>{epochs * Math.max(1, Math.floor((numSamples || 1) / (batchSize * gradAccumSteps)))}</strong>
                            </div>
                        </div>

                        {/* Action Buttons */}
                        <div className="training-actions">
                            {!isTraining ? (
                                <button className="btn btn-primary btn-large" onClick={startTraining} disabled={isStarting}>
                                    <Play size={18} />
                                    {isStarting ? 'Başlatılıyor...' : 'Eğitimi Başlat'}
                                </button>
                            ) : (
                                <button className="btn btn-danger btn-large" onClick={cancelTraining}>
                                    <Square size={18} /> Eğitimi İptal Et
                                </button>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right: Live Log & Status */}
                <div className="training-log-panel">
                    <div className="card log-card">
                        <div className="log-header">
                            <h3>📋 Eğitim Konsolu</h3>
                            {isTraining && <span className="pulse-dot" />}
                            {trainingStatus && (
                                <span className={`status-badge status-${trainingStatus}`}>
                                    {trainingStatus === 'training' ? '⏳ Eğitimde' : trainingStatus === 'completed' ? '✅ Tamamlandı' : trainingStatus === 'failed' ? '❌ Başarısız' : trainingStatus}
                                </span>
                            )}
                        </div>
                        <div className="training-log" ref={logRef}>
                            {trainingLog ? (
                                <pre>{trainingLog}</pre>
                            ) : (
                                <div className="log-empty">
                                    <p>Eğitim başlatıldığında konsol çıktıları burada görünecek...</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Training History */}
                    <div className="card" style={{ marginTop: '1rem' }}>
                        <h3 className="card-title">📜 Eğitim Geçmişi</h3>
                        {trainingHistory.length === 0 ? (
                            <p className="text-muted">Henüz eğitim kaydı yok.</p>
                        ) : (
                            <div className="history-table-wrapper">
                                <table className="history-table">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Durum</th>
                                            <th>Epoch</th>
                                            <th>Tarih</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {trainingHistory.slice(0, 10).map(m => (
                                            <tr key={m.id}>
                                                <td>
                                                    <strong>{m.name}</strong>
                                                    {(m.tags || []).length > 0 && (
                                                        <div className="mini-tags">
                                                            {m.tags.map((t, i) => <span key={i} className="mini-tag">{t}</span>)}
                                                        </div>
                                                    )}
                                                </td>
                                                <td>
                                                    <span className={`status-badge status-${m.status}`}>
                                                        {m.status === 'completed' ? '✅' : m.status === 'training' ? '⏳' : m.status === 'failed' ? '❌' : '⏸️'} {m.status}
                                                    </span>
                                                </td>
                                                <td>{m.training_params?.epochs || '-'}</td>
                                                <td>{new Date(m.created_at + (m.created_at?.endsWith('Z') ? '' : 'Z')).toLocaleDateString('tr-TR')}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default TrainingPage
