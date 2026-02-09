import { useState, useEffect, useRef } from 'react'
import { Wand2, Volume2, RefreshCw, Check, AlertCircle, Plus, Download, Settings, Folder, FolderArchive, Trash2, Cpu } from 'lucide-react'
import './App.css'

const API_BASE_URL = `http://${window.location.hostname}:5001`

function App() {
  // State
  const [word, setWord] = useState('')
  const [sentenceCount, setSentenceCount] = useState(5)
  const [sentences, setSentences] = useState([])
  const [selectedSentences, setSelectedSentences] = useState(new Set())
  const [audioItems, setAudioItems] = useState([])
  const [playingId, setPlayingId] = useState(null)
  const [stats, setStats] = useState({})

  // Folder management
  const [folders, setFolders] = useState([])
  const [selectedFolders, setSelectedFolders] = useState(new Set())
  const [isDownloadingFolders, setIsDownloadingFolders] = useState(false)

  // LLM Provider settings
  const [llmProvider, setLlmProvider] = useState('openai')
  const [ollamaModels, setOllamaModels] = useState([])
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('llama3.1:8b')
  const [ollamaAvailable, setOllamaAvailable] = useState(false)

  // TTS Parameters
  const [voices, setVoices] = useState({})
  const [voice, setVoice] = useState('tr-TR-Wavenet-D')
  const [speakingRate, setSpeakingRate] = useState(1.0)
  const [pitch, setPitch] = useState(0.0)
  const [volumeGainDb, setVolumeGainDb] = useState(0.0)

  // Loading states
  const [isGeneratingSentences, setIsGeneratingSentences] = useState(false)
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false)
  const [isExporting, setIsExporting] = useState(false)

  // Messages
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  // Custom sentence input
  const [customSentence, setCustomSentence] = useState('')

  // Editing state for sentences
  const [editingSentenceId, setEditingSentenceId] = useState(null)

  // Audio ref
  const audioRef = useRef(null)

  // Load stats on mount
  useEffect(() => {
    loadStats()
    loadItems()
    loadVoices()
    loadFolders()
    loadLLMConfig()
  }, [])

  // Clear messages after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => setError(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [error])

  useEffect(() => {
    if (success) {
      const timer = setTimeout(() => setSuccess(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [success])

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/stats`)
      const data = await response.json()
      if (data.success) {
        setStats(data.stats)
      }
    } catch (err) {
      console.error('Stats yüklenemedi:', err)
    }
  }

  const loadItems = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/items?status=generated`)
      const data = await response.json()
      if (data.success) {
        setAudioItems(data.items)
      }
    } catch (err) {
      console.error('Items yüklenemedi:', err)
    }
  }

  const loadVoices = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/voices`)
      const data = await response.json()
      if (data.success) {
        setVoices(data.voices)
        // Set default if current selection not in list
        if (!Object.keys(data.voices).includes(voice)) {
          setVoice(Object.keys(data.voices)[0])
        }
      }
    } catch (err) {
      console.error('Sesler yüklenemedi:', err)
    }
  }

  const loadFolders = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/folders`)
      const data = await response.json()
      if (data.success) {
        setFolders(data.folders)
      }
    } catch (err) {
      console.error('Klasörler yüklenemedi:', err)
    }
  }

  const loadLLMConfig = async () => {
    try {
      // Load current config
      const configResponse = await fetch(`${API_BASE_URL}/api/llm/config`)
      const configData = await configResponse.json()
      if (configData.success) {
        setLlmProvider(configData.config.provider)
        setOllamaAvailable(configData.config.ollama_available)
      }

      // Load available Ollama models
      const modelsResponse = await fetch(`${API_BASE_URL}/api/llm/models`)
      const modelsData = await modelsResponse.json()
      if (modelsData.success && modelsData.models.length > 0) {
        setOllamaModels(modelsData.models)
        // Set to first available model if current selection not available
        const configModel = configData?.config?.ollama_model || ''
        if (modelsData.models.includes(configModel)) {
          setSelectedOllamaModel(configModel)
        } else {
          setSelectedOllamaModel(modelsData.models[0])
        }
      }
    } catch (err) {
      console.error('LLM config yüklenemedi:', err)
    }
  }

  const switchLLMProvider = async (provider, model = null) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/llm/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, model })
      })
      const data = await response.json()
      if (data.success) {
        setLlmProvider(data.config.provider)
        if (model) setSelectedOllamaModel(model)
        setSuccess(`✅ LLM provider: ${provider}${model ? ` (${model})` : ''}`)
      }
    } catch (err) {
      setError('LLM provider değiştirilemedi: ' + err.message)
    }
  }

  const toggleFolder = (folderName) => {
    setSelectedFolders(prev => {
      const newSet = new Set(prev)
      if (newSet.has(folderName)) {
        newSet.delete(folderName)
      } else {
        newSet.add(folderName)
      }
      return newSet
    })
  }

  const selectAllFolders = () => setSelectedFolders(new Set(folders.map(f => f.name)))
  const deselectAllFolders = () => setSelectedFolders(new Set())

  const downloadSelectedFolders = async () => {
    if (selectedFolders.size === 0) {
      setError('Lütfen en az bir klasör seçin')
      return
    }

    setIsDownloadingFolders(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/folders/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          folders: Array.from(selectedFolders)
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'İndirme başarısız')
      }

      // Download the ZIP file
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `training_data_${Date.now()}.zip`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      setSuccess(`✅ ${selectedFolders.size} klasör ZIP olarak indirildi`)

    } catch (err) {
      setError(err.message)
    } finally {
      setIsDownloadingFolders(false)
    }
  }

  const deleteFolder = async (folderName) => {
    if (!confirm(`"${folderName}" klasörünü ve içindeki tüm dosyaları silmek istediğinize emin misiniz?`)) {
      return
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/folders/${encodeURIComponent(folderName)}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Silme başarısız')
      }

      const data = await response.json()
      setSuccess(`✅ "${folderName}" klasörü silindi (${data.files_deleted} dosya)`)

      // Remove from selected if it was selected
      setSelectedFolders(prev => {
        const newSet = new Set(prev)
        newSet.delete(folderName)
        return newSet
      })

      // Reload folders and items
      loadFolders()
      loadItems()
      loadStats()

    } catch (err) {
      setError(err.message)
    }
  }

  const downloadFolder = async (folderName) => {
    try {
      // Create a hidden link to trigger the download directly
      // This is cleaner than fetching a blob for simple file downloads
      const link = document.createElement('a')
      link.href = `${API_BASE_URL}/api/folders/${encodeURIComponent(folderName)}/download`
      link.download = `${folderName}_${Date.now()}.zip` // This might be overridden by Content-Disposition
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)

      setSuccess(`✅ "${folderName}" klasörü indiriliyor...`)
    } catch (err) {
      setError('İndirme başlatılamadı: ' + err.message)
    }
  }

  const generateSentences = async () => {
    if (!word.trim()) {
      setError('Lütfen bir kelime girin')
      return
    }

    setIsGeneratingSentences(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-sentences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word: word.trim(),
          count: sentenceCount,
          provider: llmProvider,
          model: llmProvider === 'ollama' ? selectedOllamaModel : undefined
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Cümleler oluşturulamadı')
      }

      const data = await response.json()
      setSentences(data.sentences.map((s, i) => ({ id: i, text: s })))
      setSelectedSentences(new Set(data.sentences.map((_, i) => i)))
      setSuccess(`✅ ${data.count} cümle oluşturuldu`)

    } catch (err) {
      setError(err.message)
    } finally {
      setIsGeneratingSentences(false)
    }
  }

  const updateSentence = (id, newText) => {
    setSentences(prev => prev.map(s => s.id === id ? { ...s, text: newText } : s))
  }

  const toggleSentence = (id) => {
    setSelectedSentences(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }

  const selectAll = () => setSelectedSentences(new Set(sentences.map(s => s.id)))
  const deselectAll = () => setSelectedSentences(new Set())
  const clearAllSentences = () => {
    setSentences([])
    setSelectedSentences(new Set())
  }

  const clearAllAudio = async () => {
    if (audioItems.length === 0) return

    try {
      const itemIds = audioItems.map(item => item.id)
      const response = await fetch(`${API_BASE_URL}/api/items/bulk-delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ item_ids: itemIds })
      })

      if (!response.ok) throw new Error('Toplu silme başarısız')

      const data = await response.json()
      setAudioItems([])
      loadStats()
      loadFolders()
      setSuccess(`✅ ${data.deleted_count} ses dosyası silindi`)
    } catch (err) {
      setError(err.message)
    }
  }

  const addCustomSentence = () => {
    if (!customSentence.trim()) return
    const newId = sentences.length > 0 ? Math.max(...sentences.map(s => s.id)) + 1 : 0
    setSentences(prev => [...prev, { id: newId, text: customSentence.trim() }])
    setSelectedSentences(prev => new Set([...prev, newId]))
    setCustomSentence('')
  }

  const startEditingSentence = (id) => {
    setEditingSentenceId(id)
  }

  const stopEditingSentence = () => {
    setEditingSentenceId(null)
  }

  const deleteSentence = (id) => {
    setSentences(prev => prev.filter(s => s.id !== id))
    setSelectedSentences(prev => {
      const newSet = new Set(prev)
      newSet.delete(id)
      return newSet
    })
  }

  const downloadAudio = (itemId, sentence) => {
    const link = document.createElement('a')
    link.href = `${API_BASE_URL}/api/audio/${itemId}/download`
    link.download = `${sentence.substring(0, 30).replace(/[^a-zA-Z0-9ğüşıöçĞÜŞİÖÇ ]/g, '')}.wav`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const downloadAllAudio = async () => {
    if (audioItems.length === 0) {
      setError('İndirilecek ses dosyası yok')
      return
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/audio/download-all`)

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'İndirme başarısız')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `all_audio_${Date.now()}.zip`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)

      setSuccess(`✅ ${audioItems.length} ses dosyası ZIP olarak indirildi`)
    } catch (err) {
      setError(err.message)
    }
  }

  const generateAudio = async () => {
    const selectedData = sentences
      .filter(s => selectedSentences.has(s.id))
      .map(s => ({ text: s.text, word: word }))

    if (selectedData.length === 0) {
      setError('Lütfen en az bir cümle seçin')
      return
    }

    setIsGeneratingAudio(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-audio`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sentences: selectedData,
          voice,
          speakingRate,
          pitch,
          volumeGainDb
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Ses dosyaları oluşturulamadı')
      }

      const data = await response.json()

      if (data.failed > 0) {
        // Collect errors
        const errorList = data.files.filter(f => !f.success).map(f => f.error)
        const uniqueErrors = [...new Set(errorList)].join(', ')
        setError(`⚠️ ${data.generated}/${data.total} başarılı. Hata: ${uniqueErrors}`)
      } else {
        setSuccess(`✅ ${data.generated}/${data.total} ses dosyası oluşturuldu (Hız: ${speakingRate}, Vol: ${volumeGainDb}dB)`)
      }

      loadStats()
      loadItems()
      loadFolders()

      // Mark successful sentences as generated
      // We need to map back which sentences succeeded
      // The backend returns results array which matches the order of 'selectedData' sent
      // selectedData came from sentences.filter(s => selectedSentences.has(s.id))

      // Let's just iterate data.files and find matching sentences by text/word if possible
      // Or safer: since we know successful count, wait...
      // The backend returns "files" array with "text" and "success"

      setSentences(prev => prev.map(s => {
        if (!selectedSentences.has(s.id)) return s // wasn't in this batch

        // Was it successful?
        const result = data.files.find(f => f.text === s.text)
        if (result && result.success) {
          return { ...s, isGenerated: true }
        }
        return s
      }))

      setSelectedSentences(new Set())

    } catch (err) {
      setError(err.message)
    } finally {
      setIsGeneratingAudio(false)
    }
  }

  const playAudio = (itemId) => {
    if (playingId === itemId && audioRef.current) {
      if (audioRef.current.paused) {
        audioRef.current.play()
      } else {
        audioRef.current.pause()
      }
      return
    }

    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
    }

    const audio = new Audio(`${API_BASE_URL}/api/audio/${itemId}/play`)
    audio.onended = () => {
      setPlayingId(null)
      audioRef.current = null
    }
    audio.onerror = () => {
      setError('Ses oynatılamadı')
      setPlayingId(null)
      audioRef.current = null
    }

    audioRef.current = audio
    setPlayingId(itemId)
    audio.play().catch(err => {
      setError('Ses oynatılamadı: ' + err.message)
      setPlayingId(null)
    })
  }

  const deleteItem = async (itemId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/items/${itemId}`, {
        method: 'DELETE'
      })

      if (!response.ok) throw new Error('Silinemedi')

      setAudioItems(prev => prev.filter(item => item.id !== itemId))
      setSuccess('✅ Ses dosyası silindi')
      loadStats()
      loadFolders()

    } catch (err) {
      setError(err.message)
    }
  }

  const exportData = async () => {
    setIsExporting(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Export başarısız')
      }

      const data = await response.json()
      setSuccess(`✅ ${data.item_count} öğe export edildi: ${data.metadata_path}`)
      loadStats()
      loadItems()

    } catch (err) {
      setError(err.message)
    } finally {
      setIsExporting(false)
    }
  }

  const deleteSelectedFolders = async () => {
    if (selectedFolders.size === 0) {
      setError('Lütfen silinecek en az bir klasör seçin')
      return
    }

    if (!confirm(`${selectedFolders.size} adet klasörü ve içindeki tüm dosyaları silmek istediğinize emin misiniz?`)) {
      return
    }

    // Reuse existing loading state or add a new one? 
    // Let's use isDownloadingFolders as a general "isProcessingFolders" or add a new one.
    // For simplicity, I'll reuse isDownloadingFolders to block UI or add a quick local state if preferred.
    // Actually, locking UI is good. But let's add a proper state or just use a flag.
    // I will add a new state `isDeletingFolders` in the next edit or just reuse the logic pattern.
    // Let's check available states. I see `isDownloadingFolders`. 
    // I'll add `isDeletingFolders` to state first.
    // Wait, I can't add state easily in a replace_file_content if I don't target the top of the file.
    // I will use `isDownloadingFolders` for now as "isProcessing" to disable buttons, OR just run it.
    // Actually, I should probably add the state constant at the top first. 
    // But to save steps, I will just proceed without a specific loading spinner for delete, 
    // or reuse `setIsDownloadingFolders` (not ideal naming but functional for disabling).
    // Better: I'll use `isDownloadingFolders` to disable the buttons during operation to prevent double clicks.

    try {
      const response = await fetch(`${API_BASE_URL}/api/folders/bulk-delete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          folders: Array.from(selectedFolders)
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || 'Silme başarısız')
      }

      const data = await response.json()
      setSuccess(`✅ ${data.count} klasör silindi`)

      // Clear selection
      setSelectedFolders(new Set())

      // Reload
      loadFolders()
      loadItems()
      loadStats()

    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="app">
      <div className="container">
        {/* Header */}
        <header className="header">
          <div className="header-content">
            <Wand2 className="header-icon" />
            <div>
              <h1>Training Data Generator</h1>
              <p>TTS Telaffuz Eğitimi için Sentetik Veri Oluşturucu</p>
            </div>
          </div>
        </header>

        {/* Notifications */}
        {error && (
          <div className="notification error">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}
        {success && (
          <div className="notification success">
            <Check size={20} />
            <span>{success}</span>
          </div>
        )}

        {/* Phases Grid - All 3 phases side by side */}
        <div className="phases-grid">
          {/* Step 1: Word Input */}
          <section className="card phase-card">
            <div className="step-header">
              <span className="step-number">1</span>
              <h2>Kelime Girin</h2>
            </div>
            <p className="step-description">Yanlış telaffuz edilen kelimeyi girin.</p>

            <div className="input-column">
              <div className="input-row-inline">
                <input
                  type="text"
                  value={word}
                  onChange={(e) => setWord(e.target.value)}
                  placeholder="Kelime..."
                  className="input-text"
                  onKeyDown={(e) => e.key === 'Enter' && generateSentences()}
                />
                <input
                  type="number"
                  value={sentenceCount}
                  onChange={(e) => setSentenceCount(Math.max(1, parseInt(e.target.value) || 1))}
                  min="1"
                  className="input-count"
                  placeholder="5"
                />
              </div>
              <button
                onClick={generateSentences}
                disabled={isGeneratingSentences || !word.trim()}
                className="btn btn-primary btn-full"
              >
                {isGeneratingSentences ? (
                  <RefreshCw className="spin" size={20} />
                ) : (
                  <Wand2 size={20} />
                )}
                <span>Cümle Oluştur</span>
              </button>

              {/* LLM Provider Toggle */}
              <div className="llm-provider-toggle">
                <div className="provider-switch">
                  <button
                    className={`provider-btn ${llmProvider === 'openai' ? 'active' : ''}`}
                    onClick={() => switchLLMProvider('openai')}
                  >
                    OpenAI
                  </button>
                  <button
                    className={`provider-btn ${llmProvider === 'ollama' ? 'active' : ''} ${!ollamaAvailable ? 'disabled' : ''}`}
                    onClick={() => ollamaAvailable && switchLLMProvider('ollama', selectedOllamaModel)}
                    title={!ollamaAvailable ? 'Ollama çalışmıyor' : ''}
                  >
                    <Cpu size={14} />
                    Ollama
                  </button>
                </div>

                {llmProvider === 'ollama' && ollamaModels.length > 0 && (
                  <select
                    value={selectedOllamaModel}
                    onChange={(e) => switchLLMProvider('ollama', e.target.value)}
                    className="ollama-model-select"
                  >
                    {ollamaModels.map(model => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                )}

                {llmProvider === 'ollama' && !ollamaAvailable && (
                  <span className="ollama-warning">⚠️ Ollama çalışmıyor</span>
                )}
              </div>
            </div>
          </section>

          {/* Step 2: Sentence Review */}
          <section className={`card phase-card ${sentences.length === 0 ? 'phase-inactive' : ''}`}>
            <div className="step-header">
              <span className="step-number">2</span>
              <h2>Cümleleri Düzenleyin</h2>
              {sentences.length > 0 && (
                <div className="step-actions">
                  <button onClick={selectAll} className="btn btn-small">Tümünü Seç</button>
                  <button onClick={deselectAll} className="btn btn-small">Seçimi Kaldır</button>
                  <button onClick={clearAllSentences} className="btn btn-small btn-danger-small">Tümünü Sil</button>
                </div>
              )}
            </div>

            {sentences.length === 0 ? (
              <div className="phase-empty">
                <Wand2 size={32} className="empty-icon" />
                <p>Henüz cümle yok</p>
                <span>Yukarıdan kelime girerek cümle oluşturun veya kendiniz ekleyin</span>
                <div className="custom-sentence-input" style={{ marginTop: '1rem', width: '100%' }}>
                  <input
                    type="text"
                    value={customSentence}
                    onChange={(e) => setCustomSentence(e.target.value)}
                    placeholder="Kendi cümlenizi ekleyin..."
                    className="input-text"
                    onKeyDown={(e) => e.key === 'Enter' && addCustomSentence()}
                  />
                  <button
                    onClick={addCustomSentence}
                    disabled={!customSentence.trim()}
                    className="btn btn-add"
                  >
                    <Plus size={18} />
                    <span>Ekle</span>
                  </button>
                </div>
              </div>
            ) : (
              <>
                {/* Custom sentence input */}
                <div className="custom-sentence-input">
                  <input
                    type="text"
                    value={customSentence}
                    onChange={(e) => setCustomSentence(e.target.value)}
                    placeholder="Kendi cümlenizi ekleyin..."
                    className="input-text"
                    onKeyDown={(e) => e.key === 'Enter' && addCustomSentence()}
                  />
                  <button
                    onClick={addCustomSentence}
                    disabled={!customSentence.trim()}
                    className="btn btn-add"
                  >
                    <Plus size={18} />
                    <span>Ekle</span>
                  </button>
                </div>

                <div className="sentence-list">
                  {sentences.map((sentence) => (
                    <div
                      key={sentence.id}
                      className={`sentence-item ${selectedSentences.has(sentence.id) ? 'selected' : ''} ${sentence.isGenerated ? 'generated' : ''}`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedSentences.has(sentence.id)}
                        onChange={() => toggleSentence(sentence.id)}
                        className="checkbox"
                      />
                      {editingSentenceId === sentence.id ? (
                        <input
                          type="text"
                          value={sentence.text}
                          onChange={(e) => updateSentence(sentence.id, e.target.value)}
                          onBlur={stopEditingSentence}
                          onKeyDown={(e) => e.key === 'Enter' && stopEditingSentence()}
                          className="sentence-input editing"
                          autoFocus
                        />
                      ) : (
                        <span className="sentence-text">{sentence.text}</span>
                      )}
                      <button
                        onClick={() => startEditingSentence(sentence.id)}
                        className="btn-edit"
                        title="Düzenle"
                      >
                        <span className="icon">✎</span>
                      </button>
                      <button
                        onClick={() => deleteSentence(sentence.id)}
                        className="btn-sentence-delete"
                        title="Sil"
                      >
                        <span className="icon">×</span>
                      </button>
                    </div>
                  ))}
                </div>

                <div className="tts-settings-panel">
                  <div className="settings-header">
                    <h3><Settings size={16} /> TTS Ayarları</h3>
                  </div>

                  <div className="settings-grid">
                    <div className="setting-item">
                      <label>Ses Modeli</label>
                      <select value={voice} onChange={(e) => setVoice(e.target.value)} className="select-input">
                        {Object.entries(voices).map(([key, name]) => (
                          <option key={key} value={key}>{name}</option>
                        ))}
                      </select>
                    </div>

                    <div className="setting-item">
                      <label>Hız: {speakingRate}x</label>
                      <input
                        type="range"
                        min="0.25"
                        max="4.0"
                        step="0.25"
                        value={speakingRate}
                        onChange={(e) => setSpeakingRate(parseFloat(e.target.value))}
                        className="range-input"
                      />
                    </div>

                    <div className="setting-item">
                      <label>Pitch: {pitch}</label>
                      <input
                        type="range"
                        min="-20"
                        max="20"
                        step="1"
                        value={pitch}
                        onChange={(e) => setPitch(parseFloat(e.target.value))}
                        className="range-input"
                      />
                    </div>

                    <div className="setting-item">
                      <label>Volume: {volumeGainDb} dB</label>
                      <input
                        type="range"
                        min="-10"
                        max="10.0"
                        step="1"
                        value={volumeGainDb}
                        onChange={(e) => setVolumeGainDb(parseFloat(e.target.value))}
                        className="range-input"
                      />
                    </div>
                  </div>
                </div>

                <div className="card-footer">
                  <button
                    onClick={generateAudio}
                    disabled={isGeneratingAudio || selectedSentences.size === 0}
                    className="btn btn-success"
                  >
                    {isGeneratingAudio ? (
                      <RefreshCw className="spin" size={20} />
                    ) : (
                      <Volume2 size={20} />
                    )}
                    <span>Ses Oluştur ({selectedSentences.size} cümle)</span>
                  </button>
                </div>
              </>
            )}
          </section>

          {/* Step 3: Audio Preview */}
          <section className={`card phase-card ${audioItems.length === 0 ? 'phase-inactive' : ''}`}>
            <div className="step-header">
              <span className="step-number">3</span>
              <h2>Ses Dosyalarını Dinleyin</h2>
              {audioItems.length > 0 && (
                <div className="step-actions">
                  <button onClick={downloadAllAudio} className="btn btn-small btn-success-small">Tümünü İndir</button>
                  <button onClick={clearAllAudio} className="btn btn-small btn-danger-small">Tümünü Sil</button>
                </div>
              )}
            </div>

            {audioItems.length === 0 ? (
              <div className="phase-empty">
                <Volume2 size={32} className="empty-icon" />
                <p>Henüz ses dosyası yok</p>
                <span>Cümle seçip ses oluşturun</span>
              </div>
            ) : (
              <>
                <div className="audio-list">
                  {audioItems.map((item) => (
                    <div key={item.id} className="audio-item">
                      <button
                        onClick={() => playAudio(item.id)}
                        className={`btn-play ${playingId === item.id ? 'playing' : ''}`}
                        title={playingId === item.id ? 'Durdur' : 'Oynat'}
                      >
                        <span className="icon">{playingId === item.id ? '‖' : '▶'}</span>
                      </button>
                      <div className="audio-info">
                        <p className="audio-text">{item.sentence}</p>
                        <span className="audio-meta">{item.duration_seconds?.toFixed(1)}s • {item.word}</span>
                      </div>
                      <button
                        onClick={() => downloadAudio(item.id, item.sentence)}
                        className="btn-download"
                        title="İndir"
                      >
                        <span className="icon">↓</span>
                      </button>
                      <button
                        onClick={() => deleteItem(item.id)}
                        className="btn-delete"
                        title="Sil"
                      >
                        <span className="icon">×</span>
                      </button>
                    </div>
                  ))}
                </div>

                <div className="card-footer">
                  <button
                    onClick={exportData}
                    disabled={isExporting}
                    className="btn btn-export"
                  >
                    {isExporting ? (
                      <RefreshCw className="spin" size={20} />
                    ) : (
                      <Download size={20} />
                    )}
                    <span>Export (metadata.csv)</span>
                  </button>
                </div>
              </>
            )}
          </section>
        </div>

        {/* Folder Management Panel */}
        {folders.length > 0 && (
          <section className="card folder-panel">
            <div className="folder-header">
              <div className="folder-title">
                <FolderArchive size={20} />
                <h3>Kelime Klasörleri</h3>
                <span className="folder-count">({folders.length} klasör)</span>
              </div>
              <div className="folder-actions">
                <button onClick={selectAllFolders} className="btn btn-small">Tümünü Seç</button>
                <button onClick={deselectAllFolders} className="btn btn-small">Seçimi Kaldır</button>
                <button
                  onClick={downloadSelectedFolders}
                  disabled={isDownloadingFolders || selectedFolders.size === 0}
                  className="btn btn-small btn-success-small"
                >
                  {isDownloadingFolders ? (
                    <RefreshCw className="spin" size={14} />
                  ) : (
                    <Download size={14} />
                  )}
                  <span>Seçilenleri İndir ({selectedFolders.size})</span>
                </button>
                <button
                  onClick={deleteSelectedFolders}
                  disabled={selectedFolders.size === 0}
                  className="btn btn-small btn-danger-small"
                >
                  <Trash2 size={14} />
                  <span>Seçilenleri Sil ({selectedFolders.size})</span>
                </button>
              </div>
            </div>
            <div className="folder-grid">
              {folders.map((folder) => (
                <div
                  key={folder.name}
                  className={`folder-item ${selectedFolders.has(folder.name) ? 'selected' : ''}`}
                  onClick={() => toggleFolder(folder.name)}
                >
                  <input
                    type="checkbox"
                    checked={selectedFolders.has(folder.name)}
                    onChange={() => toggleFolder(folder.name)}
                    className="checkbox"
                    onClick={(e) => e.stopPropagation()}
                  />
                  <Folder size={18} className="folder-icon" />
                  <span className="folder-name">{folder.name}</span>
                  <span className="folder-file-count">{folder.file_count} dosya</span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      downloadFolder(folder.name)
                    }}
                    className="btn-download"
                    title="Klasörü İndir"
                  >
                    <span className="icon">↓</span>
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      deleteFolder(folder.name)
                    }}
                    className="btn-folder-delete"
                    title="Klasörü Sil"
                  >
                    <span className="icon">×</span>
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Help Bar - Compact */}
        <div className="help-bar">
          <span className="help-title">Nasıl Çalışır:</span>
          <span className="help-step"><strong>1.</strong> Kelime girin</span>
          <span className="help-divider">→</span>
          <span className="help-step"><strong>2.</strong> Cümleler oluşturun</span>
          <span className="help-divider">→</span>
          <span className="help-step"><strong>3.</strong> Düzenleyin</span>
          <span className="help-divider">→</span>
          <span className="help-step"><strong>4.</strong> Ses oluşturun</span>
          <span className="help-divider">→</span>
          <span className="help-step"><strong>5.</strong> Dinleyin & Export</span>
        </div>

      </div>
    </div>
  )
}

export default App
