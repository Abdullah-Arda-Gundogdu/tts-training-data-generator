import { useState, useEffect, useRef } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { Wand2, Volume2, RefreshCw, Check, AlertCircle, Plus, Download, Settings, Folder, FolderArchive, Trash2, Cpu, AlertTriangle, X, Brain, GraduationCap, Key, Search, Info, ChevronDown, ChevronUp, ArrowRight, FileText, Upload, Layers, ChevronRight } from 'lucide-react'
import ModelsPage from './pages/ModelsPage'
import TrainingPage from './pages/TrainingPage'
import SettingsPage from './pages/SettingsPage'
import './App.css'

const API_BASE_URL = `http://${window.location.hostname}:5001`

const DEFAULT_SYSTEM_PROMPT = [
  'Sen bir Türkçe dil uzmanısın. Bir TTS (Metin-Konuşma) modeli eğitmek için, teknik veya spesifik kelimeleri içeren senkronize Türkçe cümleler üretiyorsun.',
  '',
  'Tam olarak {count} adet anlamlı, açıklayıcı ve doğal Türkçe cümle üret. Her cümle "{word}" ifadesini içermelidir.',
  '',
  'KRİTİK KURALLAR:',
  '- "{word}" ifadesini BİREBİR AYNI şekilde kullan — büyük/küçük harf, tire, numara ve yazılışı koru. Değiştirme.',
  '- İfade tek kelime veya birden fazla kelime olabilir, bütün olarak kullan.',
  '- Her cümle 25-30 kelime uzunluğunda olmalıdır. kısa veya kelime eksikliği olan cümleler ÜRETME.',
  '- {word} ifadesi eğer bir kısaltma, havacılık, teknik veya mühendislik terimiyse, cümleyi sanki bir kullanım kılavuzundan, eğitim kitabından veya teknik bir prosedürden alınmış mantıklı bir bağlamda kur.',
  '- Sadece "Şu {word} kelimesi şöyledir" gibi basit, anlamsız cümleler kurma. Kelimenin gerçek hayattaki veya teknik alandaki işlevini yansıtan cümleler kur.',
  '- "{word}" ifadesini cümlenin SADECE SONUNDA kullan.',
  '- Cümleler sesli okunmaya uygun ve akıcı olmalıdır.',
  '- HER KELİME EN AZ 20 KELİME OLMALI',
  '',
  'ZORUNLU: Sadece geçerli bir JSON dizisi döndür. Açıklama, pre-text, markdown (```json gibi) veya başka bir şey YAZMA. Sadece listeyi dön.',
  '',
  'Format:',
  '["Birinci cümle burada.", "İkinci cümle burada.", "Üçüncü cümle burada."]'
].join('\n')

function App() {
  // State — load from localStorage where applicable
  const [word, setWord] = useState(() => localStorage.getItem('tts_word') || '')
  const [sentenceCount, setSentenceCount] = useState(() => parseInt(localStorage.getItem('tts_sentenceCount')) || 5)
  const [sentences, setSentences] = useState(() => {
    try { return JSON.parse(localStorage.getItem('tts_sentences') || '[]') } catch { return [] }
  })
  const [selectedSentences, setSelectedSentences] = useState(new Set())
  const [audioItems, setAudioItems] = useState([])
  const [playingId, setPlayingId] = useState(null)
  const [stats, setStats] = useState({})
  const [systemPrompt, setSystemPrompt] = useState(() => localStorage.getItem('tts_systemPrompt') || DEFAULT_SYSTEM_PROMPT)

  // Folder management
  const [folders, setFolders] = useState([])
  const [selectedFolders, setSelectedFolders] = useState(new Set())
  const [isDownloadingFolders, setIsDownloadingFolders] = useState(false)
  const [folderSearch, setFolderSearch] = useState('')

  // Batch panel toggle (null | 'batch' | 'fileBatch')
  const [activeBatchPanel, setActiveBatchPanel] = useState(null)

  // Settings toggles
  const [showTTSSettings, setShowTTSSettings] = useState(false)
  const [showSentenceSettings, setShowSentenceSettings] = useState(false)

  // Expandable folders
  const [expandedFolder, setExpandedFolder] = useState(null)
  const [folderItems, setFolderItems] = useState({})

  // Error reports
  const [errorReports, setErrorReports] = useState([])

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

  // TTS Model settings
  const [ttsModel, setTtsModel] = useState('gemini_pro')
  const [ttsModels, setTtsModels] = useState({})
  const [ttsPrompt, setTtsPrompt] = useState('Doğal ve rahat bir tonla konuş. Günlük sohbet ediyormuş gibi, ne çok hızlı ne çok yavaş, sakin ve akıcı bir şekilde oku. Monoton olma, doğal vurgulama yap.')

  // Colloquial normalization
  const [colloquialEnabled, setColloquialEnabled] = useState(false)

  // Loading states
  const [isGeneratingSentences, setIsGeneratingSentences] = useState(false)
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [isBatchProcessing, setIsBatchProcessing] = useState(false)
  const [batchProgress, setBatchProgress] = useState('')
  const [batchWords, setBatchWords] = useState('gideceğim, yapacağım, alacağım, gelecek, bakacağız, olacak, vereceğim, söyleyeceğim, göreceğim, bileceğim, geliyorum, yapıyorsun, bakıyor, gidiyoruz, biliyorsunuz, anlıyorum, söylüyorsun, görüyor, istiyorum, düşünüyoruz, yani, işte, hani, falan, mesela, aslında, zaten, bence, galiba, acaba, abi, abla, hocam, kardeşim, ya, tamam, peki, haydi, tabii, olur, dün, yarın, şimdi, sonra, biraz, bugün, buraya, oraya, şuraya, nereden, gökyüzü, öğretmen, üzüm, çiçek, şişe, müzik, güneş, düşünce, öğrenci, küçük')
  const [batchSentencesPerWord, setBatchSentencesPerWord] = useState(15)

  // File import batch TTS
  const [isFileBatchProcessing, setIsFileBatchProcessing] = useState(false)
  const [fileBatchFile, setFileBatchFile] = useState(null)
  const [fileBatchFolder, setFileBatchFolder] = useState('file_import')
  const [fileBatchProgress, setFileBatchProgress] = useState(null) // {current, total, success, failed, skipped}
  const fileBatchAbortRef = useRef(null)

  // Messages
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  // Confirmation dialog
  const [confirmDialog, setConfirmDialog] = useState(null) // { message, onConfirm }

  // Custom sentence input
  const [customSentence, setCustomSentence] = useState('')

  // Editing state for sentences
  const [editingSentenceId, setEditingSentenceId] = useState(null)

  // How It Works panel
  const [showHowItWorks, setShowHowItWorks] = useState(() => {
    const saved = localStorage.getItem('hideHowItWorks')
    return saved !== 'true'
  })

  const toggleHowItWorks = () => {
    setShowHowItWorks(prev => {
      const next = !prev
      localStorage.setItem('hideHowItWorks', next ? 'false' : 'true')
      return next
    })
  }

  // Audio ref
  const audioRef = useRef(null)
  const location = useLocation()

  // Load stats on mount
  // Persist to localStorage
  useEffect(() => { localStorage.setItem('tts_word', word) }, [word])
  useEffect(() => { localStorage.setItem('tts_sentenceCount', sentenceCount) }, [sentenceCount])
  useEffect(() => { localStorage.setItem('tts_sentences', JSON.stringify(sentences)) }, [sentences])
  useEffect(() => { localStorage.setItem('tts_systemPrompt', systemPrompt) }, [systemPrompt])

  useEffect(() => {
    loadStats()
    loadFolders()
    // Load audio items for persisted word
    if (word.trim()) loadItems(word.trim())
    loadVoices()
    loadLLMConfig()
    loadTTSConfig()
    loadErrorReports()
    loadColloquialSettings()

    // Poll for errors every 10 seconds
    const interval = setInterval(loadErrorReports, 10000)
    return () => clearInterval(interval)
  }, [])

  // Reload TTS config when navigating back (e.g. from Settings page after saving API key)
  useEffect(() => {
    loadTTSConfig()
  }, [location.pathname])

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

  const loadItems = async (filterWord = null) => {
    try {
      const w = filterWord || word
      if (!w || !w.trim()) {
        setAudioItems([])
        return
      }
      const response = await fetch(`${API_BASE_URL}/api/items?status=generated&word=${encodeURIComponent(w.trim())}`)
      const data = await response.json()
      if (data.success) {
        setAudioItems(data.items)
      }
    } catch (err) {
      console.error('Items yüklenemedi:', err)
    }
  }

  const loadVoices = async (modelKey = null) => {
    try {
      const m = modelKey || ttsModel
      const response = await fetch(`${API_BASE_URL}/api/voices?model=${m}`)
      const data = await response.json()
      if (data.success) {
        setVoices(data.voices)
        const voiceKeys = Object.keys(data.voices)
        // Reset voice selection when the voice list changes
        setVoice(prev => {
          if (voiceKeys.length > 0 && !voiceKeys.includes(prev)) {
            return voiceKeys[0]
          }
          return prev
        })
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

  const loadErrorReports = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/errors?status=pending`)
      const data = await response.json()
      if (data.success) {
        setErrorReports(data.reports)
      }
    } catch (err) {
      console.error('Hata raporları yüklenemedi:', err)
    }
  }

  const deleteErrorReport = async (id) => {
    try {
      await fetch(`${API_BASE_URL}/api/errors/${id}`, { method: 'DELETE' })
      loadErrorReports()
    } catch (err) {
      console.error('Rapor silinemedi:', err)
    }
  }

  const resolveErrorReport = async (id) => {
    try {
      await fetch(`${API_BASE_URL}/api/errors/${id}/status`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'resolved' })
      })
      loadErrorReports()
    } catch (err) {
      console.error('Rapor güncellenemedi:', err)
    }
  }

  const handleErrorGenerate = (report) => {
    setWord(report.word)
    // Optional: Auto generate?
    // generateSentences() 
    // Let's just set the word and let user click generate to be safe/controlled

    // Mark as resolved? Or wait until generation?
    // Maybe just delete it from the list or mark it?
    // Let's keep it until they manually remove or we can auto-resolve if needed.
    // For now, just set the word.
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
    // Optimistic UI update — change immediately
    setLlmProvider(provider)
    if (model) setSelectedOllamaModel(model)

    try {
      const response = await fetch(`${API_BASE_URL}/api/llm/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, model })
      })
      const data = await response.json()
      if (!data.success) {
        // Revert on failure
        loadLLMConfig()
      }
    } catch (err) {
      setError('LLM provider değiştirilemedi: ' + err.message)
      loadLLMConfig() // Revert
    }
  }

  const loadTTSConfig = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/tts/config`)
      const data = await response.json()
      if (data.success) {
        setTtsModel(data.model)
        setTtsModels(data.models || {})
      }
    } catch (err) {
      console.error('TTS config yüklenemedi:', err)
    }
  }

  const switchTTSModel = async (modelKey) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/tts/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelKey })
      })
      const data = await response.json()
      if (data.success) {
        setTtsModel(modelKey)
        // Reload voices for the new model
        loadVoices(modelKey)
      }
    } catch (err) {
      setError('TTS model değiştirilemedi: ' + err.message)
    }
  }

  const loadColloquialSettings = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/colloquial/settings`)
      const data = await response.json()
      if (data.success) {
        setColloquialEnabled(data.settings.enabled)
      }
    } catch (err) {
      console.error('Colloquial settings yüklenemedi:', err)
    }
  }

  const toggleColloquial = async (enabled) => {
    setColloquialEnabled(enabled)
    try {
      const response = await fetch(`${API_BASE_URL}/api/colloquial/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      })
      const data = await response.json()
      if (!data.success) {
        setColloquialEnabled(!enabled) // Revert
      }
    } catch (err) {
      setColloquialEnabled(!enabled) // Revert
      setError('Konuşma dili ayarı değiştirilemedi: ' + err.message)
    }
  }

  const startBatchProcess = async () => {
    const words = batchWords.split(',').map(w => w.trim()).filter(w => w)
    if (words.length === 0) {
      setError('En az bir kelime girin')
      return
    }

    setIsBatchProcessing(true)
    setBatchProgress(`0/${words.length} kelime işleniyor...`)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/batch-process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          words,
          sentencesPerWord: batchSentencesPerWord,
          voice,
          speakingRate,
          pitch,
          volumeGainDb,
          ttsPrompt: ttsPrompt || undefined
        })
      })
      const data = await response.json()
      if (data.success) {
        setSuccess(`✅ Toplu işlem tamamlandı: ${data.total_generated} ses dosyası üretildi (${data.total_words} kelime)`)
        loadFolders()
        loadStats()
      } else {
        setError(data.error || 'Toplu işlem başarısız')
      }
    } catch (err) {
      setError('Toplu işlem hatası: ' + err.message)
    } finally {
      setIsBatchProcessing(false)
      setBatchProgress('')
    }
  }

  // ── File Import Batch TTS ────────────────────────────────────────────
  const startFileBatchTTS = async () => {
    if (!fileBatchFile) return

    setIsFileBatchProcessing(true)
    setFileBatchProgress({ current: 0, total: 0, success: 0, failed: 0, skipped: 0 })
    setError(null)
    setSuccess(null)

    const abortController = new AbortController()
    fileBatchAbortRef.current = abortController

    try {
      const formData = new FormData()
      formData.append('file', fileBatchFile)
      formData.append('word', fileBatchFolder.trim() || 'file_import')
      formData.append('voice', voice)
      formData.append('speakingRate', speakingRate)
      formData.append('pitch', pitch)
      formData.append('volumeGainDb', volumeGainDb)
      if (ttsPrompt) formData.append('ttsPrompt', ttsPrompt)

      const response = await fetch(`${API_BASE_URL}/api/batch-tts-from-file`, {
        method: 'POST',
        body: formData,
        signal: abortController.signal
      })

      if (!response.ok) {
        const err = await response.json()
        throw new Error(err.error || 'İşlem başarısız')
      }

      // Parse SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              if (data.type === 'progress') {
                setFileBatchProgress({
                  current: data.current,
                  total: data.total,
                  success: data.success,
                  failed: data.failed,
                  skipped: data.skipped
                })
              } else if (data.type === 'complete') {
                setSuccess(`✅ Dosyadan toplu TTS tamamlandı: ${data.success} başarılı, ${data.failed} başarısız, ${data.skipped} atlanmış (toplam ${data.total})`)
                loadStats()
                loadFolders()
                loadItems()
              }
            } catch (e) { /* ignore parse errors */ }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError('Dosya TTS hatası: ' + err.message)
      }
    } finally {
      setIsFileBatchProcessing(false)
      fileBatchAbortRef.current = null
    }
  }

  const cancelFileBatchTTS = () => {
    if (fileBatchAbortRef.current) {
      fileBatchAbortRef.current.abort()
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

  const toggleFolderExpand = async (folderName) => {
    if (expandedFolder === folderName) {
      setExpandedFolder(null)
      return
    }
    setExpandedFolder(folderName)
    // Fetch items if not cached
    if (!folderItems[folderName]) {
      await loadFolderItems(folderName)
    }
  }

  const loadFolderItems = async (folderName) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/items?word=${encodeURIComponent(folderName)}&status=generated&limit=500`)
      const data = await response.json()
      if (data.success) {
        setFolderItems(prev => ({ ...prev, [folderName]: data.items }))
      }
    } catch (err) {
      console.error('Klasör dosyaları yüklenemedi:', err)
    }
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
    // Clear old sentences for a fresh start
    setSentences([])
    setSelectedSentences(new Set())

    try {
      const response = await fetch(`${API_BASE_URL}/api/generate-sentences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          word: word.trim(),
          count: sentenceCount,
          provider: llmProvider,
          model: llmProvider === 'ollama' ? selectedOllamaModel : undefined,
          system_prompt: systemPrompt.trim() || undefined,
          full_prompt: systemPrompt.trim() || undefined
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
    setConfirmDialog({
      message: 'Tüm cümleleri silmek istediğinize emin misiniz?',
      onConfirm: () => {
        setSentences([])
        setSelectedSentences(new Set())
        setConfirmDialog(null)
      }
    })
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
          volumeGainDb,
          ttsPrompt: ttsPrompt || undefined
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
      loadItems(word.trim()) // Pass the current word to filter items
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
      // Also remove from folder items cache
      setFolderItems(prev => {
        const updated = { ...prev }
        for (const key in updated) {
          updated[key] = updated[key].filter(item => item.id !== itemId)
        }
        return updated
      })
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
    setConfirmDialog({
      message: `${selectedFolders.size} adet klasörü ve içindeki tüm dosyaları silmek istediğinize emin misiniz?`,
      onConfirm: () => {
        setConfirmDialog(null)
        doDeleteSelectedFolders()
      }
    })
  }

  const doDeleteSelectedFolders = async () => {

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
      {/* Toast Notifications */}
      <div className="toast-container">
        {error && (
          <div className="toast toast-error">
            <AlertCircle size={16} />
            <span>{error}</span>
            <button className="toast-close" onClick={() => setError(null)}>×</button>
          </div>
        )}
        {success && (
          <div className="toast toast-success">
            <Check size={16} />
            <span>{success}</span>
            <button className="toast-close" onClick={() => setSuccess(null)}>×</button>
          </div>
        )}
      </div>

      {/* Confirmation Dialog */}
      {confirmDialog && (
        <div className="batch-slideout-overlay" onClick={() => setConfirmDialog(null)}>
          <div className="batch-slideout-panel" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '400px' }}>
            <div className="batch-slideout-header">
              <span>⚠️ Onay</span>
            </div>
            <div className="batch-slideout-content">
              <p style={{ fontSize: '0.9rem', margin: 0 }}>{confirmDialog.message}</p>
              <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                <button className="btn btn-small btn-ghost" onClick={() => setConfirmDialog(null)}>Vazgeç</button>
                <button className="btn btn-small" style={{ background: '#ef4444', borderColor: '#ef4444' }} onClick={confirmDialog.onConfirm}>Evet, Sil</button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="container">
        {/* Navigation Header */}
        <header className="header">
          <div className="header-content">
            <Wand2 className="header-icon" />
            <div>
              <h1>Training Data Generator</h1>
              <p>TTS Telaffuz Eğitimi için Sentetik Veri Oluşturucu</p>
            </div>
          </div>
          <nav className="nav-tabs">
            <NavLink to="/" end className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              <Wand2 size={16} /> Veri Üretimi
            </NavLink>
            <NavLink to="/models" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              <Brain size={16} /> Modeller
            </NavLink>
            <NavLink to="/training" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              <GraduationCap size={16} /> Eğitim
            </NavLink>
            <NavLink to="/settings" className={({ isActive }) => `nav-tab ${isActive ? 'active' : ''}`}>
              <Settings size={16} /> Ayarlar
            </NavLink>
          </nav>
        </header>

        <Routes>
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/" element={<>

            {/* How It Works Panel */}
            {showHowItWorks ? (
              <div className="how-it-works-panel">
                <div className="how-it-works-header">
                  <div className="how-it-works-title">
                    <Info size={18} />
                    <span>Nasıl Çalışır?</span>
                  </div>
                  <button className="how-it-works-toggle" onClick={toggleHowItWorks} title="Gizle">
                    <X size={16} />
                  </button>
                </div>
                <div className="how-it-works-flow">
                  <div className="flow-step">
                    <div className="flow-step-icon" style={{ background: 'rgba(139, 92, 246, 0.2)', color: '#a78bfa' }}>
                      <Wand2 size={20} />
                    </div>
                    <div className="flow-step-content">
                      <strong>Kelime Girin</strong>
                      <span>Yanlış telaffuz edilen kelimeyi yazın</span>
                    </div>
                  </div>
                  <ArrowRight size={16} className="flow-arrow" />
                  <div className="flow-step">
                    <div className="flow-step-icon" style={{ background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa' }}>
                      <Brain size={20} />
                    </div>
                    <div className="flow-step-content">
                      <strong>AI Cümle Üretimi</strong>
                      <span>LLM otomatik cümleler oluşturur</span>
                    </div>
                  </div>
                  <ArrowRight size={16} className="flow-arrow" />
                  <div className="flow-step">
                    <div className="flow-step-icon" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#6ee7b7' }}>
                      <Volume2 size={20} />
                    </div>
                    <div className="flow-step-content">
                      <strong>TTS Ses Üretimi</strong>
                      <span>Gemini TTS ile .wav dosyaları oluşturulur</span>
                    </div>
                  </div>
                  <ArrowRight size={16} className="flow-arrow" />
                  <div className="flow-step">
                    <div className="flow-step-icon" style={{ background: 'rgba(236, 72, 153, 0.2)', color: '#f472b6' }}>
                      <GraduationCap size={20} />
                    </div>
                    <div className="flow-step-content">
                      <strong>XTTS Eğitimi</strong>
                      <span>Verileri indirip model eğitimi yapın</span>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <button className="how-it-works-show-btn" onClick={toggleHowItWorks} title="Nasıl Çalışır?">
                <Info size={14} />
                <span>Nasıl Çalışır?</span>
              </button>
            )}

            {/* Error Reports Section */
              errorReports.length > 0 && (
                <div className="card" style={{ marginBottom: '2rem', borderColor: 'rgba(239, 68, 68, 0.3)', background: 'rgba(239, 68, 68, 0.05)' }}>
                  <div className="step-header">
                    <div style={{ background: 'rgba(239, 68, 68, 0.2)', padding: '8px', borderRadius: '50%' }}>
                      <AlertTriangle size={20} color="#ef4444" />
                    </div>
                    <h2 style={{ color: '#ef4444' }}>Hatalı Telaffuz Raporları ({errorReports.length})</h2>
                  </div>

                  <div className="sentence-list">
                    {errorReports.map(report => (
                      <div key={report.id} className="sentence-item" style={{ background: 'rgba(0,0,0,0.2)' }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <strong style={{ fontSize: '1.1rem', color: 'white' }}>{report.word}</strong>
                            <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>{new Date(report.created_at + (report.created_at.endsWith('Z') ? '' : 'Z')).toLocaleString('tr-TR')}</span>
                          </div>
                          {report.explanation && (
                            <div style={{ marginTop: '4px', fontSize: '0.9rem', color: 'rgba(255,255,255,0.7)' }}>
                              "{report.explanation}"
                            </div>
                          )}
                        </div>

                        <button
                          onClick={() => handleErrorGenerate(report)}
                          className="btn btn-primary btn-small"
                          title="Bu kelime için veri üret"
                        >
                          <Wand2 size={16} />
                          <span>Üret</span>
                        </button>

                        <button
                          onClick={() => resolveErrorReport(report.id)}
                          className="btn btn-success-small"
                          title="Çözüldü olarak işaretle"
                          style={{ padding: '6px' }}
                        >
                          <Check size={16} />
                        </button>

                        <button
                          onClick={() => deleteErrorReport(report.id)}
                          className="btn btn-danger-small"
                          title="Raporu sil"
                          style={{ padding: '6px' }}
                        >
                          <X size={16} />
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* Batch Slide-out Panel */}
            {activeBatchPanel && (
              <div className="batch-slideout-overlay" onClick={() => setActiveBatchPanel(null)}>
                <div className="batch-slideout-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="batch-slideout-header">
                    <span>{activeBatchPanel === 'batch' ? '🚀 Toplu İşlem' : '📄 Dosyadan Toplu TTS'}</span>
                    <button className="batch-slideout-close" onClick={() => setActiveBatchPanel(null)}>
                      <span>✖️</span>
                    </button>
                  </div>
                  <div className="batch-slideout-content">
                    {activeBatchPanel === 'batch' && (
                      <>
                        <textarea
                          value={batchWords}
                          onChange={(e) => setBatchWords(e.target.value)}
                          rows={4}
                          className="input-text"
                          style={{ resize: 'vertical', fontSize: '0.8rem', padding: '0.6rem', lineHeight: 1.5 }}
                          placeholder="Kelimeleri virgülle ayırarak girin..."
                          disabled={isBatchProcessing}
                        />
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <select
                            value={batchSentencesPerWord}
                            onChange={(e) => setBatchSentencesPerWord(parseInt(e.target.value))}
                            className="input-select"
                            style={{ fontSize: '0.85rem', padding: '0.5rem 0.75rem' }}
                            disabled={isBatchProcessing}
                          >
                            {[5, 10, 15, 20, 25].map(n => (
                              <option key={n} value={n}>{n} cümle/kelime</option>
                            ))}
                          </select>
                          <button
                            onClick={startBatchProcess}
                            disabled={isBatchProcessing || !batchWords.trim()}
                            className="btn btn-primary"
                            style={{ flex: 1, justifyContent: 'center' }}
                          >
                            {isBatchProcessing ? (batchProgress || '⏳ İşleniyor...') : '🚀 Başlat'}
                          </button>
                        </div>
                        {isBatchProcessing && (
                          <p style={{ fontSize: '0.75rem', color: '#f59e0b', textAlign: 'center', margin: 0 }}>
                            ⚠️ Bu işlem uzun sürebilir. Sayfayı kapatmayın.
                          </p>
                        )}
                      </>
                    )}
                    {activeBatchPanel === 'fileBatch' && (
                      <>
                        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
                          Her satırda bir cümle olan .txt dosyası yükleyin.
                        </p>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                          <label className="btn btn-small" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(139, 92, 246, 0.15)', border: '1px solid rgba(139, 92, 246, 0.3)' }}>
                            <Upload size={14} />
                            {fileBatchFile ? fileBatchFile.name : 'Dosya Seç'}
                            <input
                              type="file"
                              accept=".txt"
                              style={{ display: 'none' }}
                              onChange={(e) => setFileBatchFile(e.target.files[0] || null)}
                              disabled={isFileBatchProcessing}
                            />
                          </label>
                          <input
                            type="text"
                            value={fileBatchFolder}
                            onChange={(e) => setFileBatchFolder(e.target.value)}
                            placeholder="Klasör adı"
                            className="input-text"
                            style={{ flex: 1, minWidth: '100px', maxWidth: '160px', fontSize: '0.85rem', padding: '0.45rem 0.6rem' }}
                            disabled={isFileBatchProcessing}
                          />
                        </div>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                          <button
                            onClick={startFileBatchTTS}
                            disabled={isFileBatchProcessing || !fileBatchFile}
                            className="btn btn-primary"
                            style={{ flex: 1, justifyContent: 'center' }}
                          >
                            {isFileBatchProcessing ? '⏳ İşleniyor...' : '📄 Başlat'}
                          </button>
                          {isFileBatchProcessing && (
                            <button onClick={cancelFileBatchTTS} className="btn btn-small btn-danger-small" style={{ padding: '0.45rem 0.75rem' }}>
                              <X size={14} /> İptal
                            </button>
                          )}
                        </div>
                        {fileBatchProgress && fileBatchProgress.total > 0 && (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                            <div style={{ height: '8px', borderRadius: '4px', background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                              <div style={{
                                height: '100%',
                                width: `${(fileBatchProgress.current / fileBatchProgress.total * 100).toFixed(1)}%`,
                                background: 'linear-gradient(90deg, #8b5cf6, #6d28d9)',
                                borderRadius: '4px', transition: 'width 0.3s ease'
                              }} />
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                              <span>{fileBatchProgress.current}/{fileBatchProgress.total} ({(fileBatchProgress.current / fileBatchProgress.total * 100).toFixed(0)}%)</span>
                              <span>
                                ✅ {fileBatchProgress.success}
                                {fileBatchProgress.failed > 0 && <span style={{ color: '#ef4444' }}> ❌ {fileBatchProgress.failed}</span>}
                                {fileBatchProgress.skipped > 0 && <span> ⏭️ {fileBatchProgress.skipped}</span>}
                              </span>
                            </div>
                          </div>
                        )}
                        {isFileBatchProcessing && (
                          <p style={{ fontSize: '0.75rem', color: '#8b5cf6', textAlign: 'center', margin: 0 }}>⚠️ Sayfayı kapatmayın.</p>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Sentence Settings Panel */}
            {showSentenceSettings && (
              <div className="batch-slideout-overlay" onClick={() => setShowSentenceSettings(false)}>
                <div className="batch-slideout-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="batch-slideout-header">
                    <span>⚙️ Cümle Üretim Ayarları</span>
                    <button className="batch-slideout-close" onClick={() => setShowSentenceSettings(false)}>
                      <span>✖️</span>
                    </button>
                  </div>
                  <div className="batch-slideout-content">
                    <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', margin: 0 }}>
                      LLM'e gönderilen cümle üretim promptunu buradan özelleştirebilirsiniz. Prompt'ta {'{word}'} ve {'{count}'} yer tutucuları otomatik doldurulur.
                    </p>

                    {/* LLM Provider Toggle */}
                    <div className="llm-provider-toggle" style={{ marginBottom: 0 }}>
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

                    <div>
                      <label style={{ fontSize: '0.8rem', fontWeight: 500, display: 'block', marginBottom: '6px' }}>Cümle Sayısı (varsayılan)</label>
                      <input
                        type="number"
                        value={sentenceCount}
                        onChange={(e) => setSentenceCount(Math.max(1, parseInt(e.target.value) || 1))}
                        min="1"
                        className="input-count"
                        style={{ width: '100%', textAlign: 'left', padding: '8px 12px' }}
                      />
                    </div>

                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
                        <label style={{ fontSize: '0.8rem', fontWeight: 500 }}>System Prompt</label>
                        <button
                          className="btn btn-small btn-ghost"
                          style={{ fontSize: '0.7rem', padding: '3px 8px' }}
                          onClick={() => setSystemPrompt(DEFAULT_SYSTEM_PROMPT)}
                          title="Varsayılan prompt'a sıfırla"
                        >
                          Varsayılana Sıfırla
                        </button>
                      </div>
                      <textarea
                        value={systemPrompt}
                        onChange={(e) => setSystemPrompt(e.target.value)}
                        rows={14}
                        style={{ width: '100%', resize: 'vertical', fontSize: '0.72rem', lineHeight: 1.6, padding: '10px 12px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text)', fontFamily: 'monospace' }}
                      />
                      <span style={{ fontSize: '0.65rem', color: 'var(--text-dim)', marginTop: '6px', display: 'block' }}>
                        LLM'e gönderilen tam prompt. <code>{'{word}'}</code> ve <code>{'{count}'}</code> yer tutucuları otomatik doldurulur.
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* TTS Settings Popup */}
            {showTTSSettings && (
              <div className="batch-slideout-overlay" onClick={() => setShowTTSSettings(false)}>
                <div className="batch-slideout-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="batch-slideout-header">
                    <span>⚙️ TTS Ayarları</span>
                    <button className="batch-slideout-close" onClick={() => setShowTTSSettings(false)}>
                      <span>✖️</span>
                    </button>
                  </div>
                  <div className="batch-slideout-content">
                    {/* TTS Model Selection */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' }}>
                      {Object.entries(ttsModels).map(([key, model]) => (
                        <button
                          key={key}
                          onClick={() => switchTTSModel(key)}
                          className={`btn btn-small ${ttsModel === key ? 'btn-primary' : 'btn-ghost'}`}
                          style={{ fontSize: '0.7rem', padding: '8px 6px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '4px' }}
                          title={model.description}
                        >
                          {model.label}
                        </button>
                      ))}
                    </div>

                    {/* Colloquial Toggle */}
                    <div style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      padding: '8px 12px',
                      background: colloquialEnabled ? 'rgba(81, 207, 102, 0.1)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${colloquialEnabled ? 'rgba(81, 207, 102, 0.3)' : 'var(--border)'}`,
                      borderRadius: 'var(--radius-sm)', transition: 'all 0.2s'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '1rem' }}>🗣️</span>
                        <div>
                          <span style={{ fontSize: '0.8rem', fontWeight: 500 }}>Konuşma Dili</span>
                          <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', display: 'block' }}>
                            {colloquialEnabled ? 'Metin konuşma diline çevrilecek' : 'Yazı dili olarak kalacak'}
                          </span>
                        </div>
                      </div>
                      <label style={{ position: 'relative', display: 'inline-block', width: '40px', height: '22px', cursor: 'pointer' }}>
                        <input type="checkbox" checked={colloquialEnabled} onChange={(e) => toggleColloquial(e.target.checked)} style={{ opacity: 0, width: 0, height: 0 }} />
                        <span style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, background: colloquialEnabled ? '#51cf66' : 'rgba(255,255,255,0.15)', borderRadius: '22px', transition: 'all 0.3s' }}>
                          <span style={{ position: 'absolute', height: '16px', width: '16px', left: colloquialEnabled ? '21px' : '3px', bottom: '3px', background: 'white', borderRadius: '50%', transition: 'all 0.3s', boxShadow: '0 1px 3px rgba(0,0,0,0.3)' }} />
                        </span>
                      </label>
                    </div>

                    {/* Voice Selection */}
                    <div>
                      <label style={{ fontSize: '0.8rem', fontWeight: 500, display: 'block', marginBottom: '6px' }}>Ses</label>
                      <select value={voice} onChange={(e) => setVoice(e.target.value)} className="select-input" style={{ width: '100%' }}>
                        {Object.entries(voices).map(([key, name]) => (
                          <option key={key} value={key}>{name}</option>
                        ))}
                      </select>
                    </div>

                    {/* Gemini Prompt */}
                    {ttsModels[ttsModel] && !ttsModels[ttsModel].supports_ssml_params && (
                      <div>
                        <label style={{ fontSize: '0.8rem', fontWeight: 500, display: 'block', marginBottom: '6px' }}>Prompt (stil yönlendirmesi)</label>
                        <textarea
                          value={ttsPrompt}
                          onChange={(e) => setTtsPrompt(e.target.value)}
                          placeholder="Örn: Speak slowly and clearly with a warm tone"
                          rows={2}
                          style={{ width: '100%', resize: 'vertical', fontSize: '0.8rem', padding: '8px 12px', background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)', color: 'var(--text)', fontFamily: 'inherit' }}
                        />
                      </div>
                    )}

                    {/* SSML Params */}
                    {ttsModels[ttsModel]?.supports_ssml_params && (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px' }}>
                        <div>
                          <label style={{ fontSize: '0.75rem', display: 'block', marginBottom: '4px' }}>Hız: {speakingRate}x</label>
                          <input type="range" min="0.25" max="4.0" step="0.25" value={speakingRate} onChange={(e) => setSpeakingRate(parseFloat(e.target.value))} className="range-input" />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', display: 'block', marginBottom: '4px' }}>Pitch: {pitch}</label>
                          <input type="range" min="-20" max="20" step="1" value={pitch} onChange={(e) => setPitch(parseFloat(e.target.value))} className="range-input" />
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', display: 'block', marginBottom: '4px' }}>Volume: {volumeGainDb}dB</label>
                          <input type="range" min="-10" max="10.0" step="1" value={volumeGainDb} onChange={(e) => setVolumeGainDb(parseFloat(e.target.value))} className="range-input" />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Phases Grid - All 3 phases side by side */}
            <div className="phases-grid">
              {/* Step 1: Word Input */}
              <section className="card phase-card">
                <div className="step-header">
                  <span className="step-number">1</span>
                  <h2>Kelime Girin</h2>
                  <div style={{ marginLeft: 'auto', display: 'flex', gap: '6px' }}>
                    <button
                      className={`batch-icon-btn ${showSentenceSettings ? 'active' : ''}`}
                      onClick={() => setShowSentenceSettings(prev => !prev)}
                      title="Cümle Üretim Ayarları"
                    >
                      <span style={{ fontSize: '14px' }}>⚙️</span>
                    </button>
                    <button
                      className={`batch-icon-btn ${activeBatchPanel === 'batch' ? 'active' : ''}`}
                      onClick={() => setActiveBatchPanel(prev => prev === 'batch' ? null : 'batch')}
                      title="Toplu İşlem"
                    >
                      <span style={{ fontSize: '14px' }}>🚀</span>
                    </button>
                    <button
                      className={`batch-icon-btn ${activeBatchPanel === 'fileBatch' ? 'active' : ''}`}
                      onClick={() => setActiveBatchPanel(prev => prev === 'fileBatch' ? null : 'fileBatch')}
                      title="Dosyadan Toplu TTS"
                    >
                      <span style={{ fontSize: '14px' }}>📄</span>
                    </button>
                  </div>
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
                      <button
                        className={`batch-icon-btn ${showTTSSettings ? 'active' : ''}`}
                        onClick={() => setShowTTSSettings(prev => !prev)}
                        title="TTS Ayarları"
                      >
                        <span style={{ fontSize: '14px' }}>⚙️</span>
                      </button>
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
                      <span>İndir ({selectedFolders.size})</span>
                    </button>
                    <button
                      onClick={deleteSelectedFolders}
                      disabled={selectedFolders.size === 0}
                      className="btn btn-small btn-danger-small"
                    >
                      <Trash2 size={14} />
                      <span>Sil ({selectedFolders.size})</span>
                    </button>
                  </div>
                </div>
                <div style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(0,0,0,0.15)', borderTop: '1px solid rgba(255,255,255,0.05)', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <Search size={14} style={{ opacity: 0.4, flexShrink: 0 }} />
                  <input
                    type="text"
                    placeholder="Klasör ara..."
                    value={folderSearch}
                    onChange={(e) => setFolderSearch(e.target.value)}
                    style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: 'inherit', fontSize: '0.85rem', padding: '2px 0' }}
                  />
                  {folderSearch && (
                    <button onClick={() => setFolderSearch('')} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'pointer', padding: '2px', display: 'flex' }}>
                      <X size={14} />
                    </button>
                  )}
                </div>
                <div className="folder-list-container">
                  {folders.filter(f => f.name.toLowerCase().includes(folderSearch.toLowerCase())).map((folder) => (
                    <div key={folder.name} className={`folder-list-item ${expandedFolder === folder.name ? 'expanded' : ''}`}>
                      <div className="folder-list-row">
                        <input
                          type="checkbox"
                          checked={selectedFolders.has(folder.name)}
                          onChange={() => toggleFolder(folder.name)}
                          className="checkbox"
                          onClick={(e) => e.stopPropagation()}
                        />
                        <div className="folder-list-info" onClick={() => toggleFolderExpand(folder.name)}>
                          <ChevronRight size={16} className={`folder-expand-icon ${expandedFolder === folder.name ? 'rotated' : ''}`} />
                          <Folder size={18} className="folder-icon" />
                          <span className="folder-name">{folder.name}</span>
                          <span className="folder-file-count">{folder.file_count} dosya</span>
                        </div>
                        <button
                          onClick={(e) => { e.stopPropagation(); downloadFolder(folder.name) }}
                          className="btn-download"
                          title="Klasörü İndir"
                        >
                          <span className="icon">↓</span>
                        </button>
                        <button
                          onClick={(e) => { e.stopPropagation(); deleteFolder(folder.name) }}
                          className="btn-folder-delete"
                          title="Klasörü Sil"
                        >
                          <span className="icon">×</span>
                        </button>
                      </div>
                      {expandedFolder === folder.name && (
                        <div className="folder-expanded-content">
                          {!folderItems[folder.name] ? (
                            <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                              <RefreshCw className="spin" size={16} /> Yükleniyor...
                            </div>
                          ) : folderItems[folder.name].length === 0 ? (
                            <div style={{ padding: '1rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                              Bu klasörde ses dosyası bulunamadı
                            </div>
                          ) : (
                            <div className="audio-list">
                              {folderItems[folder.name].map((item) => (
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
                                    <span className="audio-meta">{item.duration_seconds?.toFixed(1)}s • {item.voice}</span>
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
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}


          </>} />
        </Routes>

      </div>
    </div>
  )
}

export default App
