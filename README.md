<div align="center">

# 🎙️ TTS Training Data Generator

**An end-to-end web application for generating, managing, and training Text-to-Speech models.**

Generate natural sentences with AI · Synthesize audio with Google Cloud TTS (Chirp3 HD, Gemini Flash / Pro) · Fine-tune XTTS v2 models — all from a single UI.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-7-646CFF?logo=vite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## ✨ Features

### 📝 AI-Powered Sentence Generation
- **Dual LLM support** — use **OpenAI GPT** (cloud) or **Ollama** (local / privacy-focused).
- Generate natural sentences containing specific **target words** in varied contexts.
- Built-in **duplicate detection** prevents redundant data.

### 🔊 Audio Synthesis
- **Unified TTS engine** powered by **Google Cloud Text-to-Speech** — supports multiple model families from a single interface:
  - **Chirp3 HD** — high-definition neural voices
  - **Gemini 2.5 Flash** — fast, expressive, natural-sounding
  - **Gemini 2.5 Pro** — highest quality, most natural
  - **Gemini 2.5 Flash Lite** — lightweight, lower-latency
- **TTS Prompt control** — guide speech style, tone, and emotion with natural language prompts (Gemini models).
- Automatic alignment of audio files with transcript metadata.
- Outputs ready-to-use `metadata.csv` in **XTTS format**.

### 🧠 XTTS v2 Model Training
- Fine-tune Coqui **XTTS v2** directly from the browser.
- Configurable epochs, learning rate, batch size, and save intervals.
- Real-time **training console** with live log streaming.
- Automatic extraction of fine-tuned model weights.
- **Cancel training** anytime — cancelled jobs are tracked separately.

### 🗂️ Data & Model Management
- **Folder browser** — organize, preview, delete, and download output folders as ZIP.
- **Folder search** — quickly filter folders by name.
- **Model registry** — list trained models, set a default base model, run inference, and compare results.
- **Model status filters** — filter by Completed, Training, Failed, or Cancelled.
- **Settings page** — manage API keys, switch LLM / TTS providers, and adjust voice parameters.
- **Error reporting** — report mispronunciations with timezone-aware timestamps.

### 🌐 Network & UX
- Local-network accessible — serve to other devices on your LAN.
- Responsive, modern React UI with smooth animations and dark-mode aesthetics.
- **Instant UI updates** — optimistic state changes for snappy interactions.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   React Frontend                    │
│         (Vite · React Router · Lucide Icons)        │
│                                                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ TTS      │  │ Training     │  │ Models        │  │
│  │ Client   │  │ Page         │  │ Page          │  │
│  └────┬─────┘  └──────┬───────┘  └───────┬───────┘  │
└───────┼────────────────┼─────────────────┼──────────┘
        │     REST API   │                 │
┌───────▼────────────────▼─────────────────▼──────────┐
│                   Flask Backend                     │
│                                                     │
│  ┌─────────────┐ ┌────────────────┐ ┌────────────┐  │
│  │ llm_service │ │ training_      │ │ inference_ │  │
│  │ (OpenAI /   │ │ service /      │ │ service    │  │
│  │  Ollama)    │ │ xtts_trainer   │ │            │  │
│  └──────┬──────┘ └───────┬────────┘ └─────┬──────┘  │
│         │                │                │          │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐  │
│  │ Google TTS  │  │ Coqui TTS   │  │ Model       │  │
│  │ (unified)   │  │ (XTTS v2)   │  │ Registry    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│                                                     │
│              SQLite · training_data.db               │
└─────────────────────────────────────────────────────┘
```

---

## 📋 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.8+ | Backend runtime |
| **Node.js** | 18+ | Frontend tooling |
| **Google Cloud Account** | — | Cloud TTS API enabled + service-account JSON |
| **NVIDIA GPU** | CUDA capable | Required for XTTS training, optional for inference |
| **Ollama** *(optional)* | latest | Only if using local LLMs instead of OpenAI |

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Abdullah-Arda-Gundogdu/tts-training-data-generator.git
cd tts-training-data-generator
```

### 2. Backend Setup

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Configure Environment Variables

Copy the example `.env` and fill in your keys:

```bash
cp backend/.env.example backend/.env
```

```ini
# backend/.env

# OpenAI API Key (required if LLM_PROVIDER=openai)
OPENAI_API_KEY=sk-your-openai-api-key-here

# LLM Provider: "openai" or "ollama"
LLM_PROVIDER=openai

# Ollama settings (only used when LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Google Cloud credentials (required for all TTS models)
GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json

# TTS Model: "chirp3_hd", "gemini_flash", "gemini_pro", or "gemini_flash_lite"
TTS_MODEL=chirp3_hd
```

> **Note:** Place your Google Cloud service-account JSON file at `backend/google_credentials.json`. All TTS models (including Gemini variants) use Google Cloud credentials.

### 5. Download XTTS Base Files *(for training)*

If you plan to fine-tune XTTS v2, place the following base model files in `backend/xtts_base_files/`:

| File | Description |
|---|---|
| `model.pth` | Pre-trained XTTS v2 checkpoint (~1.8 GB) |
| `dvae.pth` | Discrete VAE weights (~200 MB) |
| `mel_stats.pth` | Mel-spectrogram statistics |
| `vocab.json` | Tokenizer vocabulary |

These can be downloaded from the [Coqui XTTS v2 model page](https://huggingface.co/coqui/XTTS-v2).

---

## 🏃 Usage

### Starting the Application

**Option A — Batch files (Windows):**
- Double-click `start-backend.bat`
- Double-click `start-frontend.bat`

**Option B — Manual:**
```bash
# Terminal 1: Backend
cd backend
python app.py          # Runs on http://localhost:5001

# Terminal 2: Frontend
cd frontend
npm run dev            # Runs on http://localhost:5173
```

### Workflow

```
 ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
 │  1. Generate   │────▶│  2. Review &   │────▶│  3. Synthesize │
 │   Sentences    │     │     Edit       │     │     Audio      │
 └────────────────┘     └────────────────┘     └───────┬────────┘
                                                       │
 ┌────────────────┐     ┌────────────────┐             │
 │  5. Run        │◀────│  4. Manage     │◀────────────┘
 │   Inference    │     │   & Export     │
 └────────────────┘     └────────────────┘
```

1. **Generate Sentences** — Enter a target word and count. The AI generates diverse sentences containing that word.
2. **Review & Edit** — Inspect, edit, or delete any sentence before proceeding.
3. **Synthesize Audio** — Click *Generate Audio* to produce WAV files. Optionally add a TTS prompt for style control (Gemini models).
4. **Manage & Export** — Browse output folders, search by name, download as ZIP, or delete unwanted sets.
5. **Train & Infer** — Launch XTTS fine-tuning from the Training page, then test your model on the Models page.

---

## 📁 Project Structure

```
tts-training-data-generator/
├── backend/
│   ├── app.py                  # Flask API (main entry point)
│   ├── llm_service.py          # OpenAI / Ollama sentence generation
│   ├── google_tts_service.py   # Unified TTS wrapper (Chirp3 HD, Gemini Flash/Pro/Lite)
│   ├── training_service.py     # Training job orchestration
│   ├── xtts_trainer.py         # Coqui XTTS v2 training logic
│   ├── inference_service.py    # Model inference (TTS synthesis)
│   ├── model_registry.py       # Trained model discovery & metadata
│   ├── training_database.py    # SQLite ORM for training data
│   ├── requirements.txt
│   ├── .env.example
│   └── xtts_base_files/        # Pre-trained XTTS v2 base weights
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main application shell & TTS client
│   │   ├── pages/
│   │   │   ├── TrainingPage.jsx
│   │   │   ├── ModelsPage.jsx
│   │   │   └── SettingsPage.jsx
│   │   └── App.css             # Global styles
│   ├── index.html
│   ├── vite.config.js
│   └── package.json
│
├── start-backend.bat           # Quick-start script (Windows)
├── start-frontend.bat
├── .gitignore
└── README.md
```

---

## 📄 Output Format

Each generation produces a folder inside `backend/training_output/` containing:

- **WAV audio files** — one per sentence
- **`metadata.csv`** — pipe-delimited, XTTS-compatible:

```
audio_filename|Normalized sentence text.|Normalized sentence text.
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
