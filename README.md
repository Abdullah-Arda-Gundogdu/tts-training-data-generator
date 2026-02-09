# Training Data Generator

A powerful, standalone web application designed to generate high-quality text-to-speech (TTS) training data sets. It combines AI-powered sentence generation (using OpenAI GPT or local Ollama models) with Google's Text-to-Speech API to create perfectly aligned audio-text pairs.

## 🌟 Key Features

- **Flexible AI Integration**: Choose between **OpenAI GPT** (cloud) or **Ollama** (local, privacy-focused) for generating natural sentences.
- **Multilingual Support**: Specialized in generating Turkish sentences, but capable of other languages.
- **Context-Aware Generation**: Generate sentences containing specific target words or phrases in various contexts.
- **High-Quality Audio**: Integrates **Google Cloud TTS** for realistic speech synthesis (supports WaveNet and Neural2 voices).
- **Smart Workflow**:
  - **Review & Edit**: Manually review and edit generated sentences before audio synthesis.
  - **Folder Management**: Organize outputs into folders, delete unwanted sets, and download specific collections as ZIP files.
  - **Duplicate Prevention**: Automatically detects and prevents duplicate sentence generation.
- **Export Ready**: Exports metadata in XTTS format, ready for model training.

## 🛠️ Tech Stack

- **Frontend**: React (Vite), CSS, Lucide React icons.
- **Backend**: Python (Flask), SQLite, OpenAI SDK, Google Cloud TTS Client.
- **Tools**: Ollama (optional for local LLM).

## 📋 Prerequisites

- **Node.js** (v18+ recommended)
- **Python** (3.8+ recommended)
- **Google Cloud Account**: with Cloud Text-to-Speech API enabled.
- **Ollama** (Optional): If you want to run LLMs locally.

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/TrainingDataGenerator.git
cd TrainingDataGenerator
```

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

## ⚙️ Configuration

1. Create a `.env` file in the `backend/` directory (using `.env.example` as a template).

```ini
# backend/.env

# 1. Choose your LLM Provider: 'openai' or 'ollama'
LLM_PROVIDER=openai

# 2. If using OpenAI:
OPENAI_API_KEY=sk-your-openai-key

# 3. If using Ollama (Local):
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b  # or any other installed model

# 4. Google Cloud Credentials (REQUIRED for Audio)
GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json
```

2. Place your Google Cloud service account JSON file in `backend/google_credentials.json`.

## 🏃 Usage

### Start the Application

You can start the app using the provided batch files (Windows) or manually.

**Method 1: Batch Files**
- Double-click `start-backend.bat`
- Double-click `start-frontend.bat`

**Method 2: Manual Start**
```bash
# Terminal 1: Backend
cd backend
python app.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Workflow

1. **Configure LLM**: Use the settings icon in the UI to switch between OpenAI and Ollama.
2. **Generate Text**: Enter a target word (e.g., "demo") and the number of sentences. The AI will generate sentences containing that exact word.
3. **Review**: Edit or delete sentences if they aren't perfect.
4. **Generate Audio**: Click "Generate Audio" to synthesize speech using Google TTS.
5. **Manage & Export**:
   - Files are saved in `backend/training_output/`.
   - Use the "Folders" tab to browse, delete, or download folders as ZIP archives.
   - Output includes audio files and a `metadata.csv` (XTTS format).

## 📄 Output Format

The `metadata.csv` is formatted for XTTS training:
```
path/to/audio/file.wav|Spoken text content here.
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
