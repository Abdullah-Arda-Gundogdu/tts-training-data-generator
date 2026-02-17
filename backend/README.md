# Backend — Flask API

Python/Flask backend for the TTS Training Data Generator. Handles sentence generation, audio synthesis, model training, and inference.

## Setup

```bash
# From the project root
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r backend/requirements.txt
```

### Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```ini
LLM_PROVIDER=openai                           # "openai" or "ollama"
OPENAI_API_KEY=sk-your-key-here               # Required if using OpenAI
OLLAMA_BASE_URL=http://localhost:11434         # Required if using Ollama
OLLAMA_MODEL=llama3.1:8b                       # Ollama model name
TTS_PROVIDER=google_cloud                      # "google_cloud" or "gemini"
GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json  # For Google Cloud TTS
GEMINI_API_KEY=your-gemini-api-key-here        # For Gemini 2.5 Flash TTS
```

- **Google Cloud TTS:** Place your service-account JSON as `google_credentials.json` in this directory.
- **Gemini TTS:** Get an API key from [Google AI Studio](https://aistudio.google.com).

### XTTS Base Files (for training)

Place these in `xtts_base_files/` (download from [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2)):

| File | Size |
|---|---|
| `model.pth` | ~1.8 GB |
| `dvae.pth` | ~200 MB |
| `mel_stats.pth` | ~1 KB |
| `vocab.json` | ~350 KB |

## Running

```bash
python app.py
# Server starts on http://localhost:5001
```

## Architecture

```
app.py                     ← Flask API (main entry point)
├── llm_service.py         ← Sentence generation (OpenAI / Ollama)
├── google_tts_service.py  ← Google Cloud TTS wrapper
├── gemini_tts_service.py  ← Gemini 2.5 Flash TTS wrapper
├── training_service.py    ← Training job orchestration
├── xtts_trainer.py        ← Coqui XTTS v2 training logic
├── inference_service.py   ← Model inference (TTS synthesis)
├── model_registry.py      ← Trained model DB (SQLite)
└── training_database.py   ← Training data DB (SQLite)
```

## API Endpoints

### Sentence Generation & Audio

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/generate-sentences` | Generate sentences with LLM |
| POST | `/api/generate-audio` | Synthesize WAV files (Google Cloud or Gemini TTS) |
| GET | `/api/audio/<id>/play` | Stream audio playback |
| GET | `/api/items` | List training items |
| PUT | `/api/items/<id>` | Update a training item |
| DELETE | `/api/items/<id>` | Delete a training item |
| POST | `/api/items/bulk-delete` | Bulk delete items |
| GET | `/api/stats` | Get generation statistics |
| POST | `/api/export` | Export metadata.csv |
| GET | `/api/export/download` | Download latest metadata.csv |
| GET | `/api/voices` | List available TTS voices (provider-aware) |

### Folder Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/folders` | List output folders |
| DELETE | `/api/folders/<name>` | Delete a folder and its files |
| POST | `/api/folders/bulk-delete` | Bulk delete folders |
| GET | `/api/folders/<name>/download` | Download folder as ZIP |
| POST | `/api/folders/download` | Download multiple folders as ZIP |

### LLM Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/llm/config` | Get current LLM provider config |
| POST | `/api/llm/config` | Set LLM provider and model |
| GET | `/api/llm/models` | List available Ollama models |

### TTS Configuration

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/tts/config` | Get current TTS provider config |
| POST | `/api/tts/config` | Set TTS provider (google_cloud / gemini) |

### Model Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all trained models |
| GET | `/api/models/<id>` | Get model details |
| POST | `/api/models` | Register a model manually |
| PUT | `/api/models/<id>` | Update model metadata |
| DELETE | `/api/models/<id>` | Delete model and its files |
| GET | `/api/models/<id>/test-audio` | Stream test audio |
| POST | `/api/models/<id>/synthesize` | Run inference (generate speech) |

### Training

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/training/start` | Start a training job |
| GET | `/api/training/status/<id>` | Get training status and logs |
| POST | `/api/training/cancel/<id>` | Cancel a running training job |
| GET | `/api/training/jobs` | List all active training jobs |
| POST | `/api/training/upload-csv` | Upload a CSV dataset file |

### Error Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/errors` | Report a mispronunciation |
| GET | `/api/errors` | List all error reports |
| DELETE | `/api/errors/<id>` | Delete an error report |
| PUT | `/api/errors/<id>/status` | Update report status |

### Settings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/settings/keys` | Get masked API key values |
| PUT | `/api/settings/keys` | Update API keys in `.env` |
