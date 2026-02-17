# Training Data Generator Backend

Flask API for generating TTS training data.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API keys in `.env`:
```
OPENAI_API_KEY=sk-your-key-here
GOOGLE_APPLICATION_CREDENTIALS=google_credentials.json
```

4. Add Google Cloud service account JSON as `google_credentials.json`

5. Run the server:
```bash
python app.py
```

Server runs on http://localhost:5001

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/health | Health check |
| POST | /api/generate-sentences | Generate sentences with GPT |
| POST | /api/generate-audio | Generate .wav files |
| GET | /api/audio/{id}/play | Stream audio playback |
| GET | /api/items | List training items |
| DELETE | /api/items/{id} | Delete item |
| GET | /api/stats | Get statistics |
| POST | /api/export | Export metadata.csv |
