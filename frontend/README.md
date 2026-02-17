# Frontend — React + Vite

React single-page application for the TTS Training Data Generator. Communicates with the Flask backend via REST API.

## Setup

```bash
npm install
```

## Running

```bash
npm run dev
# Runs on http://localhost:5173
```

The frontend expects the backend to be running on the same hostname, port `5001`. This is auto-detected via `window.location.hostname`.

## Pages

| Page | File | Description |
|------|------|-------------|
| **Veri Üretimi** (TTS Client) | `App.jsx` | Main data generator — sentence generation, review, audio synthesis, folder management with search |
| **Eğitim** (Training) | `pages/TrainingPage.jsx` | XTTS v2 fine-tuning — folder selection, hyperparameters, live training console |
| **Modeller** (Models) | `pages/ModelsPage.jsx` | Model registry — browse, filter (completed/training/failed/cancelled), test inference, set default base model |
| **Ayarlar** (Settings) | `pages/SettingsPage.jsx` | API key management, LLM provider, TTS model selection, voice parameters |

## Key Features

- **Optimistic UI updates** — instant feedback for provider/model switching
- **Folder search** — filter word folders by name
- **TTS prompt control** — natural language prompts for Gemini model speech style
- **Model status filters** — Completed, Training, Failed, Cancelled
- **Default base model** — star a model to use as training baseline
- **Error reporting** — report mispronunciations for re-generation

## Project Structure

```
src/
├── main.jsx          # Entry point (React Router)
├── App.jsx           # Main app shell & TTS client page
├── App.css           # Global styles (dark theme, components)
├── index.css         # CSS reset & base styles
├── pages/
│   ├── TrainingPage.jsx
│   ├── ModelsPage.jsx
│   └── SettingsPage.jsx
└── assets/
```

## Tech Stack

- **React 19** with functional components and hooks
- **Vite 7** for bundling and HMR
- **React Router 7** for client-side navigation
- **Lucide React** for icons
- Vanilla CSS with CSS custom properties (dark theme)

## Build

```bash
npm run build        # Production build → dist/
npm run preview      # Preview production build
npm run lint         # ESLint check
```
