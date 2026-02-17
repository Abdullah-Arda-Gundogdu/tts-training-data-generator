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
| **TTS Client** | `App.jsx` | Main data generator — sentence generation, review, audio synthesis, folder management |
| **Training** | `pages/TrainingPage.jsx` | XTTS v2 fine-tuning — folder selection, hyperparameters, live training console |
| **Models** | `pages/ModelsPage.jsx` | Model registry — browse trained models, run inference, test and compare |
| **Settings** | `pages/SettingsPage.jsx` | API keys, LLM provider config, voice parameters |

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
