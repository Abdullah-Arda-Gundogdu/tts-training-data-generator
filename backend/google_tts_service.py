"""
Google Cloud TTS Service - Unified Text-to-Speech Audio Generation

This module handles generating .wav files from text using Google Cloud TTS API.
Supports 4 model families:
  - Gemini Flash TTS (gemini-2.5-flash-tts)
  - Gemini Pro TTS (gemini-2.5-pro-tts)
  - Gemini 2.5 Flash Lite Preview TTS (gemini-2.5-flash-lite-preview-tts)
  - Chirp 3: HD (chirp3_hd)
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from google.cloud import texttospeech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# Gemini TTS voices (shared across Gemini Flash, Pro, and Flash Lite models)
GEMINI_VOICES = {
    "Achernar": "Female (Achernar - Clear, Friendly)",
    "Achird": "Male (Achird - Youthful, Inquisitive)",
    "Algenib": "Male (Algenib - Warm, Confident)",
    "Alnilam": "Male (Alnilam - Energetic, Excited)",
    "Aoede": "Female (Aoede - Conversational, Thoughtful)",
    "Autonoe": "Female (Autonoe - Mature, Resonant)",
    "Callirrhoe": "Female (Callirrhoe - Professional, Energetic)",
    "Charon": "Male (Charon - Deep, Authoritative)",
    "Despina": "Female (Despina - Warm, Inviting)",
    "Enceladus": "Male (Enceladus - Breathy, Calm)",
    "Erinome": "Female (Erinome - Expressive, Dynamic)",
    "Fenrir": "Male (Fenrir - Bold, Intense)",
    "Gacrux": "Female (Gacrux - Smooth, Articulate)",
    "Iapetus": "Male (Iapetus - Friendly, Casual)",
    "Kore": "Female (Kore - Clear, Versatile)",
    "Laomedeia": "Female (Laomedeia - Conversational, Engaging)",
    "Leda": "Female (Leda - Professional, Calm)",
    "Orus": "Male (Orus - Mature, Thoughtful)",
    "Puck": "Male (Puck - Upbeat, Energetic)",
    "Rasalgethi": "Male (Rasalgethi - Warm, Composed)",
    "Sadachbia": "Male (Sadachbia - Steady, Clear)",
    "Sadaltager": "Male (Sadaltager - Friendly, Enthusiastic)",
    "Schedar": "Male (Schedar - Informal, Approachable)",
    "Sulafat": "Female (Sulafat - Confident, Articulate)",
    "Umbriel": "Male (Umbriel - Smooth, Measured)",
    "Vindemiatrix": "Female (Vindemiatrix - Elegant, Poised)",
    "Zephyr": "Female (Zephyr - Light, Airy)",
    "Zubenelgenubi": "Male (Zubenelgenubi - Rich, Commanding)",
}

# Chirp 3 HD voices (Turkish-specific with language prefix)
CHIRP3_HD_VOICES = {
    "tr-TR-Chirp3-HD-Leda": "Female (Chirp3 HD Leda)",
    "tr-TR-Chirp3-HD-Orus": "Male (Chirp3 HD Orus)",
    "tr-TR-Chirp3-HD-Puck": "Male (Chirp3 HD Puck)",
    "tr-TR-Chirp3-HD-Pulcherrima": "Female (Chirp3 HD Pulcherrima)",
    "tr-TR-Chirp3-HD-Rasalgethi": "Male (Chirp3 HD Rasalgethi)",
    "tr-TR-Chirp3-HD-Sadachbia": "Male (Chirp3 HD Sadachbia)",
    "tr-TR-Chirp3-HD-Sadaltager": "Male (Chirp3 HD Sadaltager)",
    "tr-TR-Chirp3-HD-Schedar": "Male (Chirp3 HD Schedar)",
    "tr-TR-Chirp3-HD-Sulafat": "Female (Chirp3 HD Sulafat)",
    "tr-TR-Chirp3-HD-Umbriel": "Male (Chirp3 HD Umbriel)",
    "tr-TR-Chirp3-HD-Vindemiatrix": "Female (Chirp3 HD Vindemiatrix)",
    "tr-TR-Chirp3-HD-Zephyr": "Female (Chirp3 HD Zephyr)",
    "tr-TR-Chirp3-HD-Zubenelgenubi": "Male (Chirp3 HD Zubenelgenubi)",
}

# All available TTS models
TTS_MODELS = {
    "gemini_flash": {
        "label": "Gemini Flash TTS",
        "model_name": "gemini-2.5-flash-tts",
        "description": "Fast, high-quality Gemini TTS",
        "voices": GEMINI_VOICES,
        "default_voice": "Kore",
        "supports_ssml_params": False,
    },
    "chirp3_hd": {
        "label": "Chirp 3: HD",
        "model_name": None,
        "description": "Classic high-definition TTS",
        "voices": CHIRP3_HD_VOICES,
        "default_voice": "tr-TR-Chirp3-HD-Leda",
        "supports_ssml_params": True,
    },
    "gemini_pro": {
        "label": "Gemini Pro TTS",
        "model_name": "gemini-2.5-pro-tts",
        "description": "Highest quality Gemini TTS",
        "voices": GEMINI_VOICES,
        "default_voice": "Kore",
        "supports_ssml_params": False,
    },
    "gemini_flash_lite": {
        "label": "Gemini 2.5 Flash Lite Preview TTS",
        "model_name": "gemini-2.5-flash-lite-preview-tts",
        "description": "Lightweight preview model",
        "voices": GEMINI_VOICES,
        "default_voice": "Kore",
        "supports_ssml_params": False,
    },
}

DEFAULT_MODEL = "gemini_pro"
DEFAULT_OUTPUT_DIR = "training_output"

# Global client
_client = None


def setup_google_credentials(credentials_path: str = "google_credentials.json"):
    """
    Set up Google Cloud credentials from a service account JSON file.
    
    Args:
        credentials_path: Path to the Google Cloud service account JSON file
    """
    # Check current directory
    if os.path.exists(credentials_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(credentials_path)
        print(f"✅ Google Cloud credentials loaded: {credentials_path}")
        return True
        
    # Check directory of this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, credentials_path)
    if os.path.exists(script_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = script_path
        print(f"✅ Google Cloud credentials loaded: {script_path}")
        return True
        
    # Check if already set via environment variable
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print(f"✅ Google Cloud credentials from environment variable")
        return True
        
    print(f"⚠️ Google credentials file not found: {credentials_path}")
    return False


def get_client():
    """Get or initialize the Google TTS client."""
    global _client
    if _client is None:
        setup_google_credentials()
        _client = texttospeech.TextToSpeechClient()
        print("✅ Google TTS client initialized")
    return _client


def get_available_models() -> Dict:
    """Return all available TTS model definitions."""
    return {
        key: {
            "label": model["label"],
            "description": model["description"],
            "supports_ssml_params": model["supports_ssml_params"],
            "default_voice": model["default_voice"],
        }
        for key, model in TTS_MODELS.items()
    }


def get_available_voices(model_key: str = None) -> Dict[str, str]:
    """
    Return available voice options for a given model.
    
    Args:
        model_key: One of 'gemini_flash', 'gemini_pro', 'gemini_flash_lite', 'chirp3_hd'
                   If None, returns voices for the default model.
    """
    if model_key is None:
        model_key = DEFAULT_MODEL
    
    model = TTS_MODELS.get(model_key)
    if not model:
        print(f"⚠️ Unknown model key: {model_key}, falling back to {DEFAULT_MODEL}")
        model = TTS_MODELS[DEFAULT_MODEL]
    
    return model["voices"]


def sanitize_filename(text: str, max_length: int = 30) -> str:
    """
    Create a safe filename from text.
    
    Args:
        text: The text to convert to filename
        max_length: Maximum length for the filename part
    
    Returns:
        Sanitized filename string
    """
    # Take first few words
    words = text.split()[:4]
    filename = "_".join(words)
    
    # Remove/replace unsafe characters
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ('_', '-'):
            safe_chars.append(char)
        elif char == ' ':
            safe_chars.append('_')
    
    result = "".join(safe_chars)[:max_length]
    return result if result else "audio"


def synthesize_speech(
    text: str,
    output_path: str,
    voice_name: str = None,
    language_code: str = "tr-TR",
    sample_rate: int = 22050,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    volume_gain_db: float = 0.0,
    model_key: str = None,
    prompt: str = None
) -> Dict:
    """
    Generate a .wav file from text using Google Cloud TTS.
    
    Args:
        text: The text to synthesize
        output_path: Full path for the output .wav file
        voice_name: Voice name (e.g., 'Kore' for Gemini, 'tr-TR-Chirp3-HD-Leda' for Chirp3)
        language_code: Language code
        sample_rate: Audio sample rate (22050 for XTTS compatibility)
        speaking_rate: Speed of speech (0.25 to 4.0) — Chirp3 HD only
        pitch: Voice pitch (-20.0 to 20.0 semitones) — Chirp3 HD only
        volume_gain_db: Volume gain (-96.0 to 16.0 dB) — Chirp3 HD only
        model_key: TTS model key (gemini_flash, gemini_pro, gemini_flash_lite, chirp3_hd)
        prompt: Style/tone prompt for Gemini models (e.g., 'Speak slowly and clearly')
    
    Returns:
        Dict with file info (path, duration estimate, etc.)
    """
    try:
        client = get_client()
        
        # Resolve model
        if model_key is None:
            model_key = DEFAULT_MODEL
        model_def = TTS_MODELS.get(model_key, TTS_MODELS[DEFAULT_MODEL])
        
        # Default voice for this model if not specified
        if voice_name is None:
            voice_name = model_def["default_voice"]
        
        # Set up synthesis input — with optional prompt for Gemini models
        input_params = {"text": text}
        if prompt and model_def["model_name"]:
            # Gemini models support a prompt field for style control
            input_params["prompt"] = prompt
        synthesis_input = texttospeech.SynthesisInput(**input_params)
        
        # Configure voice — with or without model_name
        voice_params = {
            "language_code": language_code,
            "name": voice_name,
        }
        
        # Add model parameter for Gemini models
        if model_def["model_name"]:
            voice_params["model_name"] = model_def["model_name"]
        
        voice = texttospeech.VoiceSelectionParams(**voice_params)
        
        # Configure audio output
        audio_config_params = {
            "audio_encoding": texttospeech.AudioEncoding.LINEAR16,
            "sample_rate_hertz": sample_rate,
        }
        
        # Only add SSML parameters for models that support them (Chirp3 HD)
        if model_def["supports_ssml_params"]:
            audio_config_params["speaking_rate"] = speaking_rate
            audio_config_params["pitch"] = pitch
            audio_config_params["volume_gain_db"] = volume_gain_db
        
        audio_config = texttospeech.AudioConfig(**audio_config_params)
        
        # Make the API request
        model_label = model_def["label"]
        print(f"🔊 [{model_label}] Generating audio for: {text[:50]}... (Voice: {voice_name})")
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the audio file
        with open(output_path, "wb") as out:
            out.write(response.audio_content)
        
        # Calculate approximate duration (rough estimate)
        file_size = os.path.getsize(output_path)
        # For LINEAR16: bytes per second = sample_rate * 2 (16-bit = 2 bytes)
        duration_seconds = file_size / (sample_rate * 2)
        
        print(f"✅ [{model_label}] Audio saved: {output_path} ({duration_seconds:.1f}s)")
        
        return {
            "success": True,
            "path": output_path,
            "text": text,
            "voice": voice_name,
            "model": model_key,
            "duration_seconds": round(duration_seconds, 2),
            "file_size_bytes": file_size
        }
        
    except Exception as e:
        print(f"❌ TTS synthesis error: {e}")
        return {
            "success": False,
            "error": str(e),
            "text": text
        }


def batch_synthesize(
    items: List[Dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    voice_name: str = None,
    model_key: str = None
) -> List[Dict]:
    """
    Generate .wav files for multiple sentences.
    
    Args:
        items: List of dicts with 'text' and optionally 'word' keys
        output_dir: Directory to save .wav files
        voice_name: Voice to use
        model_key: TTS model key
    
    Returns:
        List of result dicts with file paths and status
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, item in enumerate(items):
        text = item.get("text", "")
        if not text:
            results.append({"success": False, "error": "Empty text", "index": i})
            continue
        
        # Generate filename
        text_part = sanitize_filename(text)
        filename = f"train_{timestamp}_{i:03d}_{text_part}.wav"
        output_path = os.path.join(output_dir, filename)
        
        # Synthesize
        result = synthesize_speech(
            text=text,
            output_path=output_path,
            voice_name=voice_name,
            model_key=model_key
        )
        
        result["index"] = i
        result["word"] = item.get("word", "")
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\n📊 Batch complete: {successful}/{len(items)} files generated")
    
    return results


def generate_training_filename(text: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """
    Generate a standardized training filename.
    
    Args:
        text: The sentence text
        output_dir: Output directory
    
    Returns:
        Full path for the training file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_part = sanitize_filename(text)
    filename = f"train_{timestamp}_{text_part}.wav"
    return os.path.join(output_dir, filename)
