"""
Gemini TTS Service - Text-to-Speech Audio Generation using Gemini TTS

This module handles generating .wav files from text using the Gemini TTS API.
It mirrors the interface of google_tts_service.py for seamless provider switching.

Audio pipeline: Gemini PCM → trim silence → volume gain → speed adjust
              → resample (24kHz→target) → peak normalize → save WAV
"""

import os
import wave
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

from audio_utils import (
    clean_text,
    trim_silence,
    apply_peak_normalization,
    apply_volume_gain,
    apply_speed,
    resample_pcm,
    save_wav,
    build_tts_prompt,
)

# Load environment variables
load_dotenv()

# Gemini TTS available voices with corrected genders
# Source: Google AI Studio + official documentation
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

DEFAULT_VOICE = "Kore"
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_OUTPUT_DIR = "training_output"
DEFAULT_TARGET_SAMPLE_RATE = 22050  # XTTS default

# Gemini TTS outputs at 24kHz, 16-bit, mono
GEMINI_SAMPLE_RATE = 24000
GEMINI_SAMPLE_WIDTH = 2
GEMINI_CHANNELS = 1

# Available model options
MODEL_OPTIONS = [
    "gemini-2.5-pro-preview-tts",
    "gemini-2.5-flash-preview-tts",
]

# Rate limiting: free tier allows 3 requests per minute
RATE_LIMIT_DELAY = 7  # seconds between requests
MAX_RETRIES = 3

# Global client
_client = None


def setup_gemini_credentials():
    """
    Verify that the Gemini API key is configured.
    
    Returns:
        True if API key is available, False otherwise
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and api_key != "your-gemini-api-key-here":
        print("✅ Gemini API key found")
        return True
    
    print("⚠️ Gemini API key not found. Set GEMINI_API_KEY in .env")
    return False


def get_client():
    """Get or initialize the Gemini client."""
    global _client
    if _client is None:
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key or api_key == "your-gemini-api-key-here":
                raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
            _client = genai.Client(api_key=api_key)
            print("✅ Gemini TTS client initialized")
        except ImportError:
            raise ImportError(
                "google-genai package not installed. "
                "Run: pip install google-genai"
            )
    return _client


def reset_client():
    """Reset the client (useful when API key changes)."""
    global _client
    _client = None


def get_available_voices() -> Dict[str, str]:
    """Return available Gemini TTS voice options."""
    return GEMINI_VOICES


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
    voice_name: str = DEFAULT_VOICE,
    model: str = DEFAULT_MODEL,
    language_code: str = "Turkish (Turkey)",
    sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    volume_gain_db: float = 0.0,
    style_instructions: str = "",
) -> Dict:
    """
    Generate a .wav file from text using Gemini TTS.

    Audio pipeline:
      1. Clean text (normalize whitespace)
      2. Build structured TTS prompt (language + style)
      3. Call Gemini API → raw PCM at 24 kHz
      4. Trim silence (keep 50 ms padding at both ends)
      5. Apply volume gain
      6. Apply speed adjustment (polyphase resampling)
      7. Resample to target sample rate (default 22050 Hz for XTTS)
      8. Peak-normalize to −1 dB
      9. Save WAV

    Args:
        text: The text to synthesize
        output_path: Full path for the output .wav file
        voice_name: Gemini voice name (e.g., 'Kore', 'Puck')
        model: Gemini model (e.g., 'gemini-2.5-flash-preview-tts')
        language_code: Language for TTS prompt (e.g., 'Turkish (Turkey)')
        sample_rate: Target sample rate for the saved WAV (default 22050)
        speaking_rate: Playback speed multiplier (1.0 = normal)
        pitch: Ignored — use style_instructions for pitch changes
        volume_gain_db: Volume gain in dB (0 = no change)
        style_instructions: Extra style hints for the TTS prompt

    Returns:
        Dict with file info (path, duration estimate, etc.)
    """
    from google.genai import types

    client = get_client()

    # --- pre-process text ---
    text = clean_text(text)
    if not text:
        return {"success": False, "error": "Empty text after cleaning", "text": text}

    prompt = build_tts_prompt(
        text=text,
        language=language_code,
        style_instructions=style_instructions,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(
                f"🔊 [Gemini] Generating audio for: {text[:50]}... "
                f"(Voice: {voice_name}, Model: {model})"
                + (f" [attempt {attempt}/{MAX_RETRIES}]" if attempt > 1 else "")
            )

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                ),
            )

            # Extract raw PCM audio data (24 kHz, 16-bit, mono)
            pcm_data = response.candidates[0].content.parts[0].inline_data.data

            # ── Post-processing pipeline ──
            # 1. Skip silence trimming — Gemini can cut off final syllables
            #    A bit of silence at edges is fine for XTTS training
            pcm_processed = pcm_data

            # 1b. Add 200ms silence padding at the end to prevent cut-off
            import numpy as np
            end_pad = np.zeros(int(GEMINI_SAMPLE_RATE * 0.2), dtype=np.int16)
            pcm_processed = pcm_processed + end_pad.tobytes()

            # 2. Volume gain
            pcm_processed = apply_volume_gain(pcm_processed, volume_gain_db)

            # 3. Speed adjustment
            pcm_processed = apply_speed(pcm_processed, GEMINI_SAMPLE_RATE, speaking_rate)

            # 4. Resample to target rate (e.g. 24 kHz → 22050 Hz)
            pcm_processed = resample_pcm(pcm_processed, GEMINI_SAMPLE_RATE, sample_rate)

            # 5. Peak-normalize to −1 dB for consistent loudness
            pcm_processed = apply_peak_normalization(pcm_processed, target_peak_db=-1.0)

            # 6. Save final WAV
            save_wav(output_path, pcm_processed, sample_rate=sample_rate)

            # Calculate duration from processed PCM
            duration_seconds = len(pcm_processed) / (sample_rate * GEMINI_SAMPLE_WIDTH * GEMINI_CHANNELS)
            file_size = os.path.getsize(output_path)

            print(f"✅ [Gemini] Audio saved: {output_path} ({duration_seconds:.1f}s, {sample_rate}Hz)")

            return {
                "success": True,
                "path": output_path,
                "text": text,
                "voice": voice_name,
                "model": model,
                "duration_seconds": round(duration_seconds, 2),
                "file_size_bytes": file_size,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            error_str = str(e)
            last_error = error_str

            # Check if it's a rate limit error (429 RESOURCE_EXHAUSTED)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = RATE_LIMIT_DELAY
                import re
                delay_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
                if delay_match:
                    wait_time = int(float(delay_match.group(1))) + 2

                if attempt < MAX_RETRIES:
                    print(f"⏳ [Gemini] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ [Gemini] Rate limit exceeded after {MAX_RETRIES} retries")
            else:
                print(f"❌ [Gemini] TTS synthesis error: {e}")
                break

    return {
        "success": False,
        "error": str(last_error),
        "text": text,
    }


def batch_synthesize(
    items: List[Dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    voice_name: str = DEFAULT_VOICE
) -> List[Dict]:
    """
    Generate .wav files for multiple sentences.
    
    Args:
        items: List of dicts with 'text' and optionally 'word' keys
        output_dir: Directory to save .wav files
        voice_name: Gemini voice to use
    
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
            voice_name=voice_name
        )
        
        result["index"] = i
        result["word"] = item.get("word", "")
        results.append(result)
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\n📊 [Gemini] Batch complete: {successful}/{len(items)} files generated")
    
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
