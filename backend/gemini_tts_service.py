"""
Gemini TTS Service - Text-to-Speech Audio Generation using Gemini 2.5 Flash

This module handles generating .wav files from text using the Gemini 2.5 Flash TTS API.
It mirrors the interface of google_tts_service.py for seamless provider switching.
"""

import os
import wave
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

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
DEFAULT_OUTPUT_DIR = "training_output"

# Gemini TTS outputs at 24kHz, 16-bit, mono
GEMINI_SAMPLE_RATE = 24000
GEMINI_SAMPLE_WIDTH = 2
GEMINI_CHANNELS = 1

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


def _save_wav(output_path: str, pcm_data: bytes):
    """
    Save raw PCM data as a WAV file.
    
    Gemini TTS returns raw PCM audio at 24kHz, 16-bit, mono.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(GEMINI_CHANNELS)
        wf.setsampwidth(GEMINI_SAMPLE_WIDTH)
        wf.setframerate(GEMINI_SAMPLE_RATE)
        wf.writeframes(pcm_data)


def synthesize_speech(
    text: str,
    output_path: str,
    voice_name: str = DEFAULT_VOICE,
    language_code: str = "tr-TR",
    sample_rate: int = 24000,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    volume_gain_db: float = 0.0
) -> Dict:
    """
    Generate a .wav file from text using Gemini TTS.
    Includes automatic retry with backoff for rate limit (429) errors.
    
    Args:
        text: The text to synthesize
        output_path: Full path for the output .wav file
        voice_name: Gemini voice name (e.g., 'Kore', 'Puck')
        language_code: Language code (Gemini auto-detects, this is ignored)
        sample_rate: Ignored — Gemini always outputs at 24kHz
        speaking_rate: Ignored — use natural language prompts for pacing
        pitch: Ignored — use natural language prompts for pitch
        volume_gain_db: Ignored — use natural language prompts for volume
    
    Returns:
        Dict with file info (path, duration estimate, etc.)
    """
    from google.genai import types
    
    client = get_client()
    
    last_error = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🔊 [Gemini] Generating audio for: {text[:50]}... (Voice: {voice_name})" + 
                  (f" [attempt {attempt}/{MAX_RETRIES}]" if attempt > 1 else ""))
            
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name,
                            )
                        )
                    ),
                )
            )
            
            # Extract PCM audio data
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Save as WAV
            _save_wav(output_path, pcm_data)
            
            # Calculate duration
            file_size = os.path.getsize(output_path)
            # WAV header is 44 bytes, subtract for accurate PCM duration
            pcm_size = file_size - 44
            duration_seconds = pcm_size / (GEMINI_SAMPLE_RATE * GEMINI_SAMPLE_WIDTH * GEMINI_CHANNELS)
            
            print(f"✅ [Gemini] Audio saved: {output_path} ({duration_seconds:.1f}s)")
            
            return {
                "success": True,
                "path": output_path,
                "text": text,
                "voice": voice_name,
                "duration_seconds": round(duration_seconds, 2),
                "file_size_bytes": file_size
            }
            
        except Exception as e:
            error_str = str(e)
            last_error = error_str
            
            # Check if it's a rate limit error (429 RESOURCE_EXHAUSTED)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Try to extract retry delay from error message
                wait_time = RATE_LIMIT_DELAY
                import re
                delay_match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
                if delay_match:
                    wait_time = int(float(delay_match.group(1))) + 2  # Add 2s buffer
                
                if attempt < MAX_RETRIES:
                    print(f"⏳ [Gemini] Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ [Gemini] Rate limit exceeded after {MAX_RETRIES} retries")
            else:
                # Non-rate-limit error, don't retry
                print(f"❌ [Gemini] TTS synthesis error: {e}")
                break
    
    return {
        "success": False,
        "error": str(last_error),
        "text": text
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
