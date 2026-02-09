"""
Google TTS Service - Text-to-Speech Audio Generation

This module handles generating .wav files from text using Google Cloud TTS API.
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from google.cloud import texttospeech
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Available Turkish voices
TURKISH_VOICES = {
    # Chirp 3 HD (New - Most Realistic)
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

    # Wavenet (High Quality)
    "tr-TR-Wavenet-A": "Female (Wavenet A)",
    "tr-TR-Wavenet-B": "Male (Wavenet B)", 
    "tr-TR-Wavenet-C": "Female (Wavenet C)",
    "tr-TR-Wavenet-D": "Female (Wavenet D)",
    "tr-TR-Wavenet-E": "Male (Wavenet E)",

    # Standard (Basic)
    "tr-TR-Standard-A": "Female (Standard A)",
    "tr-TR-Standard-B": "Male (Standard B)",
    "tr-TR-Standard-C": "Female (Standard C)",
    "tr-TR-Standard-D": "Female (Standard D)",
    "tr-TR-Standard-E": "Male (Standard E)",
}

DEFAULT_VOICE = "tr-TR-Wavenet-D"
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


def get_available_voices() -> Dict[str, str]:
    """Return available Turkish voice options."""
    return TURKISH_VOICES


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
    language_code: str = "tr-TR",
    sample_rate: int = 22050,
    speaking_rate: float = 1.0,
    pitch: float = 0.0,
    volume_gain_db: float = 0.0
) -> Dict:
    """
    Generate a .wav file from text using Google TTS.
    
    Args:
        text: The text to synthesize
        output_path: Full path for the output .wav file
        voice_name: Google TTS voice name
        language_code: Language code
        sample_rate: Audio sample rate (22050 for XTTS compatibility)
        speaking_rate: Speed of speech (0.25 to 4.0)
        pitch: Voice pitch (-20.0 to 20.0 semitones)
        volume_gain_db: Volume gain (-96.0 to 16.0 dB)
    
    Returns:
        Dict with file info (path, duration estimate, etc.)
    """
    try:
        client = get_client()
        
        # Set up synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Configure voice
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name
        )
        
        # Configure audio output
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            speaking_rate=speaking_rate,
            pitch=pitch,
            volume_gain_db=volume_gain_db
        )
        
        # Make the API request
        print(f"🔊 Generating audio for: {text[:50]}... (Rate: {speaking_rate}, Pitch: {pitch}, Vol: {volume_gain_db}dB)")
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
        # For 22050Hz, 16-bit mono: ~44100 bytes per second
        duration_seconds = file_size / (sample_rate * 2)
        
        print(f"✅ Audio saved: {output_path} ({duration_seconds:.1f}s)")
        
        return {
            "success": True,
            "path": output_path,
            "text": text,
            "voice": voice_name,
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
    voice_name: str = DEFAULT_VOICE
) -> List[Dict]:
    """
    Generate .wav files for multiple sentences.
    
    Args:
        items: List of dicts with 'text' and optionally 'word' keys
        output_dir: Directory to save .wav files
        voice_name: Google TTS voice to use
    
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


if __name__ == "__main__":
    # Test the service
    print("Testing Google TTS Service...")
    
    try:
        # Test single synthesis
        test_text = "Bu bir test cümlesidir."
        output_path = "training_output/test_tts.wav"
        
        result = synthesize_speech(test_text, output_path)
        print(f"\nTest result: {result}")
        
        if result["success"]:
            print(f"✅ Test file created: {result['path']}")
        else:
            print(f"❌ Test failed: {result.get('error')}")
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("\nMake sure google_credentials.json exists or GOOGLE_APPLICATION_CREDENTIALS is set")
