"""
Audio Post-Processing Utilities for TTS Training Data

Provides silence trimming (with padding), peak normalization, volume gain,
speed adjustment, resampling, and WAV conversion — all operating on raw
16-bit mono PCM byte buffers.

Ported from training_app/utils/audio.py with improvements for the
tts-training-data-generator pipeline.
"""

import io
import math
import re
import wave

import numpy as np
import scipy.signal


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove extra whitespace, newlines, and tabs — normalize for TTS."""
    if not text:
        return ""
    cleaned = re.sub(r'[\r\n\t\s]+', ' ', text)
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Silence trimming
# ---------------------------------------------------------------------------

def trim_silence(
    pcm: bytes,
    threshold_db: float = -45.0,
    chunk_ms: int = 10,
    sample_rate: int = 24000,
    pad_ms: int = 150,
) -> bytes:
    """
    Trim leading/trailing silence from raw 16-bit PCM data.

    Keeps *pad_ms* milliseconds of padding on both ends so the clip
    does not start/end abruptly — this is critical for XTTS training.
    """
    if not pcm:
        return pcm

    arr = np.frombuffer(pcm, dtype=np.int16)
    if len(arr) == 0:
        return pcm

    chunk_size = int(sample_rate * (chunk_ms / 1000.0))
    if chunk_size == 0:
        return pcm

    threshold_amp = 32768.0 * (10.0 ** (threshold_db / 20.0))
    pad_samples = int(sample_rate * (pad_ms / 1000.0))

    # --- find first non-silent chunk ---
    start_idx = 0
    for i in range(0, len(arr), chunk_size):
        chunk = arr[i : i + chunk_size]
        if np.max(np.abs(chunk)) > threshold_amp:
            start_idx = max(0, i - pad_samples)
            break

    # --- find last non-silent chunk ---
    end_idx = len(arr)
    for i in range(len(arr) - chunk_size, -1, -chunk_size):
        chunk = arr[i : i + chunk_size]
        if np.max(np.abs(chunk)) > threshold_amp:
            end_idx = min(len(arr), i + chunk_size + pad_samples)
            break

    if start_idx >= end_idx:
        return pcm

    return arr[start_idx:end_idx].tobytes()


# ---------------------------------------------------------------------------
# Peak normalization
# ---------------------------------------------------------------------------

def apply_peak_normalization(pcm: bytes, target_peak_db: float = -1.0) -> bytes:
    """Normalize PCM so the peak amplitude reaches *target_peak_db*."""
    if not pcm:
        return pcm

    arr = np.frombuffer(pcm, dtype=np.int16)
    if len(arr) == 0:
        return pcm

    peak = np.max(np.abs(arr))
    if peak == 0:
        return pcm

    target_peak_amp = 32767.0 * (10.0 ** (target_peak_db / 20.0))
    factor = target_peak_amp / peak

    arr_f = arr.astype(np.float32) * factor
    np.clip(arr_f, -32768.0, 32767.0, out=arr_f)
    return arr_f.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Volume gain
# ---------------------------------------------------------------------------

def apply_volume_gain(pcm: bytes, gain_db: float) -> bytes:
    """Apply a dB gain (positive = louder) to raw PCM data."""
    if abs(gain_db) < 1e-9 or not pcm:
        return pcm

    factor = 10.0 ** (gain_db / 20.0)
    arr = np.frombuffer(pcm, dtype=np.int16)
    arr_f = arr.astype(np.float32) * factor
    np.clip(arr_f, -32768.0, 32767.0, out=arr_f)
    return arr_f.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Speed adjustment (polyphase resampling)
# ---------------------------------------------------------------------------

def apply_speed(pcm: bytes, input_rate: int, speed: float) -> bytes:
    """
    Change playback speed via polyphase resampling.

    Uses scipy.signal.resample_poly for high-quality, anti-aliased
    rate conversion without the ringing artifacts of simple interpolation.
    """
    if abs(speed - 1.0) < 1e-9 or not pcm:
        return pcm

    arr_f = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    up = 1000
    down = int(speed * 1000)
    g = math.gcd(up, down)

    resampled = scipy.signal.resample_poly(arr_f, up // g, down // g)
    np.clip(resampled, -32768.0, 32767.0, out=resampled)
    return resampled.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Resampling (sample-rate conversion)
# ---------------------------------------------------------------------------

def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Resample 16-bit PCM from *src_rate* to *dst_rate* Hz."""
    if src_rate == dst_rate or not pcm:
        return pcm

    arr_f = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)

    g = math.gcd(dst_rate, src_rate)
    up = dst_rate // g
    down = src_rate // g

    resampled = scipy.signal.resample_poly(arr_f, up, down)
    np.clip(resampled, -32768.0, 32767.0, out=resampled)
    return resampled.astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def pcm_to_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM into an in-memory WAV container."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buffer.getvalue()


def save_wav(output_path: str, pcm_data: bytes, sample_rate: int = 24000):
    """Save raw 16-bit mono PCM data as a WAV file on disk."""
    import os

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


# ---------------------------------------------------------------------------
# TTS prompt builder
# ---------------------------------------------------------------------------

def build_tts_prompt(
    text: str,
    language: str = "Turkish (Turkey)",
    style_instructions: str = "",
) -> str:
    """
    Build a structured system+user prompt for Gemini TTS.

    Providing explicit role, language, and style instructions improves
    pronunciation accuracy and consistency across clips.
    """
    lines = [
        "You are a text-to-speech generator.",
        "Read the exact text provided in the target language.",
        f"Target language: {language}.",
    ]
    if style_instructions:
        lines.append(f"Style instructions: {style_instructions}")
    lines.append("Text to speak:")
    lines.append(text)
    return "\n".join(lines)
