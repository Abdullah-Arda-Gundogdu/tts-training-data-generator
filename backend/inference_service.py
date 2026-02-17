"""
Model Inference Service - XTTS TTS inference using trained models.

Loads a fine-tuned XTTS model and generates speech from text.
"""

import os
import io
import tempfile
import wave
import numpy as np
import torch
import traceback


# Cache loaded models to avoid reloading on every request
_loaded_models = {}  # {model_id: {"model": model, "gpt_cond_latent": ..., "speaker_embedding": ...}}


def run_inference(
    model_path: str,
    config_path: str = None,
    text: str = "",
    language: str = "tr",
    speaker_reference: str = None,
    model_id: int = None
) -> dict:
    """
    Run TTS inference with a trained XTTS model.
    
    Args:
        model_path: Path to the fine-tuned model.pth checkpoint
        config_path: Path to config.json (optional, will use defaults)
        text: Text to synthesize
        language: Language code (e.g., 'tr', 'en')
        speaker_reference: Path to speaker reference WAV for voice cloning
        model_id: Model ID for caching
    
    Returns:
        dict with 'success', 'audio_bytes' (WAV bytes), 'message'
    """
    try:
        if not text or not text.strip():
            return {"success": False, "audio_bytes": None, "message": "Text is required"}
        
        if not model_path or not os.path.isfile(model_path):
            return {"success": False, "audio_bytes": None, "message": f"Model file not found: {model_path}"}
        
        # Check cache
        cache_key = model_id or model_path
        if cache_key not in _loaded_models:
            print(f"🔊 Loading model from: {model_path}")
            
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Load config
            config = XttsConfig()
            if config_path and os.path.isfile(config_path):
                config.load_json(config_path)
            else:
                # Try to find config.json next to model.pth
                model_dir = os.path.dirname(model_path)
                auto_config = os.path.join(model_dir, "config.json")
                if os.path.isfile(auto_config):
                    config.load_json(auto_config)
                else:
                    # Use default XTTS v2 config
                    print("⚠️ No config.json found, using defaults")
            
            model = Xtts.init_from_config(config)
            
            # Load checkpoint
            model_dir = os.path.dirname(model_path)
            vocab_path = os.path.join(model_dir, "vocab.json")
            if not os.path.isfile(vocab_path):
                # Search common locations for vocab.json
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                search_dirs = [
                    os.path.join(backend_dir, "xtts_v2"),
                    os.path.join(backend_dir, "xtts_base_files"),
                ]
                found = False
                for search_dir in search_dirs:
                    candidate = os.path.join(search_dir, "vocab.json")
                    if os.path.isfile(candidate):
                        vocab_path = candidate
                        found = True
                        break

            
            final_vocab = vocab_path if (vocab_path and os.path.isfile(vocab_path)) else None
            if final_vocab:
                print(f"📝 Using vocab from: {final_vocab}")
            else:
                print("⚠️ No vocab.json found, model may not work correctly")
            
            model.load_checkpoint(
                config,
                checkpoint_dir=model_dir,
                checkpoint_path=model_path,
                vocab_path=final_vocab,
                eval=True,
                use_deepspeed=False
            )
            
            if torch.cuda.is_available():
                model.cuda()
            
            # Compute speaker embedding
            gpt_cond_latent = None
            speaker_embedding = None
            
            if speaker_reference and os.path.isfile(speaker_reference):
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=[speaker_reference],
                    gpt_cond_len=30,
                    gpt_cond_chunk_len=4,
                    max_ref_length=60
                )
            else:
                # Try to find a reference WAV in the model's own directory first
                model_dir = os.path.dirname(model_path)
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                ref_wav = None
                
                # 1) Look for WAV files next to the model
                for f in os.listdir(model_dir):
                    if f.endswith('.wav'):
                        ref_wav = os.path.join(model_dir, f)
                        break
                
                # 2) Fall back to training_output folders
                if not ref_wav:
                    training_output = os.path.join(backend_dir, "training_output")
                    if os.path.isdir(training_output):
                        for folder in os.listdir(training_output):
                            folder_path = os.path.join(training_output, folder)
                            if os.path.isdir(folder_path):
                                for f in os.listdir(folder_path):
                                    if f.endswith('.wav'):
                                        ref_wav = os.path.join(folder_path, f)
                                        break
                            if ref_wav:
                                break
                
                if ref_wav:
                    print(f"📎 Using reference wav: {ref_wav}")
                    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                        audio_path=[ref_wav],
                        gpt_cond_len=30,
                        gpt_cond_chunk_len=4,
                        max_ref_length=60
                    )
                else:
                    return {"success": False, "audio_bytes": None, 
                            "message": "No speaker reference found. Please provide a reference WAV file."}
            
            _loaded_models[cache_key] = {
                "model": model,
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding
            }
            print(f"✅ Model loaded and cached (key: {cache_key})")
        
        # Get cached model
        cached = _loaded_models[cache_key]
        model = cached["model"]
        gpt_cond_latent = cached["gpt_cond_latent"]
        speaker_embedding = cached["speaker_embedding"]
        
        # Generate speech
        print(f"🔊 Generating speech: '{text[:50]}...' (language: {language})")
        
        out = model.inference(
            text=text,
            language=language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
            length_penalty=1.0,
            repetition_penalty=10.0,
            top_k=50,
            top_p=0.85,
        )
        
        # Convert to WAV bytes
        audio_data = out["wav"]
        
        # torch tensor to numpy
        if hasattr(audio_data, 'cpu'):
            audio_data = audio_data.cpu().numpy()
        
        # Normalize to int16
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV to bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)  # XTTS default sample rate
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        print(f"✅ Speech generated: {len(audio_int16)} samples")
        
        return {
            "success": True,
            "audio_bytes": wav_buffer.read(),
            "message": "Speech generated successfully"
        }
    
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "audio_bytes": None,
            "message": error_msg
        }


def clear_model_cache(model_id=None):
    """Clear cached models to free memory."""
    global _loaded_models
    if model_id:
        _loaded_models.pop(model_id, None)
    else:
        _loaded_models.clear()
    print("🗑️ Model cache cleared")
