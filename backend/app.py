"""
Training Data Generator - Standalone Backend

Flask API for generating synthetic training data:
1. GPT generates sentences containing mispronounced words
2. Google TTS or Gemini TTS converts sentences to .wav files
3. Training pipeline for XTTS model fine-tuning
4. Model registry and inference
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import json
import os
import shutil
import zipfile
import io
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import services
from llm_service import (
    generate_sentences,
    regenerate_single_sentence,
    get_current_config as get_llm_config,
    set_provider as set_llm_provider,
    get_ollama_models
)
import google_tts_service
from google_tts_service import DEFAULT_OUTPUT_DIR
from training_database import (
    add_training_item,
    update_training_item,
    get_training_item,
    get_training_items,
    delete_training_item,
    get_training_stats,
    get_items_for_export,
    mark_items_exported,
    add_generation_batch,
    bulk_delete_items,
    check_existing_audio,
    add_error_report,
    get_error_reports,
    delete_error_report,
    update_error_report_status,
    delete_items_by_word
)
from model_registry import (
    get_models, get_model, add_model, update_model, delete_model,
    get_training_log, get_default_base_model, set_default_base_model
)
import training_service
import inference_service
import colloquial_normalizer

app = Flask(__name__)
CORS(app)

# Output directory
TRAINING_OUTPUT_DIR = "training_output"
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# TTS MODEL MANAGEMENT
# ============================================================================

# Current TTS model: one of the keys in google_tts_service.TTS_MODELS
_tts_model = os.getenv("TTS_MODEL", "gemini_pro")

def get_tts_model():
    """Get current TTS model key."""
    return _tts_model

def set_tts_model(model_key: str):
    """Set the active TTS model."""
    global _tts_model
    valid_models = list(google_tts_service.TTS_MODELS.keys())
    if model_key not in valid_models:
        raise ValueError(f"Invalid TTS model: {model_key}. Must be one of: {valid_models}")
    _tts_model = model_key
    print(f"🔊 TTS model set to: {model_key}")
    return model_key


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "Training Data Generator",
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/generate-sentences', methods=['POST'])
def api_generate_sentences():
    """Generate sentences containing a mispronounced word using GPT."""
    try:
        data = request.get_json()
        
        if not data or 'word' not in data:
            return jsonify({"error": "word field is required"}), 400
        
        word = data['word'].strip()
        if not word:
            return jsonify({"error": "word cannot be empty"}), 400
        
        count = max(int(data.get('count', 5)), 1)
        context = data.get('context')
        provider = data.get('provider')  # Optional: 'openai' or 'ollama'
        model = data.get('model')  # Optional: Ollama model name
        system_prompt = data.get('system_prompt')  # Optional: custom system prompt
        full_prompt = data.get('full_prompt')  # Optional: full prompt override
        
        # If Ollama provider specified with model, set it before generating
        if provider == 'ollama' and model:
            set_llm_provider(provider, model)
        elif provider:
            set_llm_provider(provider)
        
        sentences = generate_sentences(
            word=word,
            count=count,
            context=context,
            provider=provider,
            system_prompt=system_prompt,
            full_prompt=full_prompt
        )
        
        add_generation_batch(word=word, sentence_count=len(sentences))
        
        return jsonify({
            "success": True,
            "word": word,
            "sentences": sentences,
            "count": len(sentences)
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Sentence generation error: {e}")
        return jsonify({"error": f"Failed to generate sentences: {str(e)}"}), 500


@app.route('/api/regenerate-sentence', methods=['POST'])
def api_regenerate_sentence():
    """Regenerate a single sentence."""
    try:
        data = request.get_json()
        
        if not data or 'word' not in data:
            return jsonify({"error": "word field is required"}), 400
        
        word = data['word'].strip()
        existing = data.get('existing_sentences', [])
        context = data.get('context')
        
        sentence = regenerate_single_sentence(
            word=word,
            existing_sentences=existing,
            context=context
        )
        
        return jsonify({
            "success": True,
            "word": word,
            "sentence": sentence
        })
        
    except Exception as e:
        print(f"❌ Sentence regeneration error: {e}")
        return jsonify({"error": f"Failed to regenerate sentence: {str(e)}"}), 500


@app.route('/api/batch-process', methods=['POST'])
def api_batch_process():
    """Batch process: for each word, generate sentences + audio automatically."""
    try:
        data = request.get_json()
        
        words = data.get('words', [])
        sentences_per_word = int(data.get('sentencesPerWord', 15))
        context = data.get('context', '').strip() or None
        
        if not words:
            return jsonify({"error": "words array is required"}), 400
        
        # Get TTS settings
        model_key = get_tts_model()
        model_def = google_tts_service.TTS_MODELS.get(model_key, google_tts_service.TTS_MODELS[google_tts_service.DEFAULT_MODEL])
        voice = data.get('voice', model_def['default_voice'])
        speaking_rate = float(data.get('speakingRate', 1.0))
        pitch = float(data.get('pitch', 0.0))
        volume_gain_db = float(data.get('volumeGainDb', 0.0))
        tts_prompt = data.get('ttsPrompt', '').strip() or None
        
        all_results = []
        
        for word_idx, word in enumerate(words):
            word = word.strip()
            if not word:
                continue
            
            print(f"\n{'='*50}")
            print(f"📋 Batch [{word_idx+1}/{len(words)}]: Processing word '{word}' ({sentences_per_word} sentences)")
            print(f"{'='*50}")
            
            # Check existing sentences first to avoid unnecessary LLM calls
            existing_items = get_training_items(word=word, status="generated")
            existing_count = len(existing_items)
            
            if existing_count >= sentences_per_word:
                print(f"⏭️ Skipping '{word}': Already has {existing_count} generated audio files.")
                all_results.append({
                    "word": word,
                    "success": True,
                    "generated": 0,
                    "skipped": sentences_per_word,
                    "failed": 0,
                    "total_sentences": existing_count
                })
                continue
                
            needed_sentences = sentences_per_word - existing_count
            if existing_count > 0:
                print(f"  Found {existing_count} existing items. Generating {needed_sentences} more...")
            
            # Step 1: Generate sentences via LLM (with rate limit retry)
            sentences = []
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    sentences = generate_sentences(
                        word=word,
                        count=needed_sentences,
                        context=context
                    )
                    print(f"✅ Generated {len(sentences)} sentences for '{word}'")
                    
                    # Adding a base delay after a successful generation prevents hitting TPM/RPM limits too fast
                    if word_idx < len(words) - 1:
                        import time
                        time.sleep(2.5)
                        
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if ('429' in error_str or 'rate limit' in error_str or 'too many' in error_str) and attempt < max_attempts - 1:
                        # Exponential backoff: 20s, 40s, 80s, 160s
                        wait = 20 * (2 ** attempt)
                        print(f"⏳ Rate limited (429). Waiting {wait}s before retry {attempt + 1}/{max_attempts - 1}...")
                        import time
                        time.sleep(wait)
                    elif attempt < max_attempts - 1:
                        # General transient error backoff: 5s, 10s, 15s...
                        wait = 5 * (attempt + 1)
                        print(f"⚠️ Error. Waiting {wait}s before retry {attempt + 1}/{max_attempts - 1}... (Error: {e})")
                        import time
                        time.sleep(wait)
                    else:
                        print(f"❌ Failed to generate sentences for '{word}' after {max_attempts} attempts: {e}")
                        all_results.append({
                            "word": word,
                            "success": False,
                            "error": f"Sentence generation failed: {str(e)}",
                            "generated": 0
                        })
                        break
            
            if not sentences:
                continue
            
            # Step 2: For each sentence, normalize + generate audio (PARALLEL)
            def _process_batch_sentence(sentence_text):
                """Process one sentence: normalize → dedupe → TTS → DB."""
                spoken_text = None
                tts_text = sentence_text
                if colloquial_normalizer.is_enabled():
                    try:
                        norm_result = colloquial_normalizer.normalize_to_spoken(sentence_text)
                        spoken_text = norm_result.get("spoken", sentence_text)
                        tts_text = spoken_text
                        if spoken_text != sentence_text:
                            print(f"  📝 Normalized: '{sentence_text[:40]}...' → '{spoken_text[:40]}...'")
                    except Exception as e:
                        print(f"  ⚠️ Normalization failed: {e}")
                        spoken_text = sentence_text
                        tts_text = sentence_text
                
                existing = check_existing_audio(sentence_text, word)
                if existing:
                    print(f"  ⏭️ Skipping duplicate: {sentence_text[:50]}...")
                    return {"success": True, "skipped": True}
                
                safe_word = word.lower().replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip('_')
                word_folder = os.path.join(TRAINING_OUTPUT_DIR, safe_word) if safe_word else TRAINING_OUTPUT_DIR
                os.makedirs(word_folder, exist_ok=True)
                output_path = google_tts_service.generate_training_filename(tts_text, word_folder)
                
                result = google_tts_service.synthesize_speech(
                    text=tts_text,
                    output_path=output_path,
                    voice_name=voice,
                    speaking_rate=speaking_rate,
                    pitch=pitch,
                    volume_gain_db=volume_gain_db,
                    model_key=model_key,
                    prompt=tts_prompt
                )
                
                if result["success"]:
                    add_training_item(
                        word=safe_word or word,
                        sentence=sentence_text,
                        spoken_text=spoken_text,
                        wav_path=result["path"],
                        voice=voice,
                        duration_seconds=result.get("duration_seconds"),
                        status="generated"
                    )
                    return {"success": True}
                else:
                    return {"success": False, "error": result.get("error")}
            
            word_results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(_process_batch_sentence, s): s for s in sentences}
                for future in as_completed(futures):
                    try:
                        word_results.append(future.result())
                    except Exception as exc:
                        print(f"  ❌ Exception: {exc}")
                        word_results.append({"success": False, "error": str(exc)})
            
            successful = sum(1 for r in word_results if r.get("success") and not r.get("skipped"))
            skipped = sum(1 for r in word_results if r.get("skipped"))
            failed = sum(1 for r in word_results if not r.get("success"))
            
            print(f"📊 Word '{word}': {successful} generated, {skipped} skipped, {failed} failed")
            
            all_results.append({
                "word": word,
                "success": True,
                "generated": successful,
                "skipped": skipped,
                "failed": failed,
                "total_sentences": len(sentences)
            })
        
        total_generated = sum(r.get("generated", 0) for r in all_results)
        
        return jsonify({
            "success": True,
            "results": all_results,
            "total_words": len(words),
            "total_generated": total_generated,
            "message": f"Batch complete: {total_generated} audio files generated from {len(words)} words"
        })
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-audio', methods=['POST'])
def api_generate_audio():
    """Generate .wav files from sentences using the active TTS provider (parallel)."""
    try:
        data = request.get_json()
        
        if not data or 'sentences' not in data:
            return jsonify({"error": "sentences array is required"}), 400
        
        sentences = data['sentences']
        if not isinstance(sentences, list) or len(sentences) == 0:
            return jsonify({"error": "sentences must be a non-empty array"}), 400
        
        # Get the active TTS model
        model_key = get_tts_model()
        model_def = google_tts_service.TTS_MODELS.get(model_key, google_tts_service.TTS_MODELS[google_tts_service.DEFAULT_MODEL])
        
        voice = data.get('voice', model_def['default_voice'])
        speaking_rate = float(data.get('speakingRate', 1.0))
        pitch = float(data.get('pitch', 0.0))
        volume_gain_db = float(data.get('volumeGainDb', 0.0))
        tts_prompt = data.get('ttsPrompt', '').strip() or None
        
        def _process_sentence(item):
            """Process a single sentence: normalize → duplicate check → synthesize → DB insert."""
            text = item.get('text', '').strip()
            word = item.get('word', '').strip()
            
            if not text:
                return {"success": False, "error": "Empty text", "text": text}
            
            # Colloquial normalization: convert formal text to spoken form
            spoken_text = None
            tts_text = text  # Text to send to TTS
            if colloquial_normalizer.is_enabled():
                try:
                    norm_result = colloquial_normalizer.normalize_to_spoken(text)
                    spoken_text = norm_result.get("spoken", text)
                    tts_text = spoken_text  # TTS gets the spoken form
                    if spoken_text != text:
                        print(f"📝 Normalized: '{text[:40]}...' → '{spoken_text[:40]}...'")
                except Exception as e:
                    print(f"⚠️ Normalization failed, using original: {e}")
                    spoken_text = text
                    tts_text = text
            
            # Check for existing audio (duplicate prevention)
            existing = check_existing_audio(text, word)
            if existing:
                print(f"⏭️ Skipping duplicate: {text[:50]}...")
                return {
                    "success": True,
                    "skipped": True,
                    "text": text,
                    "spoken_text": spoken_text,
                    "id": existing['id'],
                    "path": existing['wav_path'],
                    "play_url": f"/api/audio/{existing['id']}/play",
                    "message": "Already exists"
                }
            
            # Create word-based subfolder (sanitize word for filesystem safety)
            safe_word = word.lower().replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip('_') if word else ''
            word_folder = os.path.join(TRAINING_OUTPUT_DIR, safe_word) if safe_word else TRAINING_OUTPUT_DIR
            os.makedirs(word_folder, exist_ok=True)
            output_path = google_tts_service.generate_training_filename(tts_text, word_folder)
            
            result = google_tts_service.synthesize_speech(
                text=tts_text,
                output_path=output_path,
                voice_name=voice,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain_db,
                model_key=model_key,
                prompt=tts_prompt
            )
            
            if result["success"]:
                item_id = add_training_item(
                    word=safe_word or word,
                    sentence=text,  # Original formal text
                    spoken_text=spoken_text,  # Colloquial spoken form
                    wav_path=result["path"],
                    voice=voice,
                    duration_seconds=result.get("duration_seconds"),
                    status="generated"
                )
                result["id"] = item_id
                result["play_url"] = f"/api/audio/{item_id}/play"
                result["spoken_text"] = spoken_text
            
            return result
        
        # Process sentences in parallel (max 10 concurrent API calls)
        results = [None] * len(sentences)
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_index = {
                executor.submit(_process_sentence, item): i
                for i, item in enumerate(sentences)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    results[idx] = {
                        "success": False,
                        "error": str(exc),
                        "text": sentences[idx].get('text', '')
                    }
        
        successful = sum(1 for r in results if r.get("success"))
        
        return jsonify({
            "success": True,
            "total": len(sentences),
            "generated": successful,
            "failed": len(sentences) - successful,
            "files": results
        })
        
    except Exception as e:
        print(f"❌ Audio generation error: {e}")
        return jsonify({"error": f"Failed to generate audio: {str(e)}"}), 500


@app.route('/api/audio/<int:item_id>/play', methods=['GET'])
def api_play_audio(item_id: int):
    """Stream audio file for playback."""
    try:
        item = get_training_item(item_id)
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        wav_path = item.get('wav_path')
        if not wav_path or not os.path.exists(wav_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        return send_file(
            wav_path,
            mimetype='audio/wav',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"❌ Audio playback error: {e}")
        return jsonify({"error": f"Failed to play audio: {str(e)}"}), 500


@app.route('/api/audio/<int:item_id>/download', methods=['GET'])
def api_download_audio(item_id: int):
    """Download audio file as .wav attachment."""
    try:
        item = get_training_item(item_id)
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        wav_path = item.get('wav_path')
        if not wav_path or not os.path.exists(wav_path):
            return jsonify({"error": "Audio file not found"}), 404
        
        # Generate a readable filename
        sentence = item.get('sentence', 'audio')[:30]
        # Sanitize filename
        safe_name = ''.join(c for c in sentence if c.isalnum() or c in ' -_').strip()
        if not safe_name:
            safe_name = f"audio_{item_id}"
        filename = f"{safe_name}.wav"
        
        return send_file(
            wav_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        print(f"❌ Audio download error: {e}")
        return jsonify({"error": f"Failed to download audio: {str(e)}"}), 500


@app.route('/api/audio/download-all', methods=['GET'])
def api_download_all_audio():
    """Download all generated audio files as a single ZIP."""
    try:
        items = get_training_items(status='generated', limit=1000)
        
        if not items:
            return jsonify({"error": "No audio files to download"}), 404
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for item in items:
                wav_path = item.get('wav_path')
                if wav_path and os.path.exists(wav_path):
                    # Use just the filename in ZIP
                    filename = os.path.basename(wav_path)
                    zip_file.write(wav_path, filename)
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"all_audio_{timestamp}.zip"
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        print(f"❌ Download all audio error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/items', methods=['GET'])
def api_get_items():
    """Get training items with optional filters."""
    try:
        word = request.args.get('word')
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 100)), 500)
        offset = int(request.args.get('offset', 0))
        
        items = get_training_items(
            word=word,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            "success": True,
            "items": items,
            "count": len(items)
        })
        
    except Exception as e:
        print(f"❌ Get items error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/items/<int:item_id>', methods=['GET'])
def api_get_item(item_id: int):
    """Get a single training item by ID."""
    try:
        item = get_training_item(item_id)
        
        if not item:
            return jsonify({"error": "Item not found"}), 404
        
        return jsonify({
            "success": True,
            "item": item
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/items/<int:item_id>', methods=['DELETE'])
def api_delete_item(item_id: int):
    """Delete a training item and its audio file."""
    try:
        deleted = delete_training_item(item_id)
        
        if not deleted:
            return jsonify({"error": "Item not found"}), 404
        
        return jsonify({
            "success": True,
            "message": "Item deleted"
        })
        
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/items/bulk-delete', methods=['POST'])
def api_bulk_delete_items():
    """Bulk delete multiple training items and their audio files."""
    try:
        data = request.get_json()
        
        if not data or 'item_ids' not in data:
            return jsonify({"error": "item_ids array is required"}), 400
        
        item_ids = data['item_ids']
        if not isinstance(item_ids, list) or len(item_ids) == 0:
            return jsonify({"error": "item_ids must be a non-empty array"}), 400
        
        deleted_count = bulk_delete_items(item_ids)
        
        return jsonify({
            "success": True,
            "deleted_count": deleted_count,
            "message": f"{deleted_count} items deleted"
        })
        
    except Exception as e:
        print(f"❌ Bulk delete error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/items/<int:item_id>', methods=['PUT'])
def api_update_item(item_id: int):
    """Update a training item."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        updated = update_training_item(item_id, **data)
        
        if not updated:
            return jsonify({"error": "Item not found or no valid fields"}), 404
        
        item = get_training_item(item_id)
        
        return jsonify({
            "success": True,
            "item": item
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_get_stats():
    """Get training data statistics."""
    try:
        stats = get_training_stats()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/voices', methods=['GET'])
def api_get_voices():
    """Get available TTS voices for the active model."""
    model_key = request.args.get('model', get_tts_model())
    return jsonify({
        "success": True,
        "voices": google_tts_service.get_available_voices(model_key),
        "model": model_key
    })


@app.route('/api/tts/config', methods=['GET'])
def api_get_tts_config():
    """Get current TTS model configuration."""
    model_key = get_tts_model()
    return jsonify({
        "success": True,
        "model": model_key,
        "models": google_tts_service.get_available_models()
    })


@app.route('/api/tts/config', methods=['POST'])
def api_set_tts_config():
    """Set TTS model."""
    try:
        data = request.get_json() or {}
        model_key = data.get('model')
        
        if not model_key:
            return jsonify({"error": "model field is required"}), 400
        
        set_tts_model(model_key)
        
        # Persist to .env
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        from dotenv import dotenv_values
        existing = dotenv_values(env_path) if os.path.exists(env_path) else {}
        existing['TTS_MODEL'] = model_key
        # Remove old TTS_PROVIDER key if present
        existing.pop('TTS_PROVIDER', None)
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write("# Environment variables for Training Data Generator\n")
            f.write("# Updated via Settings UI\n\n")
            for key, value in existing.items():
                # Quote values to handle special characters in API keys
                f.write(f'{key}="{value}"\n')
        
        return jsonify({
            "success": True,
            "model": model_key
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/llm/config', methods=['GET'])
def api_get_llm_config():
    """Get current LLM provider configuration."""
    try:
        config = get_llm_config()
        return jsonify({
            "success": True,
            "config": config
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/llm/config', methods=['POST'])
def api_set_llm_config():
    """Set LLM provider and model."""
    try:
        data = request.get_json() or {}
        provider = data.get('provider')
        model = data.get('model')
        
        if not provider:
            return jsonify({"error": "provider field is required"}), 400
        
        config = set_llm_provider(provider, model)
        return jsonify({
            "success": True,
            "config": config
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/llm/models', methods=['GET'])
def api_get_ollama_models():
    """Get available Ollama models."""
    try:
        models = get_ollama_models()
        return jsonify({
            "success": True,
            "models": models
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# BATCH TTS FROM FILE (SSE streaming)
# ============================================================================

@app.route('/api/batch-tts-from-file', methods=['POST'])
def api_batch_tts_from_file():
    """Batch TTS: upload a .txt file (one sentence per line) and generate audio for each.
    Streams progress via Server-Sent Events (SSE)."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read all lines from the uploaded text file
        raw_content = file.read()
        # Try UTF-8 first, fallback to latin-1
        try:
            text_content = raw_content.decode('utf-8-sig')  # handles BOM
        except UnicodeDecodeError:
            text_content = raw_content.decode('latin-1')

        all_lines = [line.strip() for line in text_content.splitlines() if line.strip()]

        if not all_lines:
            return jsonify({"error": "File is empty or has no valid sentences"}), 400

        # Get optional parameters from form data
        word = request.form.get('word', 'file_import').strip()
        model_key = get_tts_model()
        model_def = google_tts_service.TTS_MODELS.get(model_key, google_tts_service.TTS_MODELS[google_tts_service.DEFAULT_MODEL])
        voice = request.form.get('voice', model_def['default_voice'])
        speaking_rate = float(request.form.get('speakingRate', 1.0))
        pitch = float(request.form.get('pitch', 0.0))
        volume_gain_db = float(request.form.get('volumeGainDb', 0.0))
        tts_prompt = request.form.get('ttsPrompt', '').strip() or None

        print(f"\n{'='*60}")
        print(f"📄 Batch TTS from file: {file.filename}")
        print(f"   {len(all_lines)} sentences, folder='{word}', voice='{voice}', model='{model_key}'")
        print(f"{'='*60}")

        # Prepare the word folder
        safe_word = word.lower().replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_').strip('_') if word else 'file_import'
        word_folder = os.path.join(TRAINING_OUTPUT_DIR, safe_word)
        os.makedirs(word_folder, exist_ok=True)

        def generate_sse():
            import time
            total = len(all_lines)
            success_count = 0
            failed_count = 0
            skipped_count = 0

            # Process in small batches of 5
            BATCH_SIZE = 5

            for batch_start in range(0, total, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total)
                batch_lines = all_lines[batch_start:batch_end]

                def _process_one(idx, sentence_text):
                    """Process a single sentence: normalize → dedup → TTS → DB."""
                    try:
                        spoken_text = None
                        tts_text = sentence_text

                        # Colloquial normalization
                        if colloquial_normalizer.is_enabled():
                            try:
                                norm_result = colloquial_normalizer.normalize_to_spoken(sentence_text)
                                spoken_text = norm_result.get("spoken", sentence_text)
                                tts_text = spoken_text
                            except Exception:
                                spoken_text = sentence_text
                                tts_text = sentence_text

                        # Duplicate check
                        existing = check_existing_audio(sentence_text, safe_word)
                        if existing:
                            return {"status": "skipped", "idx": idx, "text": sentence_text[:60]}

                        output_path = google_tts_service.generate_training_filename(tts_text, word_folder)

                        # Retry with exponential backoff for rate limits
                        max_retries = 4
                        for attempt in range(max_retries + 1):
                            try:
                                result = google_tts_service.synthesize_speech(
                                    text=tts_text,
                                    output_path=output_path,
                                    voice_name=voice,
                                    speaking_rate=speaking_rate,
                                    pitch=pitch,
                                    volume_gain_db=volume_gain_db,
                                    model_key=model_key,
                                    prompt=tts_prompt
                                )
                                break
                            except Exception as e:
                                err_str = str(e).lower()
                                if ('429' in err_str or 'rate limit' in err_str) and attempt < max_retries:
                                    wait = 10 * (2 ** attempt)
                                    print(f"⏳ Rate limited. Waiting {wait}s... (attempt {attempt+1})")
                                    time.sleep(wait)
                                else:
                                    raise

                        if result.get("success"):
                            add_training_item(
                                word=safe_word,
                                sentence=sentence_text,
                                spoken_text=spoken_text,
                                wav_path=result["path"],
                                voice=voice,
                                duration_seconds=result.get("duration_seconds"),
                                status="generated"
                            )
                            return {"status": "success", "idx": idx, "text": sentence_text[:60]}
                        else:
                            return {"status": "failed", "idx": idx, "text": sentence_text[:60], "error": result.get("error", "Unknown")}
                    except Exception as e:
                        return {"status": "failed", "idx": idx, "text": sentence_text[:60], "error": str(e)}

                # Run this batch in parallel
                batch_results = [None] * len(batch_lines)
                with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                    future_to_idx = {
                        executor.submit(_process_one, batch_start + i, line): i
                        for i, line in enumerate(batch_lines)
                    }
                    for future in as_completed(future_to_idx):
                        local_idx = future_to_idx[future]
                        try:
                            batch_results[local_idx] = future.result()
                        except Exception as exc:
                            batch_results[local_idx] = {"status": "failed", "idx": batch_start + local_idx, "error": str(exc)}

                # Count results and send progress
                for r in batch_results:
                    if r and r["status"] == "success":
                        success_count += 1
                    elif r and r["status"] == "skipped":
                        skipped_count += 1
                    else:
                        failed_count += 1

                current = batch_end
                progress_data = json.dumps({
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "success": success_count,
                    "failed": failed_count,
                    "skipped": skipped_count,
                    "lastText": batch_lines[-1][:60] if batch_lines else ""
                })
                yield f"data: {progress_data}\n\n"

            # Final summary
            complete_data = json.dumps({
                "type": "complete",
                "total": total,
                "success": success_count,
                "failed": failed_count,
                "skipped": skipped_count
            })
            yield f"data: {complete_data}\n\n"
            print(f"\n📊 Batch TTS complete: {success_count} success, {failed_count} failed, {skipped_count} skipped out of {total}")

        return Response(generate_sse(), mimetype='text/event-stream', headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        })

    except Exception as e:
        print(f"❌ Batch TTS from file error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# COLLOQUIAL NORMALIZATION API
# ============================================================================

@app.route('/api/colloquial/settings', methods=['GET'])
def api_get_colloquial_settings():
    """Get colloquial normalization settings."""
    return jsonify({
        "success": True,
        "settings": colloquial_normalizer.get_settings()
    })


@app.route('/api/colloquial/settings', methods=['POST'])
def api_set_colloquial_settings():
    """Update colloquial normalization settings."""
    try:
        data = request.get_json() or {}
        enabled = data.get('enabled')
        provider = data.get('provider')

        colloquial_normalizer.update_settings(enabled=enabled, provider=provider)

        return jsonify({
            "success": True,
            "settings": colloquial_normalizer.get_settings()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/normalize-text', methods=['POST'])
def api_normalize_text():
    """Normalize a single text to colloquial spoken form (preview)."""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({"error": "text field is required"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "text cannot be empty"}), 400

        result = colloquial_normalizer.normalize_to_spoken(text)

        return jsonify({
            "success": True,
            **result
        })

    except Exception as e:
        print(f"❌ Normalization error: {e}")
        return jsonify({"error": f"Normalization failed: {str(e)}"}), 500


@app.route('/api/folders', methods=['GET'])
def api_get_folders():
    """Get list of word folders with file counts."""
    try:
        folders = []
        if os.path.exists(TRAINING_OUTPUT_DIR):
            for name in os.listdir(TRAINING_OUTPUT_DIR):
                path = os.path.join(TRAINING_OUTPUT_DIR, name)
                if os.path.isdir(path):
                    wav_files = [f for f in os.listdir(path) if f.endswith('.wav')]
                    if len(wav_files) > 0:
                        folders.append({
                            "name": name,
                            "file_count": len(wav_files)
                        })
        return jsonify({
            "success": True,
            "folders": folders
        })
    except Exception as e:
        print(f"❌ Get folders error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/folders/<folder_name>', methods=['DELETE'])
def api_delete_folder(folder_name: str):
    """Delete a word folder and all its audio files."""
    try:
        
        folder_path = os.path.join(TRAINING_OUTPUT_DIR, folder_name)
        
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            return jsonify({"error": "Folder not found"}), 404
        
        # Count files before deletion
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        file_count = len(wav_files)
        
        # Delete the folder and all its contents
        shutil.rmtree(folder_path)
        
        # Also delete database entries for this word
        deleted_db_count = delete_items_by_word(folder_name)
        
        print(f"✅ Deleted folder '{folder_name}' with {file_count} files")
        
        return jsonify({
            "success": True,
            "folder": folder_name,
            "files_deleted": file_count,
            "db_entries_deleted": deleted_db_count
        })
        
    except Exception as e:
        print(f"❌ Delete folder error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/folders/bulk-delete', methods=['POST'])
def api_bulk_delete_folders():
    """Bulk delete multiple folders and their contents."""
    try:
        
        data = request.get_json() or {}
        folder_names = data.get('folders', [])
        
        if not folder_names:
            return jsonify({"error": "No folders selected"}), 400
            
        deleted_folders = []
        total_files = 0
        total_db_entries = 0
        errors = []
        
        for folder_name in folder_names:
            try:
                folder_path = os.path.join(TRAINING_OUTPUT_DIR, folder_name)
                
                # Check if exists
                if os.path.exists(folder_path) and os.path.isdir(folder_path):
                    # Count files
                    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                    file_count = len(wav_files)
                    
                    # Delete folder
                    shutil.rmtree(folder_path)
                    
                    # Delete DB entries
                    db_count = delete_items_by_word(folder_name)
                    
                    deleted_folders.append(folder_name)
                    total_files += file_count
                    total_db_entries += db_count
                else:
                    # Even if folder doesn't exist on disk, we should clean up DB
                    db_count = delete_items_by_word(folder_name)
                    if db_count > 0:
                        deleted_folders.append(folder_name)
                        total_db_entries += db_count
                        
            except Exception as e:
                print(f"❌ Error deleting folder '{folder_name}': {e}")
                errors.append(f"{folder_name}: {str(e)}")
        
        return jsonify({
            "success": True,
            "deleted_folders": deleted_folders,
            "count": len(deleted_folders),
            "files_deleted": total_files,
            "db_entries_deleted": total_db_entries,
            "errors": errors
        })
        
    except Exception as e:
        print(f"❌ Bulk delete folders error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/folders/<folder_name>/download', methods=['GET'])
def api_download_folder(folder_name: str):
    """Download a specific folder as a ZIP file with clip_NNNN naming, wavs/ subfolder, and params.json."""
    try:
        # Get all generated items for this word to ensure we have sentences
        items = get_training_items(word=folder_name, status='generated')
        
        if not items:
            folder_path = os.path.join(TRAINING_OUTPUT_DIR, folder_name)
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                 return jsonify({"error": "Folder not found"}), 404
            return jsonify({"error": "No database entries found for this folder"}), 404

        # Get current TTS model info for params.json
        model_key = get_tts_model()
        model_def = google_tts_service.TTS_MODELS.get(model_key, google_tts_service.TTS_MODELS[google_tts_service.DEFAULT_MODEL])

        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            metadata_lines = []
            
            for index, item in enumerate(items, 1):
                wav_path = item.get('wav_path')
                original_sentence = item.get('sentence', '').strip()
                spoken_sentence = item.get('spoken_text') or original_sentence
                
                if wav_path and os.path.exists(wav_path):
                    clip_name = f"clip_{index:04d}"
                    
                    # Add WAV file inside wavs/ subfolder
                    zip_file.write(wav_path, f"wavs/{clip_name}.wav")
                    
                    # Add to metadata (clip_NNNN|raw_text|spoken_text)
                    metadata_lines.append(f"{clip_name}|{original_sentence}|{spoken_sentence}")
                    
                    # Generate per-clip params.json
                    voice_name = item.get('voice', model_def.get('default_voice', ''))
                    params_data = {
                        "model": model_def.get('model_name', model_key),
                        "voice_name": voice_name,
                        "language": "Turkish (Turkey)",
                        "audio_encoding": "LINEAR16",
                        "sample_rate": 22050,
                        "speed": 1.0,
                        "volume_gain": 0.0,
                        "dataset_audio_path": f"wavs/{clip_name}.wav"
                    }
                    zip_file.writestr(f"{clip_name}.params.json", json.dumps(params_data, indent=2, ensure_ascii=False))

            # Add metadata.csv to ZIP (UTF-8 with BOM for Windows compatibility)
            if metadata_lines:
                metadata_content = "\n".join(metadata_lines)
                zip_file.writestr("metadata.csv", ("\ufeff" + metadata_content).encode("utf-8"))
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{folder_name}_{timestamp}.zip"
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        print(f"❌ Download folder error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/folders/download', methods=['POST'])
def api_download_folders():
    """Download selected folders as a single ZIP file with clip_NNNN naming, wavs/ subfolder, and params.json."""
    try:
        data = request.get_json() or {}
        folder_names = data.get('folders', [])
        
        if not folder_names:
            return jsonify({"error": "No folders selected"}), 400
        
        # Get current TTS model info for params.json
        model_key = get_tts_model()
        model_def = google_tts_service.TTS_MODELS.get(model_key, google_tts_service.TTS_MODELS[google_tts_service.DEFAULT_MODEL])
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        # Global counter for sequential numbering across all folders
        global_counter = 1
        metadata_lines = []
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for folder_name in folder_names:
                # Fetch items for this folder/word from DB
                items = get_training_items(word=folder_name, status='generated')
                
                for item in items:
                    wav_path = item.get('wav_path')
                    original_sentence = item.get('sentence', '').strip()
                    spoken_sentence = item.get('spoken_text') or original_sentence
                    
                    if wav_path and os.path.exists(wav_path):
                        clip_name = f"clip_{global_counter:04d}"
                        
                        # Add WAV file inside wavs/ subfolder
                        zip_file.write(wav_path, f"wavs/{clip_name}.wav")
                        
                        # Add to metadata (clip_NNNN|raw_text|spoken_text)
                        metadata_lines.append(f"{clip_name}|{original_sentence}|{spoken_sentence}")
                        
                        # Generate per-clip params.json
                        voice_name = item.get('voice', model_def.get('default_voice', ''))
                        params_data = {
                            "model": model_def.get('model_name', model_key),
                            "voice_name": voice_name,
                            "language": "Turkish (Turkey)",
                            "audio_encoding": "LINEAR16",
                            "sample_rate": 22050,
                            "speed": 1.0,
                            "volume_gain": 0.0,
                            "dataset_audio_path": f"wavs/{clip_name}.wav"
                        }
                        zip_file.writestr(f"{clip_name}.params.json", json.dumps(params_data, indent=2, ensure_ascii=False))
                        
                        global_counter += 1
            
            # Add metadata.csv to ZIP (UTF-8 with BOM for Windows compatibility)
            if metadata_lines:
                metadata_content = "\n".join(metadata_lines)
                zip_file.writestr("metadata.csv", ("\ufeff" + metadata_content).encode("utf-8"))
        
        zip_buffer.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"training_data_{timestamp}.zip"
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )
        
    except Exception as e:
        print(f"❌ Download folders error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/export', methods=['POST'])
def api_export():
    """Export generated items as metadata.csv for training with clip_NNNN naming."""
    try:
        data = request.get_json() or {}
        word_filter = data.get('word')
        
        items = get_items_for_export(word=word_filter)
        
        if not items:
            return jsonify({
                "success": False,
                "error": "No generated items to export"
            }), 400
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = f"metadata_{timestamp}.csv"
        metadata_path = os.path.join(TRAINING_OUTPUT_DIR, metadata_filename)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for index, item in enumerate(items, 1):
                sentence = item['sentence']
                clip_name = f"clip_{index:04d}"
                
                # Format: clip_NNNN|Text|Text
                f.write(f"{clip_name}|{sentence}|{sentence}\n")
        
        item_ids = [item['id'] for item in items]
        mark_items_exported(item_ids)
        
        print(f"✅ Exported {len(items)} items to {metadata_path}")
        
        return jsonify({
            "success": True,
            "metadata_path": metadata_path,
            "item_count": len(items),
            "items": items
        })
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        return jsonify({"error": f"Failed to export: {str(e)}"}), 500


@app.route('/api/export/download', methods=['GET'])
def api_download_metadata():
    """Download the latest metadata.csv file."""
    try:
        metadata_files = [f for f in os.listdir(TRAINING_OUTPUT_DIR) 
                         if f.startswith('metadata_') and f.endswith('.csv')]
        
        if not metadata_files:
            return jsonify({"error": "No metadata file found. Run export first."}), 404
        
        latest = sorted(metadata_files)[-1]
        filepath = os.path.join(TRAINING_OUTPUT_DIR, latest)
        
        return send_file(
            filepath,
            mimetype='text/csv',
            as_attachment=True,
            download_name=latest
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/api/errors', methods=['POST'])
def api_add_error_report():
    """Report an error/mispronunciation."""
    try:
        data = request.get_json()
        
        if not data or 'word' not in data:
            return jsonify({"error": "word field is required"}), 400
        
        word = data['word'].strip()
        explanation = data.get('explanation', '').strip()
        
        if not word:
            return jsonify({"error": "word cannot be empty"}), 400
            
        report_id = add_error_report(word, explanation)
        
        print(f"📝 New error report: {word} - {explanation}")
        
        return jsonify({
            "success": True,
            "id": report_id,
            "message": "Error report added"
        })
        
    except Exception as e:
        print(f"❌ Add error report error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/errors', methods=['GET'])
def api_get_error_reports():
    """Get all error reports."""
    try:
        status = request.args.get('status')
        reports = get_error_reports(status)
        
        return jsonify({
            "success": True,
            "reports": reports
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/errors/<int:report_id>', methods=['DELETE'])
def api_delete_error_report(report_id: int):
    """Delete an error report."""
    try:
        deleted = delete_error_report(report_id)
        
        if not deleted:
            return jsonify({"error": "Report not found"}), 404
            
        return jsonify({
            "success": True,
            "message": "Report deleted"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/errors/<int:report_id>/status', methods=['PUT'])
def api_update_error_status(report_id: int):
    """Update status of an error report."""
    try:
        data = request.get_json()
        status = data.get('status')
        
        if not status:
            return jsonify({"error": "status required"}), 400
            
        updated = update_error_report_status(report_id, status)
        
        if not updated:
            return jsonify({"error": "Report not found"}), 404
            
        return jsonify({
            "success": True,
            "message": "Status updated"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# MODEL MANAGEMENT API
# ============================================================================

@app.route('/api/models', methods=['GET'])
def api_get_models():
    """Get all trained models."""
    try:
        status = request.args.get('status')
        models = get_models(status=status)
        return jsonify({"success": True, "models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>', methods=['GET'])
def api_get_model(model_id: int):
    """Get a single model by ID."""
    try:
        model = get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        return jsonify({"success": True, "model": model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models', methods=['POST'])
def api_create_model():
    """Register a new model manually."""
    try:
        data = request.get_json() or {}
        name = data.get('name', '').strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        model_id = add_model(
            name=name,
            description=data.get('description', ''),
            tags=data.get('tags', []),
            base_model=data.get('base_model', 'xtts_v2'),
            training_params=data.get('training_params', {}),
            dataset_csv=data.get('dataset_csv'),
            status=data.get('status', 'completed'),
            model_path=data.get('model_path')
        )
        return jsonify({"success": True, "model_id": model_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>', methods=['PUT'])
def api_update_model(model_id: int):
    """Update a model's metadata."""
    try:
        data = request.get_json() or {}
        updated = update_model(model_id, **data)
        if not updated:
            return jsonify({"error": "Model not found or no valid fields"}), 404
        model = get_model(model_id)
        return jsonify({"success": True, "model": model})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>', methods=['DELETE'])
def api_delete_model(model_id: int):
    """Delete a model and its files."""
    try:
        deleted = delete_model(model_id)
        if not deleted:
            return jsonify({"error": "Model not found"}), 404
        return jsonify({"success": True, "message": "Model deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>/test-audio', methods=['GET'])
def api_model_test_audio(model_id: int):
    """Stream test audio for a model."""
    try:
        model = get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        audio_path = model.get('test_audio_path')
        if not audio_path or not os.path.exists(audio_path):
            return jsonify({"error": "No test audio available"}), 404
        return send_file(audio_path, mimetype='audio/wav')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>/synthesize', methods=['POST'])
def api_model_synthesize(model_id: int):
    """Generate speech using a trained model."""
    try:
        model = get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.get('status') != 'completed':
            return jsonify({"error": "Model training not completed yet"}), 400

        model_path = model.get('model_path')
        if not model_path or not os.path.exists(model_path):
            return jsonify({"error": "Model file not found on disk"}), 404

        data = request.get_json() or {}
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "text is required"}), 400

        language = data.get('language', model.get('training_params', {}).get('language', 'tr'))
        speaker_ref = data.get('speaker_reference', model.get('training_params', {}).get('speaker_reference'))

        result = inference_service.run_inference(
            model_path=model_path,
            config_path=model.get('config_path'),
            text=text,
            language=language,
            speaker_reference=speaker_ref,
            model_id=model_id
        )

        if not result['success']:
            return jsonify({"error": result['message']}), 500

        return send_file(
            io.BytesIO(result['audio_bytes']),
            mimetype='audio/wav',
            as_attachment=False,
            download_name='synthesized.wav'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/models/<int:model_id>/training-log', methods=['GET'])
def api_model_training_log(model_id: int):
    """Get the training console log for a model."""
    try:
        model = get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        log = get_training_log(model_id, max_chars=500_000)
        return jsonify({"success": True, "log": log})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings/default-base-model', methods=['GET'])
def api_get_default_base_model():
    """Get the default base model for training."""
    try:
        result = get_default_base_model()
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings/default-base-model', methods=['PUT'])
def api_set_default_base_model():
    """Set the default base model for training."""
    try:
        data = request.get_json() or {}
        model_id = data.get('model_id')
        result = set_default_base_model(model_id)
        return jsonify({"success": True, **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# TRAINING API
# ============================================================================

@app.route('/api/training/start', methods=['POST'])
def api_start_training():
    """Start a training job."""
    try:
        data = request.get_json() or {}

        name = data.get('name', '').strip()
        if not name:
            return jsonify({"error": "name is required"}), 400

        selected_folders = data.get('selected_folders', [])
        dataset_path = data.get('dataset_path', '').strip()

        if not selected_folders and not dataset_path:
            return jsonify({"error": "Select folders or provide a dataset_path"}), 400

        result = training_service.start_training(
            name=name,
            description=data.get('description', ''),
            tags=data.get('tags', []),
            selected_folders=selected_folders,
            dataset_path=dataset_path,
            meta_file_train=data.get('meta_file_train', 'metadata.csv'),
            training_params=data.get('training_params', {})
        )

        return jsonify({"success": True, **result}), 202
    except Exception as e:
        print(f"❌ Start training error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/status/<int:model_id>', methods=['GET'])
def api_training_status(model_id: int):
    """Get training job status and logs."""
    try:
        status = training_service.get_training_status(model_id)
        if "error" in status:
            return jsonify(status), 404
        return jsonify({"success": True, **status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/cancel/<int:model_id>', methods=['POST'])
def api_cancel_training(model_id: int):
    """Cancel a running training job."""
    try:
        result = training_service.cancel_training(model_id)
        if "error" in result:
            return jsonify(result), 404
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/jobs', methods=['GET'])
def api_active_jobs():
    """Get all active training jobs."""
    try:
        jobs = training_service.get_all_active_jobs()
        return jsonify({"success": True, "jobs": jobs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/training/upload-csv', methods=['POST'])
def api_upload_csv():
    """Upload a CSV dataset file for training."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Save to datasets directory
        datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
        os.makedirs(datasets_dir, exist_ok=True)

        safe_name = file.filename.replace('..', '').replace('/', '').replace('\\', '')
        save_path = os.path.join(datasets_dir, safe_name)
        file.save(save_path)

        return jsonify({
            "success": True,
            "filename": safe_name,
            "path": save_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# SETTINGS / API KEYS
# ============================================================================

@app.route('/api/settings/keys', methods=['GET'])
def api_get_keys():
    """Get masked API key values."""
    try:
        from dotenv import dotenv_values
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        env_vars = dotenv_values(env_path) if os.path.exists(env_path) else {}

        def mask_key(val):
            if not val:
                return ""
            if len(val) <= 8:
                return "****"
            return val[:4] + "*" * (len(val) - 8) + val[-4:]

        keys = {
            "OPENAI_API_KEY": {
                "value": mask_key(env_vars.get("OPENAI_API_KEY", "")),
                "is_set": bool(env_vars.get("OPENAI_API_KEY")),
                "label": "OpenAI API Key"
            },
            "GOOGLE_APPLICATION_CREDENTIALS": {
                "value": env_vars.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
                "is_set": bool(env_vars.get("GOOGLE_APPLICATION_CREDENTIALS")),
                "label": "Google Cloud Credentials Path"
            },
            "OLLAMA_BASE_URL": {
                "value": env_vars.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                "is_set": bool(env_vars.get("OLLAMA_BASE_URL")),
                "label": "Ollama Base URL"
            },
            "LLM_PROVIDER": {
                "value": env_vars.get("LLM_PROVIDER", "openai"),
                "is_set": bool(env_vars.get("LLM_PROVIDER")),
                "label": "LLM Provider"
            },
            "TTS_MODEL": {
                "value": env_vars.get("TTS_MODEL", "gemini_pro"),
                "is_set": bool(env_vars.get("TTS_MODEL")),
                "label": "TTS Model"
            }
        }

        return jsonify({"success": True, "keys": keys})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/settings/keys', methods=['PUT'])
def api_set_keys():
    """Update API keys in .env file."""
    try:
        data = request.get_json() or {}
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

        # Read existing .env
        existing = {}
        if os.path.exists(env_path):
            from dotenv import dotenv_values
            existing = dotenv_values(env_path)

        # Update only provided keys
        allowed_keys = {'OPENAI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'OLLAMA_BASE_URL', 'LLM_PROVIDER', 'OLLAMA_MODEL', 'TTS_MODEL'}
        updated_keys = []

        for key, value in data.items():
            if key in allowed_keys and value is not None:
                existing[key] = value
                updated_keys.append(key)

        # Write back to .env
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write("# Environment variables for Training Data Generator\n")
            f.write("# Updated via Settings UI\n\n")
            for key, value in existing.items():
                f.write(f"{key}={value}\n")

        # Reload environment
        from dotenv import load_dotenv
        load_dotenv(env_path, override=True)

        return jsonify({
            "success": True,
            "updated_keys": updated_keys,
            "message": f"Updated {len(updated_keys)} key(s)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Training Data Generator API")
    print("="*50)
    print(f"📁 Output directory: {TRAINING_OUTPUT_DIR}")
    print("🌐 Starting server on http://localhost:5001")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
