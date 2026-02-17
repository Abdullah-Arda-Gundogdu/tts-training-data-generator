"""
Training Data Generator - Standalone Backend

Flask API for generating synthetic training data:
1. GPT generates sentences containing mispronounced words
2. Google TTS converts sentences to .wav files
3. Export as metadata.csv for XTTS training
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import shutil
import zipfile
import io
from datetime import datetime

# Import services
from llm_service import (
    generate_sentences,
    regenerate_single_sentence,
    get_current_config as get_llm_config,
    set_provider as set_llm_provider,
    get_ollama_models
)
from google_tts_service import (
    synthesize_speech,
    get_available_voices,
    generate_training_filename,
    DEFAULT_OUTPUT_DIR
)
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
    get_models, get_model, add_model, update_model, delete_model
)
import training_service
import inference_service

app = Flask(__name__)
CORS(app)

# Output directory
TRAINING_OUTPUT_DIR = "training_output"
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)


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
        
        # If Ollama provider specified with model, set it before generating
        if provider == 'ollama' and model:
            set_llm_provider(provider, model)
        elif provider:
            set_llm_provider(provider)
        
        sentences = generate_sentences(
            word=word,
            count=count,
            context=context,
            provider=provider
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


@app.route('/api/generate-audio', methods=['POST'])
def api_generate_audio():
    """Generate .wav files from sentences using Google TTS."""
    try:
        data = request.get_json()
        
        if not data or 'sentences' not in data:
            return jsonify({"error": "sentences array is required"}), 400
        
        sentences = data['sentences']
        if not isinstance(sentences, list) or len(sentences) == 0:
            return jsonify({"error": "sentences must be a non-empty array"}), 400
        
        voice = data.get('voice', 'tr-TR-Wavenet-D')
        speaking_rate = float(data.get('speakingRate', 1.0))
        pitch = float(data.get('pitch', 0.0))
        volume_gain_db = float(data.get('volumeGainDb', 0.0))
        
        results = []
        
        for item in sentences:
            text = item.get('text', '').strip()
            word = item.get('word', '').strip()
            
            if not text:
                results.append({
                    "success": False,
                    "error": "Empty text",
                    "text": text
                })
                continue
            
            # Check for existing audio (duplicate prevention)
            existing = check_existing_audio(text, word)
            if existing:
                print(f"⏭️ Skipping duplicate: {text[:50]}...")
                results.append({
                    "success": True,
                    "skipped": True,
                    "text": text,
                    "id": existing['id'],
                    "path": existing['wav_path'],
                    "play_url": f"/api/audio/{existing['id']}/play",
                    "message": "Already exists"
                })
                continue
            
            # Create word-based subfolder
            word_folder = os.path.join(TRAINING_OUTPUT_DIR, word.lower()) if word else TRAINING_OUTPUT_DIR
            os.makedirs(word_folder, exist_ok=True)
            output_path = generate_training_filename(text, word_folder)
            
            result = synthesize_speech(
                text=text,
                output_path=output_path,
                voice_name=voice,
                speaking_rate=speaking_rate,
                pitch=pitch,
                volume_gain_db=volume_gain_db
            )
            
            if result["success"]:
                item_id = add_training_item(
                    word=word,
                    sentence=text,
                    wav_path=result["path"],
                    voice=voice,
                    duration_seconds=result.get("duration_seconds"),
                    status="generated"
                )
                result["id"] = item_id
                result["play_url"] = f"/api/audio/{item_id}/play"
            
            results.append(result)
        
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
    """Get available Google TTS voices."""
    return jsonify({
        "success": True,
        "voices": get_available_voices()
    })


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
                    if len(wav_files) > 0:  # Only show folders with files
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
    """Download a specific folder as a ZIP file with sequential filenames and metadata."""
    try:
        # Get all generated items for this word to ensure we have sentences
        items = get_training_items(word=folder_name, status='generated')
        
        if not items:
             # Fallback to file system if DB has no entries (legacy compatibility)
            folder_path = os.path.join(TRAINING_OUTPUT_DIR, folder_name)
            if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
                 return jsonify({"error": "Folder not found"}), 404
            
            # If we rely on FS, we might miss sentences if not encoded in filename or external DB
            # But primarily we should rely on DB. 
            # If DB is empty but files exist, we just zip files as is? 
            # User wants 1,2,3.. and context|filename.
            # If not in DB, we can't get context/sentence easily.
            # Let's assume DB is source of truth for metadata.
            return jsonify({"error": "No database entries found for this folder"}), 404

        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            metadata_lines = []
            
            for index, item in enumerate(items, 1):
                wav_path = item.get('wav_path')
                original_sentence = item.get('sentence', '').strip()
                
                if wav_path and os.path.exists(wav_path):
                    # New sequential filename
                    new_filename = f"{index}.wav"
                    
                    # Add file to ZIP with new name
                    zip_file.write(wav_path, new_filename)
                    
                    # Add to metadata (FilenameWithoutExtension|Text|Text)
                    filename_no_ext = str(index)
                    metadata_lines.append(f"{filename_no_ext}|{original_sentence}|{original_sentence}")

            # Add metadata.csv to ZIP
            if metadata_lines:
                metadata_content = "\n".join(metadata_lines)
                zip_file.writestr("metadata.csv", metadata_content)
        
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
    """Download selected folders as a single ZIP file with sequential filenames and metadata."""
    try:
        data = request.get_json() or {}
        folder_names = data.get('folders', [])
        
        if not folder_names:
            return jsonify({"error": "No folders selected"}), 400
        
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
                    
                    if wav_path and os.path.exists(wav_path):
                        # Sequential filename
                        new_filename = f"{global_counter}.wav"
                        
                        # Add file to ZIP with new name
                        zip_file.write(wav_path, new_filename)
                        
                        # Add to metadata (FilenameWithoutExtension|Text|Text)
                        filename_no_ext = str(global_counter)
                        metadata_lines.append(f"{filename_no_ext}|{original_sentence}|{original_sentence}")
                        
                        global_counter += 1
            
            # Add metadata.csv to ZIP
            if metadata_lines:
                metadata_content = "\n".join(metadata_lines)
                zip_file.writestr("metadata.csv", metadata_content)
        
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
    """Export generated items as metadata.csv for training."""
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
            for item in items:
                wav_path = item['wav_path']
                sentence = item['sentence']
                
                # Get filename without extension
                filename = os.path.basename(wav_path)
                if filename.lower().endswith('.wav'):
                    filename_no_ext = filename[:-4]
                else:
                    filename_no_ext = filename
                
                # Format: FilenameNoExt|Text|Text
                f.write(f"{filename_no_ext}|{sentence}|{sentence}\n")
        
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
        allowed_keys = {'OPENAI_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'OLLAMA_BASE_URL', 'LLM_PROVIDER', 'OLLAMA_MODEL'}
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
