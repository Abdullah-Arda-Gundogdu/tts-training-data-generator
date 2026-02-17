"""
Training Service - Orchestrates XTTS training jobs

Runs training in background threads, captures logs, and manages model lifecycle.
"""

import glob
import os
import shutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional

from model_registry import (
    add_model, get_model, update_model, append_training_log, get_training_log
)


# Active training jobs: {model_id: {"thread": Thread, "cancel": Event, "status": str}}
_active_jobs: Dict[int, dict] = {}
_jobs_lock = threading.Lock()

# Directory constants
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
TRAINING_RUNS_DIR = os.path.join(BACKEND_DIR, "training_runs")
TRAINING_OUTPUT_DIR = os.path.join(BACKEND_DIR, "training_output")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TRAINING_RUNS_DIR, exist_ok=True)


def _cleanup_previous_jobs():
    """Cancel and clean up any previous training jobs."""
    with _jobs_lock:
        for mid, job in list(_active_jobs.items()):
            job["cancel"].set()
        
        for mid, job in list(_active_jobs.items()):
            thread = job["thread"]
            if thread.is_alive():
                thread.join(timeout=5)
            del _active_jobs[mid]


def _build_dataset_from_folders(
    selected_folders: List[str],
    dest_dir: str,
    log_callback
) -> tuple:
    """
    Build a training dataset directory from selected folders.
    
    For each selected folder:
      - Query the database for items in that folder (word == folder name)
      - Copy WAV files to dest_dir/wavs/
      - Generate metadata.csv in LJSpeech format: filename|text|text
    
    Returns:
        (dataset_path, meta_filename, num_samples)
    """
    from training_database import get_items_for_export
    
    wavs_dir = os.path.join(dest_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    
    metadata_lines = []
    total_copied = 0
    skipped = 0
    
    for folder_name in selected_folders:
        log_callback(f"📁 Processing folder: {folder_name}\n")
        
        # Get items from DB for this folder
        items = get_items_for_export(word=folder_name)
        
        if not items:
            # Fallback: scan the folder directly for WAV files
            folder_path = os.path.join(TRAINING_OUTPUT_DIR, folder_name)
            if os.path.isdir(folder_path):
                wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                log_callback(f"  ⚠️ No DB items found, but found {len(wav_files)} WAV files in folder\n")
                for wav_file in wav_files:
                    src = os.path.join(folder_path, wav_file)
                    dst = os.path.join(wavs_dir, wav_file)
                    if not os.path.isfile(dst):
                        shutil.copy2(src, dst)
                    # Use filename as text placeholder since we have no sentence
                    basename = os.path.splitext(wav_file)[0]
                    metadata_lines.append(f"wavs/{wav_file}|{basename}|{basename}")
                    total_copied += 1
            else:
                log_callback(f"  ⚠️ Folder not found: {folder_path}\n")
            continue
        
        log_callback(f"  Found {len(items)} items in database\n")
        
        for item in items:
            wav_path = item.get("wav_path", "")
            sentence = item.get("sentence", "")
            
            if not wav_path or not sentence:
                skipped += 1
                continue
            
            # Resolve WAV path
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(BACKEND_DIR, wav_path)
            
            if not os.path.isfile(wav_path):
                log_callback(f"  ⚠️ WAV not found: {wav_path}\n")
                skipped += 1
                continue
            
            # Copy WAV to dataset wavs/ directory
            wav_filename = os.path.basename(wav_path)
            dst = os.path.join(wavs_dir, wav_filename)
            if not os.path.isfile(dst):
                shutil.copy2(wav_path, dst)
            
            # LJSpeech format: stem|text|text
            # The ljspeech formatter adds "wavs/" prefix and ".wav" suffix automatically,
            # so we only store the filename stem (without extension or directory).
            wav_stem = os.path.splitext(wav_filename)[0]
            metadata_lines.append(f"{wav_stem}|{sentence}|{sentence}")
            total_copied += 1
    
    # Write metadata.csv
    meta_filename = "metadata.csv"
    meta_path = os.path.join(dest_dir, meta_filename)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")
    
    log_callback(f"📊 Dataset built: {total_copied} samples, {skipped} skipped\n")
    log_callback(f"📄 Metadata written to: {meta_path}\n")
    
    return dest_dir, meta_filename, total_copied


def start_training(
    name: str,
    description: str = "",
    tags: list = None,
    selected_folders: list = None,
    dataset_path: str = "",
    meta_file_train: str = "metadata.csv",
    training_params: dict = None
) -> dict:
    """
    Start a new training job.
    
    Returns dict with 'model_id' and 'message'.
    """
    params = training_params or {}
    
    # Clean up any previous training jobs to release file handles
    _cleanup_previous_jobs()
    
    # Create model entry in DB
    model_id = add_model(
        name=name,
        description=description,
        tags=tags or [],
        training_params=params,
        dataset_csv=", ".join(selected_folders) if selected_folders else dataset_path,
        status="training"
    )
    
    # Create cancel event
    cancel_event = threading.Event()
    
    # Start training thread
    thread = threading.Thread(
        target=_training_worker,
        args=(model_id, selected_folders or [], dataset_path, meta_file_train, params, cancel_event),
        daemon=True
    )
    
    with _jobs_lock:
        _active_jobs[model_id] = {
            "thread": thread,
            "cancel": cancel_event,
            "status": "starting",
            "started_at": datetime.now().isoformat()
        }
    
    thread.start()
    
    return {
        "model_id": model_id,
        "message": f"Training started for model '{name}' (ID: {model_id})"
    }


def get_training_status(model_id: int) -> dict:
    """Get the current status of a training job."""
    model = get_model(model_id)
    if not model:
        return {"error": "Model not found"}
    
    job_info = {}
    with _jobs_lock:
        if model_id in _active_jobs:
            job_info = {
                "is_running": _active_jobs[model_id]["thread"].is_alive(),
                "job_status": _active_jobs[model_id]["status"],
                "started_at": _active_jobs[model_id].get("started_at")
            }
    
    return {
        "model_id": model_id,
        "name": model["name"],
        "status": model["status"],
        "training_log": get_training_log(model_id),
        "training_params": model.get("training_params", {}),
        "error_message": model.get("error_message"),
        "created_at": model.get("created_at"),
        "completed_at": model.get("completed_at"),
        **job_info
    }


def cancel_training(model_id: int) -> dict:
    """Cancel a running training job."""
    with _jobs_lock:
        if model_id not in _active_jobs:
            return {"error": "No active training job for this model"}
        
        job = _active_jobs[model_id]
        job["cancel"].set()
        job["status"] = "cancelling"
        
        # Try to interrupt the training thread
        thread = job["thread"]
        if thread.is_alive():
            import ctypes
            tid = thread.ident
            if tid is not None:
                # Try multiple times with different exceptions
                for exc_type in [KeyboardInterrupt, SystemExit, KeyboardInterrupt]:
                    try:
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            ctypes.c_ulong(tid),
                            ctypes.py_object(exc_type)
                        )
                    except Exception:
                        pass
                    time.sleep(0.1)
        
        # Always clean up from active jobs so user can start new training
        del _active_jobs[model_id]
    
    update_model(model_id, status="cancelled")
    append_training_log(model_id, "\n⚠️ Training cancelled by user\n")
    
    return {"message": "Training cancelled"}


def get_all_active_jobs() -> list:
    """Get list of all active training jobs."""
    with _jobs_lock:
        return [
            {
                "model_id": mid,
                "status": info["status"],
                "is_running": info["thread"].is_alive(),
                "started_at": info.get("started_at")
            }
            for mid, info in _active_jobs.items()
        ]


def _training_worker(
    model_id: int,
    selected_folders: List[str],
    dataset_path: str,
    meta_file_train: str,
    params: dict,
    cancel_event: threading.Event
):
    """Background worker that runs the actual training."""
    
    def log_callback(msg: str):
        """Capture training output to database (non-blocking)."""
        try:
            append_training_log(model_id, msg)
        except Exception:
            pass  # Never let logging stall training
        # Update job status periodically
        try:
            with _jobs_lock:
                if model_id in _active_jobs:
                    _active_jobs[model_id]["status"] = "training"
        except Exception:
            pass
    
    try:
        log_callback("🚀 Starting training job...\n")
        
        # Import here to avoid loading heavy deps at module level
        from xtts_trainer import run_training
        
        # Prepare output path for this specific model
        model = get_model(model_id)
        safe_name = "".join(
            c for c in (model["name"] if model else f"model_{model_id}")
            if c.isalnum() or c in (' ', '_', '-')
        ).strip().replace(' ', '_')
        
        run_output_path = os.path.join(TRAINING_RUNS_DIR, f"{safe_name}_{model_id}_{datetime.now().strftime('%H%M%S')}")
        run_name = f"XTTS_{safe_name}"
        
        # -------------------------------------------------------
        # Build dataset from selected folders OR use manual path
        # -------------------------------------------------------
        if selected_folders:
            log_callback(f"📂 Building dataset from {len(selected_folders)} selected folders...\n")
            dataset_dir = os.path.join(run_output_path, "dataset")
            os.makedirs(dataset_dir, exist_ok=True)
            
            dataset_path, meta_file_train, num_samples = _build_dataset_from_folders(
                selected_folders=selected_folders,
                dest_dir=dataset_dir,
                log_callback=log_callback
            )
            
            if num_samples == 0:
                raise ValueError("No audio samples found in selected folders. Check that folders have generated audio.")
            
            # Override num_samples with actual count
            params["num_samples"] = num_samples
            log_callback(f"✅ Dataset ready: {num_samples} samples from folders: {', '.join(selected_folders)}\n")
        else:
            # Resolve manual dataset path to absolute
            if dataset_path and not os.path.isabs(dataset_path):
                dataset_path = os.path.join(BACKEND_DIR, dataset_path)
        
        # Run training
        result = run_training(
            dataset_path=dataset_path,
            meta_file_train=meta_file_train,
            language=params.get("language", "tr"),
            num_samples=params.get("num_samples", 78),
            desired_epochs=params.get("epochs", 40),
            batch_size=params.get("batch_size", 1),
            grad_accum_steps=params.get("grad_accum_steps", 10),
            learning_rate=params.get("learning_rate", 5e-06),
            save_step=params.get("save_step", 30),
            save_n_checkpoints=params.get("save_n_checkpoints", 5),
            speaker_reference=params.get("speaker_reference"),
            output_path=run_output_path,
            run_name=run_name,
            log_callback=log_callback
        )
        
        if cancel_event.is_set():
            log_callback("⚠️ Training was cancelled\n")
            update_model(model_id, status="cancelled")
            return
        
        if result["success"]:
            # Copy best model to models directory
            model_dir = os.path.join(MODELS_DIR, f"{safe_name}_{model_id}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = None
            config_path = None
            
            if result.get("best_model_path") and os.path.isfile(result["best_model_path"]):
                model_path = os.path.join(model_dir, "model.pth")
                shutil.copy2(result["best_model_path"], model_path)
                log_callback(f"✅ Model saved to: {model_path}\n")
            
            # Copy config.json — Trainer saves it inside a subdirectory,
            # so search recursively

            config_candidates = glob.glob(
                os.path.join(run_output_path, "**", "config.json"), recursive=True
            )
            if config_candidates:
                # Use the most recently modified one
                src_config = max(config_candidates, key=os.path.getmtime)
                config_path = os.path.join(model_dir, "config.json")
                shutil.copy2(src_config, config_path)
                log_callback(f"✅ Config saved to: {config_path}\n")
            
            # Also copy vocab.json so inference can find it next to model.pth
            backend_dir_local = os.path.dirname(os.path.abspath(__file__))
            for vocab_src_dir in [
                os.path.join(backend_dir_local, "xtts_v2"),
                os.path.join(backend_dir_local, "xtts_base_files"),
            ]:
                vocab_src = os.path.join(vocab_src_dir, "vocab.json")
                if os.path.isfile(vocab_src):
                    shutil.copy2(vocab_src, os.path.join(model_dir, "vocab.json"))
                    log_callback(f"✅ Vocab copied to model dir\n")
                    break
            
            update_model(
                model_id,
                status="completed",
                completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_path=model_path,
                config_path=config_path
            )
            log_callback("🎉 Training completed and model saved!\n")
            
            # Auto-cleanup: remove training run directory to free disk space
            # Only if model files were successfully copied to models/ dir
            if (os.path.isfile(model_path) and 
                os.path.isfile(os.path.join(model_dir, "vocab.json"))):
                try:
                    shutil.rmtree(run_output_path, ignore_errors=True)
                    log_callback(f"🧹 Cleaned up training run directory: {run_output_path}\n")
                except Exception as cleanup_err:
                    log_callback(f"⚠️ Could not clean up training run dir: {cleanup_err}\n")
        else:
            update_model(
                model_id,
                status="failed",
                error_message=result.get("message", "Unknown error"),
                completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            log_callback(f"❌ Training failed: {result.get('message')}\n")
    
    except Exception as e:
        import traceback
        error_msg = f"Training error: {str(e)}"
        log_callback(f"❌ {error_msg}\n")
        log_callback(traceback.format_exc() + "\n")
        update_model(
            model_id,
            status="failed",
            error_message=str(e),
            completed_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    finally:
        with _jobs_lock:
            if model_id in _active_jobs:
                _active_jobs[model_id]["status"] = "finished"
