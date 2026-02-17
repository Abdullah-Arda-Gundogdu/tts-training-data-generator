"""
XTTS Trainer - Fine-tuning XTTS v2.0 model

Copied and refactored from C:/TTS/.../xtts_content/xtts.py
All training parameters are now function arguments instead of globals.
"""

import os
import sys
import io
import torch

# PyTorch 2.6 changed torch.load default to weights_only=True.
# Coqui TTS checkpoints contain config objects that need pickle, so we
# patch torch.load to keep the old behaviour (safe — files come from HuggingFace).
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load


def run_training(
    dataset_path: str,
    meta_file_train: str = "metadata.csv",
    language: str = "tr",
    num_samples: int = 78,
    desired_epochs: int = 40,
    batch_size: int = 1,
    grad_accum_steps: int = 10,
    learning_rate: float = 5e-06,
    save_step: int = 30,
    save_n_checkpoints: int = 5,
    speaker_reference: str = None,
    output_path: str = None,
    run_name: str = "GPT_XTTS_v2.0_FT",
    project_name: str = "XTTS_trainer",
    log_callback=None
):
    """
    Run XTTS fine-tuning with the given parameters.
    
    Args:
        dataset_path: Path to the dataset directory containing audio + metadata.csv
        meta_file_train: Name of the metadata CSV file
        language: Language code (e.g., 'tr', 'en')
        num_samples: Number of samples in the dataset
        desired_epochs: Number of training epochs
        batch_size: Batch size per step
        grad_accum_steps: Gradient accumulation steps
        learning_rate: Learning rate
        save_step: Save checkpoint every N steps
        save_n_checkpoints: Number of checkpoints to keep
        speaker_reference: Path to speaker reference WAV
        output_path: Path for training outputs (checkpoints, logs)
        run_name: Name for this training run
        project_name: Project name for logging
        log_callback: Function to call with log messages (str -> None)
    
    Returns:
        dict with 'success', 'output_path', 'best_model_path', 'message'
    """
    
    def log(msg):
        """Log message to callback and stdout."""
        print(msg)
        if log_callback:
            log_callback(msg + "\n")
    
    try:
        log("📦 Importing training dependencies...")
        
        from trainer import Trainer, TrainerArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        from TTS.tts.datasets import load_tts_samples
        from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
        from TTS.tts.models.xtts import XttsAudioConfig
        from TTS.utils.manage import ModelManager
        
        log("✅ Dependencies imported successfully")
        
        # Resolve paths
        if output_path is None:
            output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_runs")
        os.makedirs(output_path, exist_ok=True)
        
        # Calculate training steps
        effective_batch_size = batch_size * grad_accum_steps
        steps_per_epoch = max(1, num_samples // effective_batch_size)
        max_train_steps = desired_epochs * steps_per_epoch
        
        log(f"Training Configuration:")
        log(f"  Dataset path: {dataset_path}")
        log(f"  Dataset size: {num_samples} samples")
        log(f"  Desired epochs: {desired_epochs}")
        log(f"  Batch size: {batch_size}")
        log(f"  Gradient accumulation steps: {grad_accum_steps}")
        log(f"  Effective batch size: {effective_batch_size}")
        log(f"  Steps per epoch: {steps_per_epoch}")
        log(f"  Total training steps: {max_train_steps}")
        log(f"  Learning rate: {learning_rate}")
        log(f"  Save every: {save_step} steps")
        log("-" * 50)
        
        # Dataset config — metadata.csv uses LJSpeech pipe-separated format for all languages
        config_dataset = BaseDatasetConfig(
            formatter="ljspeech",
            dataset_name=f"{language}_fine_tune",
            path=dataset_path,
            meta_file_train=meta_file_train,
            language=language,
        )
        
        DATASETS_CONFIG_LIST = [config_dataset]
        
        # Shared base model files directory (downloaded once, reused across all runs)
        CHECKPOINTS_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_base_files")
        os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)
        
        # DVAE files
        DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
        MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"
        
        DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
        MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))
        
        # Download DVAE files if needed
        if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
            log(" > Downloading DVAE files!")
            ModelManager._download_model_files(
                [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
            )
        
        # XTTS v2.0 checkpoint
        TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
        XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
        
        # Look for local model files first
        CHECKPOINTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_v2")
        if not os.path.isdir(CHECKPOINTS_PATH):
            CHECKPOINTS_PATH = CHECKPOINTS_OUT_PATH
        
        TOKENIZER_FILE = os.path.join(CHECKPOINTS_PATH, "vocab.json")
        XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_PATH, "model.pth")
        
        # Download XTTS v2.0 files if needed
        if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
            log(" > Downloading XTTS v2.0 files!")
            ModelManager._download_model_files(
                [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
            )
            TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
            XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
        
        # Speaker reference
        SPEAKER_REFERENCE = []
        if speaker_reference and os.path.isfile(speaker_reference):
            SPEAKER_REFERENCE = [speaker_reference]
        
        LANGUAGE = config_dataset.language
        
        # Model args
        log("🔧 Configuring model...")
        model_args = GPTArgs(
            max_conditioning_length=132300,
            min_conditioning_length=66150,
            debug_loading_failures=False,
            max_wav_length=255995,
            max_text_length=200,
            mel_norm_file=MEL_NORM_FILE,
            dvae_checkpoint=DVAE_CHECKPOINT,
            xtts_checkpoint=XTTS_CHECKPOINT,
            tokenizer_file=TOKENIZER_FILE,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        
        audio_config = XttsAudioConfig()
        # Newer TTS versions removed dvae_sample_rate from the dataclass,
        # but GPTTrainer still references it — add it manually.
        if not hasattr(audio_config, 'dvae_sample_rate'):
            audio_config.dvae_sample_rate = 22050
        
        # Build test sentences
        test_sentences = []
        if SPEAKER_REFERENCE:
            test_sentences = [
                {
                    "text": "Modern üretim tesislerinde günlük olarak yüzlerce adet farklı ürün üretilmektedir.",
                    "speaker_wav": SPEAKER_REFERENCE,
                    "language": LANGUAGE,
                },
                {
                    "text": "Bu modern tasarımda üç adet kanepe ve sekiz adet sürahi bulunuyor.",
                    "speaker_wav": SPEAKER_REFERENCE,
                    "language": LANGUAGE,
                },
            ]
        
        # LR scheduler milestones
        total_steps = max_train_steps
        milestones = [
            int(total_steps * 0.33),
            int(total_steps * 0.66),
            int(total_steps * 0.83)
        ]
        
        config = GPTTrainerConfig(
            output_path=output_path,
            model_args=model_args,
            run_name=run_name,
            project_name=project_name,
            run_description=f"XTTS Fine-tuning - {desired_epochs} epochs",
            dashboard_logger="tensorboard",
            logger_uri=None,
            audio=audio_config,
            batch_size=batch_size,
            batch_group_size=min(48, num_samples),
            eval_batch_size=batch_size,
            num_loader_workers=0,  # Must be 0 on Windows to avoid multiprocessing issues
            eval_split_max_size=min(256, max(1, num_samples // 10)),
            eval_split_size=0.1,
            print_step=1,  # Log every step for visibility
            plot_step=20,
            log_model_step=50,
            save_step=save_step,
            save_n_checkpoints=save_n_checkpoints,
            save_checkpoints=True,
            print_eval=True,
            epochs=desired_epochs,
            optimizer="AdamW",
            optimizer_wd_only_on_weights=True,
            optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
            lr=learning_rate,
            lr_scheduler="MultiStepLR",
            lr_scheduler_params={"milestones": milestones, "gamma": 0.5, "last_epoch": -1},
            test_sentences=test_sentences,
        )
        
        # Init model
        log("🔧 Initializing model from config...")
        model = GPTTrainer.init_from_config(config)
        
        # Load training samples
        log("📂 Loading training samples...")
        train_samples, eval_samples = load_tts_samples(
            DATASETS_CONFIG_LIST,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        
        log(f"  Training samples: {len(train_samples)}")
        log(f"  Evaluation samples: {len(eval_samples)}")
        log(f"  Total samples: {len(train_samples) + len(eval_samples)}")
        log("-" * 50)
        
        # Monkey-patch remove_experiment_folder to handle Windows file locks
        # When training fails, the Trainer tries to delete the experiment folder
        # but its own FileHandler still holds trainer_0_log.txt open → WinError 32
        import trainer.generic_utils as _trainer_utils
        import trainer.trainer as _trainer_mod
        
        def _safe_remove_experiment_folder(experiment_path):
            """Silently handle WinError 32 during cleanup."""
            try:
                import shutil
                shutil.rmtree(experiment_path, ignore_errors=True)
            except Exception:
                pass
        
        _trainer_utils.remove_experiment_folder = _safe_remove_experiment_folder
        _trainer_mod.remove_experiment_folder = _safe_remove_experiment_folder
        
        log("✅ Windows workarounds applied")
        
        # Init trainer
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=False,  # Don't eval before training starts
                grad_accum_steps=grad_accum_steps,
            ),
            config,
            output_path=output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        
        log(f"🚀 Starting training...")
        log(f"Training will run for {desired_epochs} epochs ({max_train_steps} steps)")
        log(f"Checkpoints will be saved every {save_step} steps")
        log("-" * 50)
        
        # Capture Trainer output via logging handlers instead of replacing sys.stdout.
        # Replacing sys.stdout with a custom object breaks tqdm progress bars and
        # can deadlock because every print triggers a DB write + lock acquisition.
        import logging
        
        class _LogHandler(logging.Handler):
            def __init__(self, cb):
                super().__init__()
                self._cb = cb
            def emit(self, record):
                try:
                    msg = self.format(record)
                    if msg.strip():
                        self._cb(msg + '\n')
                except Exception:
                    pass  # Never let logging errors crash training
        
        _handler = _LogHandler(log_callback)
        
        # Attach to both trainer and TTS loggers
        for logger_name in ("trainer", "TTS"):
            _logger = logging.getLogger(logger_name)
            _logger.addHandler(_handler)
            _logger.setLevel(logging.INFO)
        
        # Also capture raw print output via a minimal stdout wrapper that
        # only forwards to the callback WITHOUT blocking the original stdout.
        import sys
        _orig_stdout = sys.stdout
        _orig_stderr = sys.stderr
        
        class _TeeWriter:
            """Wraps the original stream, forwarding a copy to the callback.
            
            Unlike the old LogRedirector (which subclassed io.TextIOBase),
            this delegates ALL attribute access to the original stream so that
            tqdm, CUDA, and other C-level code see a real file descriptor.
            """
            def __init__(self, original, callback):
                object.__setattr__(self, '_original', original)
                object.__setattr__(self, '_callback', callback)
            
            def write(self, text):
                self._original.write(text)
                if text and text.strip():
                    try:
                        self._callback(text if text.endswith('\n') else text + '\n')
                    except Exception:
                        pass
                return len(text) if text else 0
            
            def __getattr__(self, name):
                return getattr(self._original, name)
        
        sys.stdout = _TeeWriter(_orig_stdout, log_callback)
        sys.stderr = _TeeWriter(_orig_stderr, log_callback)
        
        try:
            trainer.fit()
        finally:
            sys.stdout = _orig_stdout
            sys.stderr = _orig_stderr
            for logger_name in ("trainer", "TTS"):
                logging.getLogger(logger_name).removeHandler(_handler)
        
        log("✅ Training completed successfully!")
        
        # Find the best model checkpoint
        best_model_path = _find_best_model(output_path, run_name)
        
        return {
            "success": True,
            "output_path": output_path,
            "best_model_path": best_model_path,
            "message": "Training completed successfully"
        }
        
    except Exception as e:
        error_msg = f"❌ Training failed: {str(e)}"
        log(error_msg)
        import traceback
        log(traceback.format_exc())
        return {
            "success": False,
            "output_path": output_path if 'output_path' in dir() else None,
            "best_model_path": None,
            "message": str(e)
        }


def _find_best_model(output_path: str, run_name: str) -> str:
    """Find the best/latest model checkpoint in the output directory."""
    import glob
    
    # Look for the best model or latest checkpoint
    patterns = [
        os.path.join(output_path, run_name, "**", "best_model.pth"),
        os.path.join(output_path, run_name, "**", "model.pth"),
        os.path.join(output_path, run_name, "**", "checkpoint_*.pth"),
        os.path.join(output_path, "**", "best_model.pth"),
        os.path.join(output_path, "**", "model.pth"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            # Return the most recently modified file
            return max(files, key=os.path.getmtime)
    
    return None
