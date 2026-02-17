"""
Model Registry - SQLite database for trained model tracking

Tracks trained models with metadata (name, description, tags, paths, training params).
"""

import os
import sqlite3
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager

# Database file path (same DB as training data)
DATABASE_PATH = "training_data.db"

_db_lock = threading.Lock()


def get_db_path():
    """Get the database file path."""
    return DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_model_registry():
    """Initialize the models table."""
    with _db_lock:
        with get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    model_path TEXT,
                    config_path TEXT,
                    base_model TEXT DEFAULT 'xtts_v2',
                    training_params TEXT DEFAULT '{}',
                    dataset_csv TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    completed_at DATETIME,
                    test_audio_path TEXT,
                    training_log TEXT DEFAULT '',
                    error_message TEXT
                )
            """)


def add_model(
    name: str,
    description: str = "",
    tags: List[str] = None,
    base_model: str = "xtts_v2",
    training_params: dict = None,
    dataset_csv: str = None,
    status: str = "pending"
) -> int:
    """Add a new model entry. Returns model ID."""
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO models (name, description, tags, base_model, training_params, dataset_csv, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    name,
                    description,
                    json.dumps(tags or []),
                    base_model,
                    json.dumps(training_params or {}),
                    dataset_csv,
                    status
                )
            )
            return cursor.lastrowid


# Columns to select by default (excludes training_log to avoid reading huge text)
_MODEL_COLS = (
    "id, name, description, tags, model_path, config_path, base_model, "
    "training_params, dataset_csv, status, created_at, completed_at, "
    "test_audio_path, error_message"
)


def get_models(status: Optional[str] = None) -> List[Dict]:
    """Get all models, optionally filtered by status."""
    with get_connection() as conn:
        if status:
            rows = conn.execute(
                f"SELECT {_MODEL_COLS} FROM models WHERE status = ? ORDER BY created_at DESC", (status,)
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_MODEL_COLS} FROM models ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_model(model_id: int) -> Optional[Dict]:
    """Get a single model by ID (excludes training_log for performance)."""
    with get_connection() as conn:
        row = conn.execute(f"SELECT {_MODEL_COLS} FROM models WHERE id = ?", (model_id,)).fetchone()
        return _row_to_dict(row) if row else None


def update_model(model_id: int, **kwargs) -> bool:
    """Update model fields. Returns True if updated."""
    allowed = {
        'name', 'description', 'tags', 'model_path', 'config_path',
        'base_model', 'training_params', 'dataset_csv', 'status',
        'completed_at', 'test_audio_path', 'training_log', 'error_message'
    }

    updates = {}
    for key, value in kwargs.items():
        if key not in allowed:
            continue
        if key in ('tags', 'training_params'):
            value = json.dumps(value) if isinstance(value, (list, dict)) else value
        updates[key] = value

    if not updates:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    values = list(updates.values()) + [model_id]

    with _db_lock:
        with get_connection() as conn:
            result = conn.execute(
                f"UPDATE models SET {set_clause} WHERE id = ?", values
            )
            return result.rowcount > 0


def delete_model(model_id: int) -> bool:
    """Delete a model and optionally its files. Returns True if deleted."""
    model = get_model(model_id)
    if not model:
        return False

    # Delete model files if they exist
    for path_key in ('model_path', 'config_path', 'test_audio_path'):
        path = model.get(path_key)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # Try to remove the model directory
    if model.get('model_path'):
        model_dir = os.path.dirname(model['model_path'])
        if os.path.isdir(model_dir):
            try:
                import shutil
                shutil.rmtree(model_dir, ignore_errors=True)
            except OSError:
                pass

    with _db_lock:
        with get_connection() as conn:
            result = conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
            return result.rowcount > 0


# Maximum training log size (500 KB) — prevents DB bloat from error cascades
MAX_LOG_SIZE = 500_000


def append_training_log(model_id: int, log_text: str):
    """Append text to a model's training log, capping at MAX_LOG_SIZE."""
    # Use a timeout to avoid blocking the training thread indefinitely
    if not _db_lock.acquire(timeout=1):
        return  # Skip this log line rather than freezing training
    try:
        with get_connection() as conn:
            # Get current log length
            row = conn.execute(
                "SELECT LENGTH(COALESCE(training_log, '')) FROM models WHERE id = ?",
                (model_id,)
            ).fetchone()
            current_len = row[0] if row else 0
            
            if current_len + len(log_text) > MAX_LOG_SIZE:
                # Trim old content: keep last 80% of max to make room
                keep_size = int(MAX_LOG_SIZE * 0.8)
                conn.execute(
                    "UPDATE models SET training_log = SUBSTR(training_log, -?) WHERE id = ?",
                    (keep_size, model_id)
                )
            
            conn.execute(
                "UPDATE models SET training_log = COALESCE(training_log, '') || ? WHERE id = ?",
                (log_text, model_id)
            )
    finally:
        _db_lock.release()


def get_training_log(model_id: int, max_chars: int = 100_000) -> str:
    """Get the last max_chars of a model's training log efficiently."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT SUBSTR(training_log, -?) FROM models WHERE id = ?",
            (max_chars, model_id)
        ).fetchone()
        return row[0] if row and row[0] else ""


def truncate_training_log(model_id: int, keep_chars: int = 10_000):
    """Truncate a model's training log to the last keep_chars characters."""
    with _db_lock:
        with get_connection() as conn:
            conn.execute(
                "UPDATE models SET training_log = SUBSTR(training_log, -?) WHERE id = ?",
                (keep_chars, model_id)
            )


def _row_to_dict(row) -> Dict:
    """Convert a sqlite3.Row to a dictionary with JSON parsing."""
    d = dict(row)
    # Parse JSON fields
    for key in ('tags', 'training_params'):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                d[key] = [] if key == 'tags' else {}
    return d


# Initialize on import
init_model_registry()
