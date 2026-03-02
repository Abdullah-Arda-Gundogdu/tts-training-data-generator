"""
Training Database - SQLite database for training items

This module handles persistence for generated training data items.
"""

import os
import sqlite3
import json
import threading
from datetime import datetime
from typing import List, Dict, Optional
from contextlib import contextmanager

# Database file path
DATABASE_PATH = "training_data.db"

# Thread-safe lock
_db_lock = threading.Lock()


def get_db_path() -> str:
    """Get the database file path."""
    return DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_training_db():
    """Initialize the training database with required tables."""
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Training items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    sentence TEXT NOT NULL,
                    spoken_text TEXT,
                    wav_path TEXT,
                    status TEXT DEFAULT 'pending',
                    voice TEXT DEFAULT 'tr-TR-Wavenet-D',
                    duration_seconds REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    exported_at TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # Migration: add spoken_text column if not exists (for existing DBs)
            try:
                cursor.execute("ALTER TABLE training_items ADD COLUMN spoken_text TEXT")
                print("📝 Added spoken_text column to training_items")
            except sqlite3.OperationalError:
                pass  # Column already exists
            
            # Generation batches table for tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_batches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    sentence_count INTEGER DEFAULT 0,
                    audio_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Error reports table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT NOT NULL,
                    explanation TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print("✅ Training database initialized")


def add_training_item(
    word: str,
    sentence: str,
    wav_path: Optional[str] = None,
    voice: str = "tr-TR-Wavenet-D",
    duration_seconds: Optional[float] = None,
    status: str = "pending",
    spoken_text: Optional[str] = None
) -> int:
    """
    Add a new training item to the database.
    
    Args:
        word: The target word for pronunciation
        sentence: The full sentence text (formal/written form)
        wav_path: Path to the generated .wav file (if exists)
        voice: Google TTS voice used
        duration_seconds: Audio duration
        status: Item status (pending, generated, exported)
        spoken_text: Colloquial spoken form of the sentence (for alignment)
    
    Returns:
        The ID of the created item
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_items 
                (word, sentence, spoken_text, wav_path, voice, duration_seconds, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (word, sentence, spoken_text, wav_path, voice, duration_seconds, status))
            conn.commit()
            return cursor.lastrowid


def update_training_item(
    item_id: int,
    **kwargs
) -> bool:
    """
    Update a training item.
    
    Args:
        item_id: The item ID to update
        **kwargs: Fields to update (wav_path, status, sentence, etc.)
    
    Returns:
        True if updated, False if not found
    """
    if not kwargs:
        return False
    
    allowed_fields = {'word', 'sentence', 'spoken_text', 'wav_path', 'status', 'voice', 
                      'duration_seconds', 'exported_at', 'metadata'}
    
    updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
    if not updates:
        return False
    
    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    values = list(updates.values()) + [item_id]
    
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE training_items 
                SET {set_clause}
                WHERE id = ?
            """, values)
            conn.commit()
            return cursor.rowcount > 0


def get_training_item(item_id: int) -> Optional[Dict]:
    """Get a single training item by ID."""
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_items WHERE id = ?
            """, (item_id,))
            row = cursor.fetchone()
            return dict(row) if row else None


def get_training_items(
    word: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict]:
    """
    Get training items with optional filters.
    
    Args:
        word: Filter by target word
        status: Filter by status
        limit: Maximum items to return
        offset: Pagination offset
    
    Returns:
        List of training items
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM training_items WHERE 1=1"
            params = []
            
            if word:
                query += " AND word = ?"
                params.append(word)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def delete_training_item(item_id: int) -> bool:
    """
    Delete a training item.
    
    Args:
        item_id: The item ID to delete
    
    Returns:
        True if deleted, False if not found
    """
    # Get item first to check for wav file
    item = get_training_item(item_id)
    
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM training_items WHERE id = ?", (item_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
    
    # Delete associated wav file if exists
    if deleted and item and item.get('wav_path'):
        try:
            if os.path.exists(item['wav_path']):
                os.remove(item['wav_path'])
                print(f"🗑️ Deleted audio file: {item['wav_path']}")
        except Exception as e:
            print(f"⚠️ Could not delete audio file: {e}")
    
    return deleted


def get_training_stats() -> Dict:
    """
    Get statistics about training items.
    
    Returns:
        Dictionary with counts by status and other stats
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            # Count by status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM training_items 
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Count unique words
            cursor.execute("""
                SELECT COUNT(DISTINCT word) as unique_words 
                FROM training_items
            """)
            unique_words = cursor.fetchone()['unique_words']
            
            # Total items
            cursor.execute("SELECT COUNT(*) as total FROM training_items")
            total = cursor.fetchone()['total']
            
            return {
                "total": total,
                "pending": status_counts.get("pending", 0),
                "generated": status_counts.get("generated", 0),
                "exported": status_counts.get("exported", 0),
                "unique_words": unique_words
            }


def get_items_for_export(word: Optional[str] = None) -> List[Dict]:
    """
    Get all generated items ready for export.
    
    Args:
        word: Optional filter by word
    
    Returns:
        List of items with generated audio
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM training_items 
                WHERE status = 'generated' AND wav_path IS NOT NULL
            """
            params = []
            
            if word:
                query += " AND word = ?"
                params.append(word)
            
            query += " ORDER BY word, created_at"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def mark_items_exported(item_ids: List[int]) -> int:
    """
    Mark items as exported.
    
    Args:
        item_ids: List of item IDs to mark
    
    Returns:
        Number of items updated
    """
    if not item_ids:
        return 0
    
    placeholders = ",".join(["?" for _ in item_ids])
    now = datetime.now().isoformat()
    
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE training_items 
                SET status = 'exported', exported_at = ?
                WHERE id IN ({placeholders})
            """, [now] + item_ids)
            conn.commit()
            return cursor.rowcount


def bulk_delete_items(item_ids: List[int]) -> int:
    """
    Delete multiple training items in a single transaction.
    
    Args:
        item_ids: List of item IDs to delete
    
    Returns:
        Number of items deleted
    """
    if not item_ids:
        return 0
    
    # Get items first to find wav files to delete
    items_to_delete = []
    for item_id in item_ids:
        item = get_training_item(item_id)
        if item:
            items_to_delete.append(item)
    
    placeholders = ",".join(["?" for _ in item_ids])
    
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM training_items WHERE id IN ({placeholders})", item_ids)
            conn.commit()
            deleted_count = cursor.rowcount
    
    # Delete associated wav files
    for item in items_to_delete:
        if item.get('wav_path'):
            try:
                if os.path.exists(item['wav_path']):
                    os.remove(item['wav_path'])
            except Exception as e:
                print(f"⚠️ Could not delete audio file: {e}")
    
    print(f"🗑️ Bulk deleted {deleted_count} items")
    return deleted_count


def check_existing_audio(sentence: str, word: str) -> Optional[Dict]:
    """
    Check if audio already exists for a sentence+word combination.
    
    Args:
        sentence: The sentence text
        word: The target word
    
    Returns:
        Existing item dict if found, None otherwise
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM training_items 
                WHERE sentence = ? AND word = ? AND status = 'generated' AND wav_path IS NOT NULL
                LIMIT 1
            """, (sentence, word))
            row = cursor.fetchone()
            return dict(row) if row else None


def add_generation_batch(word: str, sentence_count: int = 0) -> int:
    """Track a generation batch."""
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO generation_batches (word, sentence_count)
                VALUES (?, ?)
            """, (word, sentence_count))
            conn.commit()
            return cursor.lastrowid


def delete_items_by_word(word: str) -> int:
    """
    Delete all training items for a specific word.
    
    Args:
        word: The word to delete items for
    
    Returns:
        Number of items deleted
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM training_items WHERE LOWER(word) = LOWER(?)", (word,))
            conn.commit()
            deleted_count = cursor.rowcount
    
    print(f"🗑️ Deleted {deleted_count} database entries for word '{word}'")
    return deleted_count



def add_error_report(word: str, explanation: str) -> int:
    """
    Add a new error report.
    
    Args:
        word: The word with error
        explanation: User explanation
        
    Returns:
        The ID of the created report
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO error_reports (word, explanation)
                VALUES (?, ?)
            """, (word, explanation))
            conn.commit()
            return cursor.lastrowid


def get_error_reports(status: Optional[str] = None) -> List[Dict]:
    """
    Get error reports.
    
    Args:
        status: Optional filter by status
        
    Returns:
        List of error reports
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM error_reports"
            params = []
            
            if status:
                query += " WHERE status = ?"
                params.append(status)
                
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


def delete_error_report(report_id: int) -> bool:
    """
    Delete an error report.
    
    Args:
        report_id: The ID to delete
        
    Returns:
        True if deleted
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM error_reports WHERE id = ?", (report_id,))
            conn.commit()
            return cursor.rowcount > 0


def update_error_report_status(report_id: int, status: str) -> bool:
    """
    Update status of an error report.
    
    Args:
        report_id: The ID to update
        status: New status
        
    Returns:
        True if updated
    """
    with _db_lock:
        with get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE error_reports 
                SET status = ?
                WHERE id = ?
            """, (status, report_id))
            conn.commit()
            return cursor.rowcount > 0


# Initialize on import
init_training_db()


