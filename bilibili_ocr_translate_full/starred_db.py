"""SQLite database for starred words in Learn Mode. Persistent across app sessions."""
import os
import sqlite3
import sys


def _db_path():
    """Path to the starred words database."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "starred_words.db")


def _get_conn():
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    """Create table if not exists."""
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS starred_words (
                word TEXT PRIMARY KEY,
                pinyin TEXT,
                definition TEXT,
                created_at REAL,
                provider TEXT,
                provider_display TEXT,
                model TEXT
            )
        """)
        # Add new columns if they don't exist (for existing databases)
        try:
            conn.execute("ALTER TABLE starred_words ADD COLUMN provider TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            conn.execute("ALTER TABLE starred_words ADD COLUMN provider_display TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE starred_words ADD COLUMN model TEXT")
        except sqlite3.OperationalError:
            pass


def add_star(word: str, pinyin: str = "", definition: str = "", provider: str = None, provider_display: str = None, model: str = None) -> bool:
    """Add a word to starred. Returns True if added, False if already existed."""
    _init_db()
    import time
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO starred_words (word, pinyin, definition, created_at, provider, provider_display, model) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (word, pinyin or "", definition or "", time.time(), provider or "", provider_display or "", model or ""),
            )
            return conn.total_changes > 0
    except Exception:
        return False


def remove_star(word: str) -> bool:
    """Remove a word from starred. Returns True if removed."""
    _init_db()
    try:
        with _get_conn() as conn:
            conn.execute("DELETE FROM starred_words WHERE word = ?", (word,))
            return conn.total_changes > 0
    except Exception:
        return False


def is_starred(word: str) -> bool:
    """Check if a word is starred."""
    _init_db()
    try:
        with _get_conn() as conn:
            row = conn.execute("SELECT 1 FROM starred_words WHERE word = ?", (word,)).fetchone()
            return row is not None
    except Exception:
        return False


def get_all_starred() -> list[dict]:
    """Return all starred words as [{word, pinyin, definition, _metadata}, ...] ordered by created_at desc."""
    _init_db()
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT word, pinyin, definition, provider, provider_display, model FROM starred_words ORDER BY created_at DESC"
            ).fetchall()
            result = []
            for r in rows:
                kw = {"word": r["word"], "pinyin": r["pinyin"] or "", "definition": r["definition"] or ""}
                # Add metadata if available
                if r["provider"] or r["provider_display"] or r["model"]:
                    kw["_metadata"] = {
                        "provider": r["provider"] or "",
                        "provider_display": r["provider_display"] or "",
                        "model": r["model"] or "",
                    }
                result.append(kw)
            return result
    except Exception:
        return []
