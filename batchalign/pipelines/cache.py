"""
cache.py
Caching infrastructure for Batchalign pipelines.

This module provides a SQLite-based cache for storing per-utterance analysis
results, enabling efficient re-processing of unchanged content. The cache uses
WAL mode for concurrent access from multiple worker processes.

Architecture:
- CacheKeyGenerator: Abstract base for task-specific key generation and serialization
- MorphotagCacheKey: Implementation for morphosyntax task caching
- UtteranceSegmentationCacheKey: Implementation for utterance segmentation task caching
- AlignmentCacheKey: Implementation for forced alignment caching
- CacheManager: Thread-safe SQLite operations with automatic connection management
"""

from abc import ABC, abstractmethod
from pathlib import Path
import hashlib
import json
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Any

from platformdirs import user_cache_dir
from filelock import FileLock

from batchalign.document import Utterance, Morphology, Dependency

L = logging.getLogger("batchalign")


def _get_batchalign_version() -> str:
    """Read the batchalign version from the version file."""
    try:
        version_path = Path(__file__).parent.parent / "version"
        with open(version_path, 'r') as f:
            return f.readline().strip()
    except Exception:
        return "unknown"


class CacheKeyGenerator(ABC):
    """Abstract base for task-specific cache key generation and serialization.

    Each task type (morphosyntax, alignment, etc.) needs its own implementation
    to define:
    1. How to generate a unique key from input parameters
    2. How to serialize output data for caching
    3. How to deserialize cached data back into utterance fields
    """

    @abstractmethod
    def generate_key(self, utterance: Utterance, *args: Any, **options: Any) -> str:
        """Generate a unique cache key for the given utterance and options.

        The key should be deterministic: same inputs must produce same key.

        Args:
            utterance: The input utterance to process
            *args: Task-specific positional arguments
            **options: Task-specific options that affect output

        Returns:
            A SHA256 hash string uniquely identifying this input combination.
        """
        ...

    @abstractmethod
    def serialize_output(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Extract cacheable data from processed results.

        Returns:
            A JSON-serializable dictionary containing the computed results.
        """
        ...

    @abstractmethod
    def deserialize_output(
        self, data: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Apply cached data to results or return new objects.
        """
        ...


class MorphotagCacheKey(CacheKeyGenerator):
    """Cache key generator for morphosyntax (morphotag) task.

    The cache key is computed as:
        SHA256(normalized_text + lang + retokenize + mwt_hash)

    where:
    - normalized_text: utterance.strip(join_with_spaces=True)
    - lang: Document language code (e.g., "eng")
    - retokenize: Boolean flag as "0" or "1"
    - mwt_hash: SHA256 of sorted MWT dictionary items (or empty if none)

    The cached data stores Form.morphology and Form.dependency for each
    form in the utterance.
    """

    def generate_key(
        self,
        utterance: Utterance,
        lang: str,
        retokenize: bool,
        mwt: dict[str, Any]
    ) -> str:
        """Generate cache key from utterance text and processing options.

        Args:
            utterance: The input utterance
            lang: Language code (e.g., "eng")
            retokenize: Whether retokenization is enabled
            mwt: Multi-word token dictionary (may be empty)

        Returns:
            A SHA256 hash string.
        """
        # Normalize text: strip to core content with spaces
        normalized_text = utterance.strip(join_with_spaces=True)

        # Compute MWT hash
        if mwt:
            mwt_items = sorted((k, tuple(v) if isinstance(v, (list, tuple)) else (v,))
                               for k, v in mwt.items())
            mwt_str = str(mwt_items)
            mwt_hash = hashlib.sha256(mwt_str.encode()).hexdigest()
        else:
            mwt_hash = ""

        # Combine all components
        key_parts = [
            normalized_text,
            lang,
            "1" if retokenize else "0",
            mwt_hash,
        ]
        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def serialize_output(self, utterance: Utterance) -> dict[str, Any]:
        """Extract morphology and dependency data from utterance forms.

        Args:
            utterance: The processed utterance with morphology/dependency

        Returns:
            Dictionary with 'morphology' and 'dependency' lists,
            plus 'retokenized_text' if the utterance has custom text.
        """
        morphology_data: list[list[dict[str, Any]] | None] = []
        dependency_data: list[list[dict[str, Any]] | None] = []

        for form in utterance.content:
            # Serialize morphology
            if form.morphology:
                morph_list = [
                    {
                        "lemma": m.lemma,
                        "pos": m.pos,
                        "feats": m.feats,
                    }
                    for m in form.morphology
                ]
                morphology_data.append(morph_list)
            else:
                morphology_data.append(None)

            # Serialize dependency
            if form.dependency:
                dep_list = [
                    {
                        "id": d.id,
                        "dep_id": d.dep_id,
                        "dep_type": d.dep_type,
                    }
                    for d in form.dependency
                ]
                dependency_data.append(dep_list)
            else:
                dependency_data.append(None)

        result: dict[str, Any] = {
            "morphology": morphology_data,
            "dependency": dependency_data,
        }

        # Include retokenized text if present
        if utterance.text is not None:
            result["retokenized_text"] = utterance.text

        return result

    def deserialize_output(
        self, data: dict[str, Any], utterance: Utterance
    ) -> None:
        """Apply cached morphology/dependency to utterance forms.

        Modifies the utterance in-place. If the cached data has a different
        number of forms than the utterance, logs a warning and does nothing.

        Args:
            data: Cached data from serialize_output
            utterance: The utterance to apply cached results to
        """
        morphology_data = data.get("morphology", [])
        dependency_data = data.get("dependency", [])

        # Check for length mismatch
        if len(morphology_data) != len(utterance.content):
            L.warning(
                f"Cache deserialize: form count mismatch "
                f"(cached={len(morphology_data)}, utterance={len(utterance.content)})"
            )
            return

        # Apply morphology and dependency to each form
        for idx, form in enumerate(utterance.content):
            morph_list = morphology_data[idx]
            if morph_list is not None:
                form.morphology = [
                    Morphology(
                        lemma=m["lemma"],
                        pos=m["pos"],
                        feats=m["feats"],
                    )
                    for m in morph_list
                ]

            dep_list = dependency_data[idx] if idx < len(dependency_data) else None
            if dep_list is not None:
                form.dependency = [
                    Dependency(
                        id=d["id"],
                        dep_id=d["dep_id"],
                        dep_type=d["dep_type"],
                    )
                    for d in dep_list
                ]

        # Apply retokenized text if present
        if "retokenized_text" in data:
            utterance.text = data["retokenized_text"]


class UtteranceSegmentationCacheKey(CacheKeyGenerator):
    """Cache key generator for utterance segmentation (utseg) task.

    The cache key is computed as:
        SHA256(normalized_text + lang)

    where:
    - normalized_text: utterance.strip(join_with_spaces=True)
    - lang: Document primary language code

    The cached data stores the resulting utterances (their content and timing).
    Since utseg splits one utterance into multiple, serialize_output takes
    a LIST of utterances.
    """

    def generate_key(
        self,
        utterance: Utterance,
        lang: str,
    ) -> str:
        """Generate cache key from utterance text and language.

        Args:
            utterance: The input utterance
            lang: Language code

        Returns:
            A SHA256 hash string.
        """
        # Normalize text: strip to core content with spaces
        normalized_text = utterance.strip(join_with_spaces=True)

        # Combine components
        combined = f"{normalized_text}|{lang}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def serialize_output(self, utterances: list[Utterance]) -> dict[str, Any]:
        """Extract content and timing from a list of resulting utterances.

        Args:
            utterances: The list of utterances resulting from segmentation

        Returns:
            Dictionary containing serialized utterance data.
        """
        results = []
        for ut in utterances:
            # We only need the text/type of each form, plus timing if available
            forms_data = [
                {
                    "text": form.text,
                    "type": int(form.type),
                    "time": form.time,
                }
                for form in ut.content
            ]
            results.append({
                "content": forms_data,
                "text": ut.text,
                "time": ut.time
            })

        return {"utterances": results}

    def deserialize_output(
        self, data: dict[str, Any], utterance: Utterance
    ) -> list[Utterance]:
        """Convert cached data back into a list of Utterance objects.

        This doesn't modify the input utterance in-place because it produces
        MULTIPLE utterances.

        Args:
            data: Cached data from serialize_output
            utterance: The original utterance (to copy tier info from)

        Returns:
            List of new Utterance objects.
        """
        from batchalign.document import Form, TokenType

        results = []
        tier = utterance.tier

        for ut_data in data.get("utterances", []):
            content = [
                Form(
                    text=f["text"],
                    type=TokenType(f["type"]),
                    time=tuple(f["time"]) if f["time"] else None
                )
                for f in ut_data["content"]
            ]
            new_ut = Utterance(
                content=content,
                tier=tier,
                text=ut_data.get("text"),
                time=tuple(ut_data["time"]) if ut_data.get("time") else None
            )
            results.append(new_ut)

        return results


class AlignmentCacheKey(CacheKeyGenerator):
    """Cache key generator for forced alignment (fa) task.

    The cache key is computed as:
        SHA256(audio_hash + normalized_text + pauses_flag)
    """

    def generate_key(
        self,
        text: str,
        audio_hash: str,
        pauses: bool
    ) -> str:
        """Generate cache key from audio segment hash and text.

        Args:
            text: The normalized text being aligned
            audio_hash: tiny hash of the audio segment
            pauses: Whether pauses were requested

        Returns:
            A SHA256 hash string.
        """
        combined = f"{audio_hash}|{text}|{'1' if pauses else '0'}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def serialize_output(self, res: list[tuple[str, Any]]) -> dict[str, Any]:
        """Extract word-level timings from whisper result.

        Args:
            res: List of (word, timing) tuples from WhisperFAModel

        Returns:
            Dictionary with 'timings' list.
        """
        return {"timings": [time for word, time in res]}

    def deserialize_output(self, data: dict[str, Any]) -> list[Any]:
        """Get cached word timings.

        Args:
            data: Cached data

        Returns:
            List of timing values.
        """
        return data.get("timings", [])


class CacheManager:
    """Thread-safe SQLite cache with WAL mode for concurrent access.

    The cache stores per-utterance analysis results keyed by a hash of the
    input text and processing options. Each worker process creates its own
    connection; SQLite WAL mode handles synchronization via shared memory.

    Attributes:
        cache_dir: Path to the cache directory (default: system-appropriate cache dir)
        db_path: Path to the SQLite database file
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize cache at system-appropriate cache directory or custom directory.

        Args:
            cache_dir: Optional custom cache directory. If None, uses
                       platformdirs.user_cache_dir("batchalign", "batchalign") as the default location.
        """
        if cache_dir is None:
            cache_dir = Path(user_cache_dir("batchalign", "batchalign"))
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "cache.db"
        self._connection: sqlite3.Connection | None = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the database connection for this process.

        Each process needs its own connection. The connection is cached
        per CacheManager instance.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # Wait up to 30 seconds for locks
                isolation_level=None,  # Autocommit mode for WAL
            )
            # Configure WAL mode and pragmas for concurrent access
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            self._connection.execute("PRAGMA synchronous = NORMAL")
        return self._connection

    def _init_db(self) -> None:
        """Initialize the database schema if not already present.

        Uses a file lock to ensure only one process creates the table, preventing
        database locking errors during concurrent initialization.
        """
        lock_path = self.cache_dir / "cache.db.lock"
        lock = FileLock(lock_path)

        with lock:
            conn = self._get_connection()

            # Check if table exists to avoid unnecessary DDL which can cause locking
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cache_entries'"
            )
            if cursor.fetchone():
                return

            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        task TEXT NOT NULL,
                        engine_version TEXT NOT NULL,
                        batchalign_version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        data BLOB NOT NULL
                    )
                """)
                # Create indexes for common queries
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_task ON cache_entries(task)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_created ON cache_entries(created_at)"
                )
            except sqlite3.OperationalError:
                # If another process somehow created it between check and create (unlikely with lock)
                pass

    def get(self, key: str, task: str, engine_version: str) -> dict[str, Any] | None:
        """Retrieve a cached entry if it exists and version matches.

        Args:
            key: The cache key (SHA256 hash of inputs)
            task: The task type (e.g., "morphosyntax")
            engine_version: The expected engine version (e.g., stanza version)

        Returns:
            The cached data dictionary, or None if not found or version mismatch.
        """
        conn = self._get_connection()
        cursor = conn.execute(
            """
            SELECT data, engine_version FROM cache_entries
            WHERE key = ? AND task = ?
            """,
            (key, task)
        )
        row = cursor.fetchone()
        if row is None:
            L.debug(f"Cache MISS for {task}: {key[:16]}... (not found)")
            return None

        data_blob, stored_version = row
        if stored_version != engine_version:
            L.debug(
                f"Cache MISS for {task}: {key[:16]}... "
                f"(version mismatch: stored={stored_version}, expected={engine_version})"
            )
            return None

        L.debug(f"Cache HIT for {task}: {key[:16]}...")
        return json.loads(data_blob)

    def get_batch(self, keys: list[str], task: str, engine_version: str) -> dict[str, dict[str, Any]]:
        """Retrieve multiple cached entries in a single query.

        Args:
            keys: List of cache keys (SHA256 hashes)
            task: The task type
            engine_version: The expected engine version

        Returns:
            A dictionary mapping key to cached data dictionary.
            Only keys that were found and had matching version are included.
        """
        if not keys:
            return {}

        conn = self._get_connection()
        # SQLite has a limit on the number of parameters (usually 999 or 32766)
        # 900 is a safe chunk size.
        CHUNK_SIZE = 900
        results = {}

        for i in range(0, len(keys), CHUNK_SIZE):
            chunk = keys[i:i + CHUNK_SIZE]
            placeholders = ",".join(["?"] * len(chunk))
            cursor = conn.execute(
                f"""
                SELECT key, data, engine_version FROM cache_entries
                WHERE key IN ({placeholders}) AND task = ?
                """,
                (*chunk, task)
            )
            for key, data_blob, stored_version in cursor.fetchall():
                if stored_version == engine_version:
                    results[key] = json.loads(data_blob)
                else:
                    L.debug(
                        f"Cache MISS for {task}: {key[:16]}... "
                        f"(version mismatch: stored={stored_version}, expected={engine_version})"
                    )

        return results

    def put_batch(
        self,
        entries: list[tuple[str, dict[str, Any]]],
        task: str,
        engine_version: str,
        batchalign_version: str
    ) -> None:
        """Store multiple entries in the cache in a single transaction.

        Args:
            entries: List of (key, data) tuples
            task: The task type
            engine_version: The engine version
            batchalign_version: The batchalign version
        """
        if not entries:
            return

        import time
        conn = self._get_connection()
        created_at = datetime.now(timezone.utc).isoformat()

        # Use transaction for batch insertion with retry logic
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn.execute("BEGIN TRANSACTION")
                for key, data in entries:
                    data_blob = json.dumps(data)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO cache_entries
                        (key, task, engine_version, batchalign_version, created_at, data)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (key, task, engine_version, batchalign_version, created_at, data_blob)
                    )
                conn.execute("COMMIT")
                return
            except sqlite3.OperationalError as e:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def put(
        self,
        key: str,
        task: str,
        engine_version: str,
        batchalign_version: str,
        data: dict[str, Any]
    ) -> None:
        """Store an entry in the cache.

        Args:
            key: The cache key (SHA256 hash of inputs)
            task: The task type (e.g., "morphosyntax")
            engine_version: The engine version used to produce this result
            batchalign_version: The batchalign version
            data: The data to cache (will be JSON-encoded)
        """
        import time

        conn = self._get_connection()
        created_at = datetime.now(timezone.utc).isoformat()
        data_blob = json.dumps(data)

        # Use INSERT OR REPLACE with retry logic for concurrent writes
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache_entries
                    (key, task, engine_version, batchalign_version, created_at, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, task, engine_version, batchalign_version, created_at, data_blob)
                )
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise

    def clear(self) -> int:
        """Delete all cache entries and return bytes freed.

        This removes the database files and recreates an empty database.

        Returns:
            The number of bytes freed by clearing the cache.
        """
        # Calculate size before clearing
        bytes_freed = 0
        for suffix in ["", "-wal", "-shm"]:
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                bytes_freed += path.stat().st_size

        # Close connection
        if self._connection is not None:
            self._connection.close()
            self._connection = None

        # Remove files
        for suffix in ["", "-wal", "-shm"]:
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                path.unlink()

        # Recreate empty database
        self._init_db()

        return bytes_freed

    def stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            A dictionary containing:
            - location: Path to the cache database
            - size_bytes: Total size in bytes
            - total_entries: Total number of cached entries
            - by_task: Dict mapping task names to entry counts
            - by_engine_version: Dict mapping "task engine_version" to counts
        """
        conn = self._get_connection()

        # Calculate total size
        size_bytes = 0
        for suffix in ["", "-wal", "-shm"]:
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                size_bytes += path.stat().st_size

        # Count total entries
        cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
        total_entries = cursor.fetchone()[0]

        # Count by task
        cursor = conn.execute(
            "SELECT task, COUNT(*) FROM cache_entries GROUP BY task"
        )
        by_task = {row[0]: row[1] for row in cursor.fetchall()}

        # Count by engine version
        cursor = conn.execute(
            """
            SELECT task, engine_version, COUNT(*)
            FROM cache_entries
            GROUP BY task, engine_version
            """
        )
        by_engine_version = {}
        for task, version, count in cursor.fetchall():
            by_engine_version[f"{task} {version}"] = count

        return {
            "location": str(self.db_path),
            "size_bytes": size_bytes,
            "total_entries": total_entries,
            "by_task": by_task,
            "by_engine_version": by_engine_version,
        }