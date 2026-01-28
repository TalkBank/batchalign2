"""
Tests for the Batchalign caching system.

These tests verify:
1. CacheManager basic operations (get/put/clear/stats)
2. Cache key generation determinism
3. Version mismatch handling
4. Serialization/deserialization round-trips
"""

import tempfile
import shutil
from pathlib import Path
from typing import Any

import pytest

from batchalign.document import (
    Utterance, Form, Morphology, Dependency, TokenType, Tier
)
from batchalign.pipelines.cache import (
    CacheManager, MorphotagCacheKey, _get_batchalign_version
)


@pytest.fixture
def temp_cache_dir() -> Path:
    """Create a temporary directory for cache testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def cache_manager(temp_cache_dir: Path) -> CacheManager:
    """Create a CacheManager with a temporary directory."""
    return CacheManager(cache_dir=temp_cache_dir)


@pytest.fixture
def sample_utterance() -> Utterance:
    """Create a sample utterance for testing."""
    forms = [
        Form(text="hello", type=TokenType.REGULAR),
        Form(text="world", type=TokenType.REGULAR),
        Form(text=".", type=TokenType.PUNCT),
    ]
    return Utterance(content=forms, tier=Tier())


@pytest.fixture
def sample_utterance_with_morphology() -> Utterance:
    """Create a sample utterance with morphology and dependency."""
    forms = [
        Form(
            text="hello",
            type=TokenType.REGULAR,
            morphology=[Morphology(lemma="hello", pos="intj", feats="")],
            dependency=[Dependency(id=1, dep_id=0, dep_type="ROOT")],
        ),
        Form(
            text="world",
            type=TokenType.REGULAR,
            morphology=[Morphology(lemma="world", pos="noun", feats="-Sing")],
            dependency=[Dependency(id=2, dep_id=1, dep_type="VOCATIVE")],
        ),
        Form(
            text=".",
            type=TokenType.PUNCT,
            morphology=[Morphology(lemma=".", pos="punct", feats="")],
            dependency=[Dependency(id=3, dep_id=1, dep_type="PUNCT")],
        ),
    ]
    return Utterance(content=forms, tier=Tier())


class TestCacheManager:
    """Tests for CacheManager basic operations."""

    def test_init_creates_directory(self, temp_cache_dir: Path) -> None:
        """Test that CacheManager creates the cache directory."""
        cache_subdir = temp_cache_dir / "nested" / "cache"
        manager = CacheManager(cache_dir=cache_subdir)
        assert cache_subdir.exists()
        assert (cache_subdir / "cache.db").exists()

    def test_get_returns_none_for_missing_key(
        self, cache_manager: CacheManager
    ) -> None:
        """Test that get returns None for non-existent keys."""
        result = cache_manager.get("nonexistent", "morphosyntax", "1.0.0")
        assert result is None

    def test_put_and_get_basic(self, cache_manager: CacheManager) -> None:
        """Test basic put and get operations."""
        key = "test_key_123"
        task = "morphosyntax"
        engine_version = "1.11.0"
        ba_version = "0.8.0"
        data = {"morphology": [[{"lemma": "test", "pos": "noun", "feats": ""}]]}

        cache_manager.put(key, task, engine_version, ba_version, data)
        result = cache_manager.get(key, task, engine_version)

        assert result is not None
        assert result == data

    def test_get_batch(self, cache_manager: CacheManager) -> None:
        """Test retrieving multiple entries in one query."""
        task = "morphosyntax"
        version = "1.0.0"
        ba_version = "0.8.0"

        # Put some entries
        entries = {
            "k1": {"val": 1},
            "k2": {"val": 2},
            "k3": {"val": 3}
        }
        for k, v in entries.items():
            cache_manager.put(k, task, version, ba_version, v)

        # Get batch
        results = cache_manager.get_batch(["k1", "k2", "nonexistent", "k3"], task, version)

        assert len(results) == 3
        assert results["k1"] == {"val": 1}
        assert results["k2"] == {"val": 2}
        assert results["k3"] == {"val": 3}
        assert "nonexistent" not in results

    def test_put_batch(self, cache_manager: CacheManager) -> None:
        """Test storing multiple entries in one transaction."""
        task = "morphosyntax"
        version = "1.0.0"
        ba_version = "0.8.0"

        entries = [
            ("kb1", {"val": 10}),
            ("kb2", {"val": 20}),
            ("kb3", {"val": 30})
        ]

        cache_manager.put_batch(entries, task, version, ba_version)

        # Verify
        assert cache_manager.get("kb1", task, version) == {"val": 10}
        assert cache_manager.get("kb2", task, version) == {"val": 20}
        assert cache_manager.get("kb3", task, version) == {"val": 30}

    def test_get_returns_none_for_version_mismatch(
        self, cache_manager: CacheManager
    ) -> None:
        """Test that get returns None when engine version doesn't match."""
        key = "test_key_456"
        task = "morphosyntax"
        ba_version = "0.8.0"
        data = {"test": "data"}

        # Put with version 1.10.0
        cache_manager.put(key, task, "1.10.0", ba_version, data)

        # Get with version 1.11.0 should return None
        result = cache_manager.get(key, task, "1.11.0")
        assert result is None

        # Get with same version should work
        result = cache_manager.get(key, task, "1.10.0")
        assert result == data

    def test_put_replaces_existing(self, cache_manager: CacheManager) -> None:
        """Test that put replaces existing entries with same key."""
        key = "test_key_789"
        task = "morphosyntax"
        ba_version = "0.8.0"

        cache_manager.put(key, task, "1.0.0", ba_version, {"value": 1})
        cache_manager.put(key, task, "1.0.0", ba_version, {"value": 2})

        result = cache_manager.get(key, task, "1.0.0")
        assert result == {"value": 2}

    def test_clear_removes_all_entries(self, cache_manager: CacheManager) -> None:
        """Test that clear removes all cache entries."""
        # Add some entries
        cache_manager.put("key1", "morphosyntax", "1.0.0", "0.8.0", {"a": 1})
        cache_manager.put("key2", "morphosyntax", "1.0.0", "0.8.0", {"b": 2})

        # Verify they exist
        assert cache_manager.get("key1", "morphosyntax", "1.0.0") is not None
        assert cache_manager.get("key2", "morphosyntax", "1.0.0") is not None

        # Clear
        bytes_freed = cache_manager.clear()
        assert bytes_freed > 0

        # Verify they're gone
        assert cache_manager.get("key1", "morphosyntax", "1.0.0") is None
        assert cache_manager.get("key2", "morphosyntax", "1.0.0") is None

    def test_stats_returns_correct_counts(
        self, cache_manager: CacheManager
    ) -> None:
        """Test that stats returns accurate statistics."""
        # Add entries with different tasks and versions
        cache_manager.put("key1", "morphosyntax", "1.10.0", "0.8.0", {"a": 1})
        cache_manager.put("key2", "morphosyntax", "1.10.0", "0.8.0", {"b": 2})
        cache_manager.put("key3", "morphosyntax", "1.11.0", "0.8.0", {"c": 3})

        stats = cache_manager.stats()

        assert stats["total_entries"] == 3
        assert stats["by_task"]["morphosyntax"] == 3
        assert "morphosyntax 1.10.0" in stats["by_engine_version"]
        assert stats["by_engine_version"]["morphosyntax 1.10.0"] == 2
        assert stats["by_engine_version"]["morphosyntax 1.11.0"] == 1


class TestMorphotagCacheKey:
    """Tests for MorphotagCacheKey generation and serialization."""

    def test_generate_key_deterministic(
        self, sample_utterance: Utterance
    ) -> None:
        """Test that the same inputs always produce the same key."""
        key_gen = MorphotagCacheKey()

        key1 = key_gen.generate_key(
            sample_utterance, lang="eng", retokenize=False, mwt={}
        )
        key2 = key_gen.generate_key(
            sample_utterance, lang="eng", retokenize=False, mwt={}
        )

        assert key1 == key2
        # Keys should be SHA256 hashes (64 hex characters)
        assert len(key1) == 64

    def test_generate_key_different_for_different_inputs(
        self, sample_utterance: Utterance
    ) -> None:
        """Test that different inputs produce different keys."""
        key_gen = MorphotagCacheKey()

        key1 = key_gen.generate_key(
            sample_utterance, lang="eng", retokenize=False, mwt={}
        )
        key2 = key_gen.generate_key(
            sample_utterance, lang="spa", retokenize=False, mwt={}
        )
        key3 = key_gen.generate_key(
            sample_utterance, lang="eng", retokenize=True, mwt={}
        )
        key4 = key_gen.generate_key(
            sample_utterance, lang="eng", retokenize=False, mwt={"don't": ("do", "n't")}
        )

        # All keys should be different
        keys = {key1, key2, key3, key4}
        assert len(keys) == 4

    def test_serialize_and_deserialize_roundtrip(
        self, sample_utterance_with_morphology: Utterance
    ) -> None:
        """Test that serialize/deserialize produces identical results."""
        key_gen = MorphotagCacheKey()

        # Serialize
        data = key_gen.serialize_output(sample_utterance_with_morphology)

        # Create a new utterance without morphology
        new_forms = [
            Form(text="hello", type=TokenType.REGULAR),
            Form(text="world", type=TokenType.REGULAR),
            Form(text=".", type=TokenType.PUNCT),
        ]
        new_utterance = Utterance(content=new_forms, tier=Tier())

        # Deserialize
        key_gen.deserialize_output(data, new_utterance)

        # Verify morphology was applied
        assert new_utterance.content[0].morphology is not None
        assert new_utterance.content[0].morphology[0].lemma == "hello"
        assert new_utterance.content[0].morphology[0].pos == "intj"

        # Verify dependency was applied
        assert new_utterance.content[0].dependency is not None
        assert new_utterance.content[0].dependency[0].id == 1
        assert new_utterance.content[0].dependency[0].dep_type == "ROOT"

    def test_deserialize_handles_length_mismatch(
        self, sample_utterance_with_morphology: Utterance
    ) -> None:
        """Test that deserialize handles mismatched form counts gracefully."""
        key_gen = MorphotagCacheKey()

        # Serialize a 3-form utterance
        data = key_gen.serialize_output(sample_utterance_with_morphology)

        # Create a 2-form utterance
        new_forms = [
            Form(text="hello", type=TokenType.REGULAR),
            Form(text=".", type=TokenType.PUNCT),
        ]
        new_utterance = Utterance(content=new_forms, tier=Tier())

        # Deserialize should not crash, but should not modify
        key_gen.deserialize_output(data, new_utterance)

        # Morphology should remain None due to mismatch
        assert new_utterance.content[0].morphology is None


class TestVersionDetection:
    """Tests for version detection utilities."""

    def test_get_batchalign_version_returns_string(self) -> None:
        """Test that version detection returns a non-empty string."""
        version = _get_batchalign_version()
        assert isinstance(version, str)
        assert len(version) > 0


def _concurrent_cache_worker(
    cache_dir_str: str, worker_id: int, result_queue: Any
) -> None:
    """Worker function for concurrent cache access test.

    Must be at module level for pickling to work with multiprocessing.
    """
    import time
    from pathlib import Path

    try:
        manager = CacheManager(cache_dir=Path(cache_dir_str))

        # Write some entries
        for i in range(10):
            key = f"worker_{worker_id}_key_{i}"
            manager.put(
                key,
                "morphosyntax",
                "1.0.0",
                "0.8.0",
                {"worker": worker_id, "index": i}
            )

        # Read entries from all workers (including self)
        reads = 0
        for w in range(4):
            for i in range(10):
                key = f"worker_{w}_key_{i}"
                # Retry with backoff on busy
                for attempt in range(3):
                    result = manager.get(key, "morphosyntax", "1.0.0")
                    if result is not None:
                        reads += 1
                        break
                    time.sleep(0.01 * (attempt + 1))

        result_queue.put(("success", worker_id, reads))
    except Exception as e:
        result_queue.put(("error", worker_id, str(e)))


class TestConcurrentAccess:
    """Tests for concurrent cache access."""

    def test_multiple_processes_can_read_write(
        self, temp_cache_dir: Path
    ) -> None:
        """Test that multiple processes can access the cache simultaneously."""
        import multiprocessing

        # Create processes
        result_queue: multiprocessing.Queue[tuple[str, int, int | str]] = multiprocessing.Queue()
        processes = []
        for worker_id in range(4):
            p = multiprocessing.Process(
                target=_concurrent_cache_worker,
                args=(str(temp_cache_dir), worker_id, result_queue)
            )
            processes.append(p)

        # Start all processes
        for p in processes:
            p.start()

        # Wait for completion
        for p in processes:
            p.join(timeout=30)

        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        # Check all workers completed successfully
        assert len(results) == 4
        for status, worker_id, data in results:
            assert status == "success", f"Worker {worker_id} failed: {data}"

        # Verify cache has all entries
        manager = CacheManager(cache_dir=temp_cache_dir)
        stats = manager.stats()
        assert stats["total_entries"] == 40  # 4 workers * 10 entries each
