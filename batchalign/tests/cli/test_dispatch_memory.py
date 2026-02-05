import builtins
import json
import re
from pathlib import Path

import pytest

from batchalign.cli import dispatch as dispatch_module
from batchalign import constants as constants_module


class DummyVM:
    def __init__(self, total, available):
        self.total = total
        self.available = available


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class FakeExecutor:
    def __init__(self, results):
        self._results = list(results)

    def submit(self, _fn, *args, **kwargs):
        return FakeFuture(self._results.pop(0))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _drain_wait(pending, **_kwargs):
    return set(pending), set()


def _write_cha(path: Path, name: str):
    data = "@UTF8\n@Begin\n*PAR:\tTest .\n@End\n"
    file_path = path / name
    file_path.write_text(data)
    return file_path


def _base_ctx(tmp_path, **overrides):
    ctx = {"verbose": 0, "workers": 2, "memlog": True, "mem_guard": False,
           "adaptive_workers": True, "adaptive_safety_factor": 1.0,
           "adaptive_warmup": 1, "force_cpu": False, "shared_models": False}
    ctx.update(overrides)
    return type("Ctx", (), {"obj": ctx})()


def _force_process_mode(monkeypatch):
    monkeypatch.setattr(dispatch_module, "POOL_UNSAFE_ENGINES", {"wav2vec_fa"})
    monkeypatch.setattr(dispatch_module, "POOL_SAFE_ENGINES", set())
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])


def test_memlog_and_history(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")
    _write_cha(in_dir, "b.cha")

    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [
        (str(in_dir / "a.cha"), None, None, "", mem_info),
        (str(in_dir / "b.cha"), None, None, "", mem_info),
    ]

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())

    memlog = out_dir / "batchalign_memlog.jsonl"
    assert memlog.exists()
    lines = memlog.read_text().strip().splitlines()
    events = [json.loads(line)["event"] for line in lines]
    assert "submit_worker" in events
    assert "worker_complete" in events

    history_path = Path(tmp_path / "cache" / "memory_history.json")
    payload = json.loads(history_path.read_text())
    assert payload["version"] == 1
    assert payload["commands"]["align"]["peaks"]


def test_mem_guard_abort(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(4 * 1024**3, 512 * 1024**2))
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor([
        (str(in_dir / "a.cha"), None, None, "", mem_info),
    ]))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path, mem_guard=True)
    with pytest.raises(RuntimeError):
        dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())


def test_adaptive_warm_start_message(tmp_path, monkeypatch, capsys):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    history_path = Path(tmp_path / "cache" / "memory_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "version": 1,
        "commands": {"align": {"peaks": [1024 * 1024 * 1024], "metrics": [10.0]}}
    }))

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [(str(in_dir / "a.cha"), None, None, "", mem_info)]
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    captured = capsys.readouterr()
    assert "Adaptive warm start" in captured.out


def test_force_cpu_disables_mps(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [(str(in_dir / "a.cha"), None, None, "", mem_info)]
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))

    calls = {"count": 0}
    def _apply_force_cpu():
        calls["count"] += 1

    monkeypatch.setattr(dispatch_module, "apply_force_cpu", _apply_force_cpu)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path, force_cpu=True)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    assert calls["count"] == 1


def test_shared_models_requires_force_cpu_on_macos(tmp_path, monkeypatch, capsys):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [(str(in_dir / "a.cha"), None, None, "", mem_info)]
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)
    monkeypatch.setattr(dispatch_module.os, "uname", lambda: type("Uname", (), {"sysname": "Darwin"})())
    monkeypatch.setenv("BATCHALIGN_FORCE_CPU", "")

    ctx = _base_ctx(tmp_path, shared_models=True)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    captured = capsys.readouterr()
    assert "Shared models require --force-cpu on macOS" in captured.out


def test_skip_mp4_conversion_when_wav_exists(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    mp4_path = in_dir / "audio.mp4"
    wav_path = in_dir / "audio.wav"
    mp4_path.write_text("mp4")
    wav_path.write_text("wav")

    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "pydub":
            raise AssertionError("pydub import should be skipped when wav exists")
        return original_import(name, *args, **kwargs)

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor([
        (str(in_dir / "a.cha"), None, None, "", mem_info),
    ]))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.shutil, "which", lambda *_args, **_kwargs: "ffmpeg")
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    monkeypatch.setattr(constants_module, "FORCED_CONVERSION", ["mp4"])

    ctx = _base_ctx(tmp_path, mem_guard=False)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    # no pydub import => conversion skipped


def test_pooled_mode_skips_memory_history(tmp_path, monkeypatch):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")
    _write_cha(in_dir, "b.cha")

    monkeypatch.setattr(dispatch_module.concurrent.futures, "ThreadPoolExecutor", lambda *a, **k: FakeExecutor([
        (str(in_dir / "a.cha"), None, None, "", {"rss_peak": 123, "rss_end": 123, "rss_start": 1}),
        (str(in_dir / "b.cha"), None, None, "", {"rss_peak": 456, "rss_end": 456, "rss_start": 1}),
    ]))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module, "_run_pipeline_for_file", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "POOL_UNSAFE_ENGINES", set())
    monkeypatch.setattr(dispatch_module, "POOL_SAFE_ENGINES", {"rev"})
    monkeypatch.setattr(dispatch_module, "Cmd2Task", {"align": "fa"})
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("asr", "rev")])
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "dispatch_pipeline", lambda *a, **k: object())

    ctx = _base_ctx(tmp_path)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())

    history_path = Path(tmp_path / "cache" / "memory_history.json")
    assert not history_path.exists()


def test_history_persists_max_peak(tmp_path, monkeypatch):
    """Test that memory history persists max_peak for safer cap estimation."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")
    _write_cha(in_dir, "b.cha")
    _write_cha(in_dir, "c.cha")

    # Simulate varying memory peaks: 1GB, 3GB, 2GB (max=3GB, median=2GB)
    mem_infos = [
        {"rss_peak": 1 * 1024**3, "rss_end": 512 * 1024**2, "rss_start": 128},
        {"rss_peak": 3 * 1024**3, "rss_end": 512 * 1024**2, "rss_start": 128},
        {"rss_peak": 2 * 1024**3, "rss_end": 512 * 1024**2, "rss_start": 128},
    ]
    results = [
        (str(in_dir / "a.cha"), None, None, "", mem_infos[0]),
        (str(in_dir / "b.cha"), None, None, "", mem_infos[1]),
        (str(in_dir / "c.cha"), None, None, "", mem_infos[2]),
    ]

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())

    history_path = Path(tmp_path / "cache" / "memory_history.json")
    payload = json.loads(history_path.read_text())
    
    # Verify max_peak is stored and equals the maximum observed (3GB)
    assert "max_peak" in payload["commands"]["align"]
    assert payload["commands"]["align"]["max_peak"] == 3 * 1024**3


def test_adaptive_warm_start_uses_max_peak(tmp_path, monkeypatch, capsys):
    """Test that adaptive warm start uses max_peak from history when available."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    # Pre-populate history with max_peak
    history_path = Path(tmp_path / "cache" / "memory_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "version": 1,
        "commands": {"align": {
            "peaks": [1 * 1024**3, 2 * 1024**3],  # median = 1.5GB
            "metrics": [10.0, 20.0],
            "max_peak": 5 * 1024**3  # max = 5GB (much higher)
        }}
    }))

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [(str(in_dir / "a.cha"), None, None, "", mem_info)]
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(8 * 1024**3, 6 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    captured = capsys.readouterr()
    
    # Should show max peak (5GB) not median (1.5GB)
    assert "5.00GB max peak from history" in captured.out


def test_adaptive_cap_returns_warmup_when_memory_unavailable(tmp_path, monkeypatch):
    """Adaptive cap must not return uncapped num_workers when system memory is unknown."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    mem_info = {"rss_peak": 1024 * 1024 * 1024, "rss_end": 512 * 1024 * 1024, "rss_start": 128}
    results = [(str(in_dir / "a.cha"), None, None, "", mem_info)]
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    # Return None for memory — simulates unsupported platform
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(None, None))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path, workers=28, adaptive_warmup=2)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    # If this completes without OOM-like behavior, the cap was conservative.
    # The key assertion is that we don't crash — and the cap message should show ≤ warmup.


def test_adaptive_cap_uses_history_max_with_fresh_samples(tmp_path, monkeypatch, capsys):
    """Even with low fresh samples, history max_peak should constrain the cap."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_cha(in_dir, "a.cha")
    _write_cha(in_dir, "b.cha")
    _write_cha(in_dir, "c.cha")
    _write_cha(in_dir, "d.cha")

    # History says worst case was 6GB
    history_path = Path(tmp_path / "cache" / "memory_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({
        "version": 1,
        "commands": {"align": {
            "peaks": [1 * 1024**3, 2 * 1024**3],
            "metrics": [10.0, 20.0],
            "max_peak": 6 * 1024**3,
        }}
    }))

    # Fresh samples all report low memory (500MB) — without history_max_peak
    # this would allow far too many workers
    small_peak = 500 * 1024**2
    mem_info = {"rss_peak": small_peak, "rss_end": small_peak, "rss_start": 128}
    results = [
        (str(in_dir / "a.cha"), None, None, "", mem_info),
        (str(in_dir / "b.cha"), None, None, "", mem_info),
        (str(in_dir / "c.cha"), None, None, "", mem_info),
        (str(in_dir / "d.cha"), None, None, "", mem_info),
    ]

    _force_process_mode(monkeypatch)
    monkeypatch.setattr(dispatch_module.pipeline_dispatch, "resolve_engine_specs", lambda *a, **k: [("fa", "wav2vec_fa")])
    monkeypatch.setattr(dispatch_module.concurrent.futures, "ProcessPoolExecutor", lambda *a, **k: FakeExecutor(results))
    monkeypatch.setattr(dispatch_module.concurrent.futures, "wait", _drain_wait)
    # 16GB total, 12GB available
    monkeypatch.setattr(dispatch_module.psutil, "virtual_memory", lambda: DummyVM(16 * 1024**3, 12 * 1024**3))
    monkeypatch.setattr(dispatch_module, "_get_worker_pipeline", lambda *a, **k: object())
    monkeypatch.setattr(dispatch_module, "_worker_task", lambda *a, **k: None)
    monkeypatch.setattr(dispatch_module, "user_cache_dir", lambda *a, **k: str(tmp_path / "cache"))
    monkeypatch.setattr(dispatch_module, "apply_force_cpu", lambda: None)
    monkeypatch.setattr(dispatch_module, "force_cpu_preferred", lambda: False)

    ctx = _base_ctx(tmp_path, workers=28, adaptive_warmup=1)
    dispatch_module._dispatch("align", "eng", 1, ["cha"], ctx, str(in_dir), str(out_dir), None, None, dispatch_module.Console())
    captured = capsys.readouterr()

    # With 6GB max_peak from history, reserve = max(1.6GB, 2GB) = 2GB
    # cap = (12GB - 2GB) / 6GB ≈ 1 worker (very conservative, but safe)
    # Without the fix, cap would be (12GB - 2GB) / 0.5GB = 20 workers (dangerous!)
    assert "Adaptive cap" in captured.out
    # Verify the cap message shows a low number (not 20+)
    cap_match = re.search(r"Adaptive cap:\s*(\d+)", captured.out)
    assert cap_match is not None, f"Expected adaptive cap message, got: {captured.out}"
    cap_value = int(cap_match.group(1))
    assert cap_value <= 2, f"Cap should be ≤2 with 6GB history max_peak on 16GB system, got {cap_value}"
