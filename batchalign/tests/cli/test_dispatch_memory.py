import builtins
import json
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

    def submit(self, _fn, _args):
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
        "commands": {"align": {"peaks": [1024 * 1024 * 1024], "sizes": [1024]}}
    }))

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
