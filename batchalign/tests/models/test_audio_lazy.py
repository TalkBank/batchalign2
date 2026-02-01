import torch

from batchalign.models.utils import ASRAudioFile


def test_lazy_chunk_uses_partial_load(monkeypatch):
    calls = {"load": 0}

    seen = {"offset": None, "frames": None}

    def _fake_load(_path, frame_offset=0, num_frames=-1, **_kwargs):
        calls["load"] += 1
        seen["offset"] = frame_offset
        seen["frames"] = num_frames
        samples = max(int(num_frames), 0) if num_frames >= 0 else 0
        if samples == 0:
            samples = 10
        return torch.zeros(1, samples), 16000

    import torchaudio
    monkeypatch.setattr(torchaudio, "info", lambda _path: type("Info", (), {"sample_rate": 16000, "num_frames": 16000})(), raising=True)
    monkeypatch.setattr(torchaudio, "load", _fake_load, raising=True)

    audio = ASRAudioFile.lazy("fake.wav", 16000)
    chunk = audio.chunk(0, 1000)

    assert calls["load"] == 1
    assert seen["offset"] == 0
    assert seen["frames"] > 0
    assert chunk.numel() > 0


def test_lazy_audio_flag_disables_lazy(monkeypatch):
    from batchalign.models import utils as utils_module

    utils_module.set_lazy_audio_enabled(False)
    try:
        try:
            ASRAudioFile.lazy("fake.wav", 16000)
            assert False, "Expected lazy audio to be disabled"
        except RuntimeError:
            pass
    finally:
        utils_module.set_lazy_audio_enabled(True)
