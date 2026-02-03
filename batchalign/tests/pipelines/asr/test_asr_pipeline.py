import pytest

from batchalign.pipelines import BatchalignPipeline
from batchalign.models import WhisperASRModel


DOC = "hello . \x150_880\x15\nthis is a test of Batchline I'm going to read some random crap as I see on the screen . \x15880_5260\x15\njust to test batchline . \x155260_7180\x15\nthe primary area for recording editing and arranging audio are midi and drummer regions divided into different track types . \x157180_13320\x15\npress command slash for more info . \x1513320_15400\x15\ntest [/] test . \x1515400_16380\x15\nI don't know what to say . \x1516380_17460\x15\nbut &-um here's some retracing . \x1517460_19180\x15\nso just for fun . \x1519180_20180\x15\n&-um <I like I like> [/] I like beans . \x1520180_24160\x15\nbeans are fun . \x1524380_25080\x15\nthank you very much . \x1525240_25960\x15"

def test_whisper_asr_pipeline(en_doc, monkeypatch):
    def _fake_call(self, _audio):
        return {
            "monologues": [
                {
                    "speaker": 0,
                    "elements": [
                        {"type": "text", "value": "hello", "ts": 0.0, "end_ts": 0.88},
                        {"type": "punct", "value": ".", "ts": 0.88, "end_ts": 0.88},
                    ]
                }
            ]
        }

    monkeypatch.setattr(WhisperASRModel, "load", lambda self, _path: type("Audio", (), {"all": lambda _self: None})())
    monkeypatch.setattr(WhisperASRModel, "__call__", _fake_call)

    whisper = BatchalignPipeline.new("asr", lang="eng", num_speakers=1, asr="whisper")
    doc = whisper(en_doc)

    assert str(doc).startswith("hello .")
