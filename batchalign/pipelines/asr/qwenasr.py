"""
qwenasr.py
Support for Qwen3-ASR, a speech recognition model from Alibaba Qwen team.
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import BertUtteranceModel, BertCantoneseUtteranceModel, resolve

import torch
import pycountry
import numpy as np
import soundfile as sf

import logging
L = logging.getLogger("batchalign")


class QwenASREngine(BatchalignEngine):

    @property
    def tasks(self):
        if self.__engine:
            return [Task.ASR, Task.UTTERANCE_SEGMENTATION]
        else:
            return [Task.ASR]

    def __init__(self, model="Qwen/Qwen3-ASR-1.7B", forced_aligner="Qwen/Qwen3-ASR-1.7B", lang="eng"):
        self.__lang = lang
        self.__model_id = model

        # resolve language name for Qwen (expects full name like "English", "Chinese", "Cantonese")
        if lang == "mys":
            self.__language = "Malay"
        else:
            self.__language = pycountry.languages.get(alpha_3=lang).name
        if self.__language == "Yue Chinese":
            self.__language = "Cantonese"
        if "greek" in self.__language.lower():
            self.__language = "Greek"

        # pick device: prefer CUDA, then MPS, fallback CPU
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "cpu"  # MPS often unsupported for new architectures
            dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32

        L.info(f"QwenASR: loading {model} on {device} ({dtype})")

        from qwen_asr import Qwen3ASRModel
        self.__model = Qwen3ASRModel.from_pretrained(
            model,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs={"torch_dtype": dtype, "device_map": device},
            torch_dtype=dtype,
            device_map=device,
            max_inference_batch_size=32,
            max_new_tokens=512,
        )

        # utterance segmentation model
        if resolve("utterance", self.__lang) is not None:
            L.debug("Initializing utterance model...")
            if lang != "yue":
                self.__engine = BertUtteranceModel(resolve("utterance", lang))
            else:
                self.__engine = BertCantoneseUtteranceModel(resolve("utterance", lang))
            L.debug("Done.")
        else:
            self.__engine = None

    def generate(self, audio_file_path, **kwargs):
        """Transcribe an audio file using Qwen3-ASR.

        Parameters
        ----------
        audio_file_path : str
            Path to the audio file.

        Returns
        -------
        Document
        """

        results = self.__model.transcribe(
            audio=audio_file_path,
            language=self.__language,
            return_time_stamps=True,
        )

        turns = []

        for result in results:
            text = result.text
            if not text or not text.strip():
                continue

            turn = []

            # if timestamps are available, use them
            if result.time_stamps is not None and hasattr(result.time_stamps, "items"):
                for item in result.time_stamps.items:
                    turn.append({
                        "type": "text",
                        "ts": item.start_time,
                        "end_ts": item.end_time,
                        "value": item.text,
                    })
            else:
                # no timestamps: emit words without timing
                words = list(text) if self.__lang in ("yue", "zho", "cmn") else text.split()
                for word in words:
                    turn.append({
                        "type": "text",
                        "ts": None,
                        "end_ts": None,
                        "value": word,
                    })

            if turn:
                turns.append({
                    "elements": turn,
                    "speaker": 0,
                })

        L.debug("QwenASR transcription done.")

        doc = process_generation(
            {"monologues": turns},
            self.__lang,
            utterance_engine=self.__engine,
        )
        media = Media(type=MediaType.AUDIO, name=Path(audio_file_path).stem, url=audio_file_path)
        doc.media = media
        return doc
