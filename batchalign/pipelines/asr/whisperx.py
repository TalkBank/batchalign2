from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel, BertUtteranceModel

import pycountry
import string

import logging
L = logging.getLogger("batchalign")

from batchalign.utils.utils import correct_timing, silence
from batchalign.models.resolve import resolve
import warnings

from contextlib import redirect_stdout, redirect_stderr
import io
import os

import gc

class WhisperXEngine(BatchalignEngine):

    @property
    def tasks(self):
        # if there is no utterance segmentation scheme, we only
        # run ASR
        if self.__engine:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR ]


    def __init__(self, lang="eng"):
        import torch

        try:
            import whisperx
            L.info("Batchalign: If you got a warning about pyannote versioning, bad things will not happen. Please disregard.")
        except ImportError:
            raise ImportError("Cannot import WhisperX, please ensure it is installed.\nHint: install WhisperX by running `pip install git+https://github.com/m-bain/whisperx.git`.")

        from batchalign.utils.device import force_cpu_preferred
        # Determine device at init time to defer torch import
        if force_cpu_preferred():
            self.__device = "cpu"
        else:
            self.__device = "cuda" if torch.cuda.is_available() else "cpu"

        if lang == "yue":
            language = "yue"
        else:
            language = pycountry.languages.get(alpha_3=lang).alpha_2

        self.__lang = lang
        self.__lang_code = language

        L.info("Loading (and possibly downloading) WhisperX models...")
        self.__model = whisperx.load_model("large-v2", device=self.__device,
                                        compute_type=("float16"
                                                        if self.__device == "cuda" else "float32"),
                                        language=language)
        self.__fa, self.__meta = whisperx.load_align_model(device=self.__device,
                                                        language_code=language)
        L.info("Done loading WhisperX models!")

        if resolve("utterance", self.__lang) != None:
            L.debug("Initializing utterance model...")
            self.__engine = BertUtteranceModel(resolve("utterance", self.__lang))
            L.debug("Done.")
        else:
            self.__engine = None

    def generate(self, source_path, **kwargs):

        try:
            import whisperx

        except ImportError:
            raise ImportError("Cannot import WhisperX, please ensure it is installed.\nHint: run `pip install -U batchalign[whisperx]` or install WhisperX via instructions (https://github.com/m-bain/whisperX).")


        # load audio
        audio = None
        segments = None
        try:
            import torchaudio
            import torch

            info = torchaudio.info(source_path)
            sample_rate = info.sample_rate
            num_frames = info.num_frames
            chunk_seconds = 60
            chunk_frames = int(chunk_seconds * sample_rate)

            if num_frames <= chunk_frames:
                raise ValueError("File shorter than chunk size; use full load.")

            segments = []
            start_frame = 0
            while start_frame < num_frames:
                end_frame = min(start_frame + chunk_frames, num_frames)
                audio_arr, rate = torchaudio.load(source_path, frame_offset=start_frame, num_frames=end_frame - start_frame)
                if rate != 16000:
                    from torchaudio import transforms as T
                    audio_arr = T.Resample(rate, 16000)(audio_arr)
                    rate = 16000
                audio_chunk = torch.mean(audio_arr, dim=0).float().cpu().numpy()
                chunk_result = self.__model.transcribe(audio_chunk, batch_size=8)
                aligned = whisperx.align(chunk_result["segments"], self.__fa,
                                         self.__meta, audio_chunk,
                                         self.__device, return_char_alignments=False)
                offset = start_frame / sample_rate
                for segment in aligned["segments"]:
                    if segment.get("start") is not None:
                        segment["start"] += offset
                    if segment.get("end") is not None:
                        segment["end"] += offset
                    for word in segment.get("words", []):
                        if word.get("start") is not None:
                            word["start"] += offset
                        if word.get("end") is not None:
                            word["end"] += offset
                    segments.append(segment)
                start_frame = end_frame
        except Exception:
            segments = []

        if len(segments) == 0:
            audio = whisperx.load_audio(source_path)
            result = self.__model.transcribe(audio, batch_size=8)
            result = whisperx.align(result["segments"], self.__fa,
                                    self.__meta, audio,
                                    self.__device, return_char_alignments=False)
            segments = result["segments"]

        # tally turns together
        turns = []
        current_turn = []

        for segment in segments:
            for word in segment["words"]:
                stripped = word["word"].translate(str.maketrans('', '', ",")).replace("...", ".").strip()
                if stripped != "":
                    if word.get("start") != None and word.get("end") != None:
                        text = {
                            "type": "text",
                            "ts": word["start"],
                            "end_ts": word["end"],
                            "value": stripped,
                        }

                        current_turn.append(text)

            turns.append({
                "elements": current_turn,
                "speaker": 0
            })
            current_turn = []

        L.debug("Whisper Done.")
        doc = process_generation(({"monologues": turns}), self.__lang,
                                 utterance_engine=self.__engine)

        # define media tier
        media = Media(type=MediaType.AUDIO, name=Path(source_path).stem, url=source_path)
        doc.media = media

        return correct_timing(doc)
