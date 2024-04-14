from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel, BertUtteranceModel

import pycountry
import string

import logging
L = logging.getLogger("batchalign")

from batchalign.utils.utils import correct_timing, silence
from batchalign.models import resolve
import warnings

from contextlib import redirect_stdout, redirect_stderr
import io
import os

import gc 
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

        try:
            import whisperx
            print("Batchalign: If you got a warning about pyannote versioning, bad things will not happen. Please disregard.")
        except ImportError:
            raise ImportError("Cannot import WhisperX, please ensure it is installed.\nHint: install WhisperX by running `pip install git+https://github.com/m-bain/whisperx.git`.")


        if lang == "yue":
            language = "yue"
        else:
            language = pycountry.languages.get(alpha_3=lang).alpha_2

        self.__lang = lang
        self.__lang_code = language

        L.info("Loading (and possibly downloading) WhisperX models...")
        self.__model = whisperx.load_model("large-v2", device=DEVICE,
                                        compute_type=("float16"
                                                        if DEVICE == "cuda" else "float32"),
                                        language=language)
        self.__fa, self.__meta = whisperx.load_align_model(device=DEVICE,
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
        audio = whisperx.load_audio(source_path)

        # transcribe and align the audio
        result = self.__model.transcribe(audio, batch_size=8)
        result = whisperx.align(result["segments"], self.__fa,
                                self.__meta, audio,
                                DEVICE, return_char_alignments=False)

        # tally turns together
        turns = []
        current_turn = []

        for segment in result["segments"]:
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


    # model="openai/whisper-large-v2", language="english"

# e = WhisperXEngine()
# tmp = e.generate("./batchalign/tests/pipelines/asr/support/test.mp3")
# tmp[0][0]
# tmp.model_dump()
# file = "./batchalign/tests/pipelines/asr/support/test.mp3"


