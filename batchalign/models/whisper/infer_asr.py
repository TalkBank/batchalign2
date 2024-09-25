from torchaudio import transforms as T
from torchaudio import load
import numpy as np 
import os

import re
from transformers import pipeline

from dataclasses import dataclass

from collections import defaultdict
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperTokenizer, GenerationConfig, WhisperForConditionalGeneration



from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *

from batchalign.pipelines.asr.utils import *

from batchalign.models.utils import _extract_token_timestamps as ett
from batchalign.models.utils import ASRAudioFile


WhisperForConditionalGeneration._extract_token_timestamps = ett

import pycountry 

import logging
L = logging.getLogger("batchalign")

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
# PYTORCH_ENABLE_MPS_FALLBACK=1
# pretrained model path
# # PRETRAINED = "openai/whisper-small"
# PRETRAINED = "talkbank/CHATWhisper-en-large-v1"
# # PRETRAINED = "openai/whisper-large-v2"
# # FILE = "./data/test.wav"
# FILE = "../talkbank-alignment/testing_playground_2/input/test.wav"
# # FILE = "../talkbank-alignment/broken2/input/53.wav"

# inference engine
class WhisperASRModel(object):
    """An ASR Engine

    Parameters
    ----------
    model : str
        The model path to load from.
    target_sample_rate : optional, int
        The sample rate to cast to. Defaults 16000 by Whisper.

    Example
    -------
    >>> engine = ASREngine("./model/my_model")
    >>> file = engine.load("./data/myfile.wav")
    >>> engine(file.chunk(7000, 13000)) # transcribes 7000th ms to 13000th ms
    """

    def __init__(self, base="alvanlii/whisper-small-cantonese", language="english", device=0):
        L.debug("Initializing whisper model...")
        if language == "yue":
            self.model_name = "alvanlii/whisper-small-cantonese"
            lang = "yue"
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
                chunk_length_s=30,
                device=device,
            )
            self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        else:
            self.model_name = "openai/whisper-large-v3"
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.model_name,
                tokenizer=WhisperTokenizer.from_pretrained(self.model_name),
                chunk_length_s=25,
                stride_length_s=3,
                torch_dtype=torch.float32,
                device=device,
                return_timestamps="word",
            )
        
        L.debug(f"Using model: {self.model_name}")
        L.debug("Whisper initialization done.")

    def transcribe(self, file):
        return self.pipe(file)["text"]

    def load(self, f):
        """Load an audio file for procesing.

        Parameters
        ----------
        f : str
            The audio .wav file name to process.
        num_speakers : int
            The number of speakers

        Returns
        -------
        Tuple[ASRAudioFile, List[dict]]
            Return processed audio file and speaker segments.
        """

        # function: load and resample audio
        audio_arr, rate = load(f)

        # resample if needed
        if rate != self.sample_rate:
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)

        # transpose and mean
        resampled = torch.mean(audio_arr.transpose(0,1), dim=1)

        # and return the audio file
        return ASRAudioFile(f, resampled, self.sample_rate)

    def __call__(self, data, segments=None):
        # we now perform the sweep line algorithm to align the
        # segment timestamps against the words
        groups = []

        L.info(f"Whisper transcribing file...")
        L.debug("Whisper Preprocessing...")
        if segments is not None:
            secs = np.array(range(len(segments))) * 0.5 + 0.1 / 2.0
            cur_start = 0
            cur_spk = segments[0]

            for indx, i in zip(secs, segments):
                if i != cur_spk:
                    # results is by 0.1 second steps
                    groups.append({
                        "type": "segment",
                        "start": cur_start/10,
                        "end": indx/10,
                        "payload": int(cur_spk)
                    })
                    cur_start = indx
                    cur_spk = i
        else:
            groups.append({
                "type": "segment",
                "start": 0,
                "end": len(data)/self.sample_rate,
                "payload": 0
            })

        L.debug("Whisper Transcribing...")
        words = self.pipe(data.cpu().numpy(),
                          batch_size=1, 
                          generate_kwargs = {
                              "repetition_penalty": 1.001,
                              "generation_config": self.__config,
                              "task": "transcribe",
                              "language": self.lang
                          })
                                             # "do_sample": True,
                                             # "temperature": 0.1
                                             # })
                                             # "temperature": 0,
  #"temperature": 0.75,
                                             # })
        # to filter out the one word prompt
        words = words["chunks"]

        # filter out the elements in the prompt, which has timestamp (0,0)
        # words = list(filter(lambda x:x["timestamp"] != (0.0, 0.0), words))


        L.debug("Whisper Postprocessing...")
        for word in words:
            groups.append({
                "type": "text",
                "start": word["timestamp"][0],
                "end": word["timestamp"][1],
                "payload": word["text"]
            })

        # sorting the output to perform sweep
        groups = list(sorted(groups, key=lambda x:x["start"]))

        # tally turns together
        turns = []
        current_speaker = 0
        current_turn = []

        current_segment = groups.pop(0)
        while len(groups) > 0:
            element = groups.pop(0)

            if element["type"] == "text":
                pl = element["payload"].strip()
                before = re.findall(r"^\W+", pl)
                after = re.findall(r"\W+$", pl)
                texts = []
                if len(before) > 0:
                    texts.append({
                        "type": "punct",
                        "ts": element["start"],
                        "end_ts": element["end"] if element["end"] else element["start"]+1,
                        "value": before[0],
                    })
                    pl = pl.strip(before[0])
                if len(after) > 0:
                    pl = pl.strip(after[0])
                texts.append({
                    "type": "text",
                    "ts": element["start"],
                    "end_ts": element["end"] if element["end"] else element["start"]+1,
                    "value": pl.strip(),
                })
                if len(after) > 0:
                    texts.append({
                        "type": "punct",
                        "ts": element["start"],
                        "end_ts": element["end"] if element["end"] else element["start"]+1,
                        "value": after[0],
                    })

                for text in texts:
                    if text["ts"] != text["end_ts"] and text["value"].strip() != "…" and text["value"].strip() != "":
                        # text with no DTW time is likely a spurious retrace
                        current_turn.append(text)
            elif element["type"] == "segment" and current_speaker != element["payload"]:
                turns.append({
                    "elements": current_turn,
                    "speaker": current_speaker[0] if type(current_speaker) == tuple else current_speaker
                })
                current_speaker = element["payload"],
                current_turn = []

        turns.append({
            "elements": current_turn,
            "speaker": current_speaker[0] if type(current_speaker) == tuple else current_speaker
        })

        L.debug("Whisper Done.")
        return ({"monologues": turns})

