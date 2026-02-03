import numpy as np 
import os

import re

from dataclasses import dataclass

from collections import defaultdict
from pathlib import Path

# Heavy imports removed from top-level

from batchalign.document import *

from batchalign.models.utils import ASRAudioFile

import pycountry 

import logging
L = logging.getLogger("batchalign")

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

    def __init__(self, model, base="openai/whisper-large-v3", language="english", target_sample_rate=16000):
        import torch
        from transformers import pipeline, WhisperProcessor, WhisperTokenizer, GenerationConfig, WhisperForConditionalGeneration
        from batchalign.models.utils import _extract_token_timestamps as ett
        
        # Monkey patch
        WhisperForConditionalGeneration._extract_token_timestamps = ett

        from batchalign.utils.device import force_cpu_preferred
        if force_cpu_preferred():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

        L.debug("Initializing whisper model...")
        self.__config = GenerationConfig.from_pretrained(base)
        self.__config.no_repeat_ngram_size = 4
        self.__config.use_cache = True
        
        if language == "Cantonese":
            self.__config.no_repeat_ngram_size = 4
            self.__config.no_timestamps_token_id = 50363
            self.__config.alignment_heads = [
                [5, 3],
                [5, 9],
                [8, 0],
                [8, 4],
                [8, 8],
                [9, 0],
                [9, 7],
                [9, 9],
                [10, 5]
            ]

        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=WhisperTokenizer.from_pretrained(base),
                chunk_length_s=25,
                stride_length_s=3,
                device=device,
                torch_dtype=torch.bfloat16,
                return_timestamps=True,
            )
        except TypeError:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=WhisperTokenizer.from_pretrained(base),
                chunk_length_s=25,
                stride_length_s=3,
                device=device,
                torch_dtype=torch.float16,
                return_timestamps=True,
            )
        L.debug("Done, initalizing processor and config...")
        processor = WhisperProcessor.from_pretrained(base)
        L.debug("Whisper initialization done.")

        # force decoder IDs to create language
        self.lang = language

        # save the target sample rate
        self.sample_rate = target_sample_rate

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
        import torch
        from batchalign.models import audio_io
        from torchaudio import transforms as T

        # function: load and resample audio (lazy by default)
        try:
            info = audio_io.info(f)
            sample_rate = info.sample_rate
            lazy_audio = ASRAudioFile.lazy(f, sample_rate)
        except Exception:
            audio_arr, rate = audio_io.load(f)
            if rate != self.sample_rate:
                audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)
            resampled = torch.mean(audio_arr.transpose(0,1), dim=1)
            return ASRAudioFile(f, resampled, self.sample_rate)

        if sample_rate != self.sample_rate:
            # Force eager load to resample once if sample rate differs
            audio_arr, rate = audio_io.load(f)
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)
            resampled = torch.mean(audio_arr.transpose(0,1), dim=1)
            return ASRAudioFile(f, resampled, self.sample_rate)

        return lazy_audio

    def __call__(self, data, segments=None):
        if isinstance(data, ASRAudioFile):
            data = data.all()
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
        config = {
            "repetition_penalty": 1.001,
            "generation_config": self.__config,
            "task": "transcribe",
            "language": self.lang
        }

        if self.lang == "Cantonese":
            config = {
                "repetition_penalty": 1.001,
                "generation_config": self.__config,
                # "task": "transcribe",
                # "language": self.lang
            }

        words = self.pipe(data.cpu().numpy(),
                          batch_size=1, 
                          generate_kwargs=config)

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
            timestamp = word.get("timestamp")
            if not timestamp or len(timestamp) < 2:
                continue
            start, end = timestamp
            if start is None:
                continue
            if end is None:
                end = start + 1
            groups.append({
                "type": "text",
                "start": start,
                "end": end,
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
                pl = pl.replace("「", "")
                pl = pl.replace("」", "")
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
                current_speaker = element["payload"]
                current_turn = []

        turns.append({
            "elements": current_turn,
            "speaker": current_speaker[0] if type(current_speaker) == tuple else current_speaker
        })

        L.debug("Whisper Done.")
        return ({"monologues": turns})
