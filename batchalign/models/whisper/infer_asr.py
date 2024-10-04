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

    def __init__(self, model, base="openai/whisper-large-v3", language="english", target_sample_rate=16000):
        L.debug("Initializing whisper model...")
        self.__config = GenerationConfig.from_pretrained(base)
        self.__config.no_repeat_ngram_size = 4

        if language == "Cantonese":
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                # tokenizer=WhisperTokenizer.from_pretrained(base),
                chunk_length_s=30,
                # stride_length_s=3,
                device=DEVICE,
                # torch_dtype=torch.float32,
                return_timestamps="word",
            )
            self.__config = GenerationConfig.from_model_config(self.pipe.model.config)
            self.__config.no_repeat_ngram_size = 4
            self.__config.use_cache = False

            forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language="yue", task="transcribe")
            
            suppress_tokens = []

            # Define other parameters
            return_attention_mask = False
            pad_token_id = 50257
            bos_token_id = 50257
            eos_token_id = 50257
            decoder_start_token_id = 50258
            begin_suppress_tokens = [
                220,
                50257
            ],
            alignment_heads = [
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
            lang_to_id = {"<|yue|>": 50325}
            task_to_id = {"transcribe": 50359}
            is_multilingual = True
            max_initial_timestamp_index = 50
            no_timestamps_token_id = 50363
            prev_sot_token_id = 50361
            max_length = 448

            # Assign values to generation config
            self.__config.forced_decoder_ids = forced_decoder_ids
            self.__config.suppress_tokens = suppress_tokens
            self.__config.pad_token_id = pad_token_id
            self.__config.bos_token_id = bos_token_id
            self.__config.eos_token_id = eos_token_id
            self.__config.decoder_start_token_id = decoder_start_token_id
            self.__config.lang_to_id = lang_to_id
            self.__config.task_to_id = task_to_id
            self.__config.alignment_heads = alignment_heads
            self.__config.alignment_heads = alignment_heads
            self.__config.begin_suppress_tokens = begin_suppress_tokens
            self.__config.is_multilingual = is_multilingual
            self.__config.max_initial_timestamp_index = max_initial_timestamp_index
            self.__config.no_timestamps_token_id = no_timestamps_token_id
            self.__config.prev_sot_token_id = prev_sot_token_id
            self.__config.max_length =max_length

            self.pipe.model.generation_config = self.__config

        else:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=WhisperTokenizer.from_pretrained(base),
                chunk_length_s=25,
                stride_length_s=3,
                device=DEVICE,
                torch_dtype=torch.float32,
                return_timestamps="word",
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
        config = {
            "repetition_penalty": 1.001,
            "generation_config": self.__config,
            "task": "transcribe",
            "language": self.lang
        }

        if self.lang == "Cantonese":
            config = {
                "repetition_penalty": 1.001,
                # "generation_config": self.__config,
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
                current_speaker = element["payload"],
                current_turn = []

        turns.append({
            "elements": current_turn,
            "speaker": current_speaker[0] if type(current_speaker) == tuple else current_speaker
        })

        L.debug("Whisper Done.")
        return ({"monologues": turns})

