# inference engine
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

from batchalign.models.utils import _extract_token_timestamps as ett
from batchalign.models.utils import ASRAudioFile

WhisperForConditionalGeneration._extract_token_timestamps = ett

import pycountry

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

import logging
L = logging.getLogger("batchalign")

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
            self.processor = WhisperProcessor.from_pretrained(model)
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                # tokenizer=WhisperTokenizer.from_pretrained(base),
                chunk_length_s=30,
                stride_length_s=3,
                device=DEVICE,
                # torch_dtype=torch.float32,
                feature_extractor=self.processor.feature_extractor,
                return_timestamps="word",
            )
            self.__config = GenerationConfig.from_model_config(self.pipe.model.config)
            self.__config.no_repeat_ngram_size = 4
            self.__config.use_cache = False

            forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language="yue", task="transcribe")

            suppress_tokens = []

            return_attention_mask = False
            pad_token_id = 50257
            bos_token_id = 50257
            eos_token_id = 50257
            decoder_start_token_id = 50258
            begin_suppress_tokens = [
                220,
                50257
            ]
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

            self.__config.forced_decoder_ids = forced_decoder_ids
            self.__config.suppress_tokens = suppress_tokens
            self.__config.pad_token_id = pad_token_id
            self.__config.bos_token_id = bos_token_id
            self.__config.eos_token_id = eos_token_id
            self.__config.decoder_start_token_id = decoder_start_token_id
            self.__config.lang_to_id = lang_to_id
            self.__config.task_to_id = task_to_id
            self.__config.alignment_heads = alignment_heads
            self.__config.begin_suppress_tokens = begin_suppress_tokens
            self.__config.is_multilingual = is_multilingual
            self.__config.max_initial_timestamp_index = max_initial_timestamp_index
            self.__config.no_timestamps_token_id = no_timestamps_token_id
            self.__config.prev_sot_token_id = prev_sot_token_id

            self.pipe.model.generation_config = self.__config
            processor = WhisperProcessor.from_pretrained(base)

            self.lang = language
            self.sample_rate = target_sample_rate

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
        L.debug("Done, initializing processor and config...")
        processor = WhisperProcessor.from_pretrained(base)
        
        L.debug("Whisper initializion done.")

        self.lang = language
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
        
        audio_arr, rate = load(f)
        L.debug(f"Loaded audio shape: {audio_arr.shape}, Sample rate: {rate}")

        if rate != self.sample_rate:
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)
            L.debug(f"Resampled audio shape: {audio_arr.shape}, Target Sample rate: {self.sample_rate}")

        resampled = torch.mean(audio_arr, dim=0) if audio_arr.dim() > 1 else audio_arr
        resampled = resampled.flatten()
        L.debug(f"Flattened audio shape: {resampled.shape}, Min value: {torch.min(resampled)}, Max value: {torch.max(resampled)}")
        

        if len(resampled) < self.sample_rate // 2:
            return None

        if torch.all(torch.abs(resampled) < 1e-5):
            return None

        return ASRAudioFile(f, resampled, self.sample_rate)

    def __call__(self, data, segments=None):
        groups = []
        
        L.info("Whisper transcribing file...")

        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError("Input data must be a PyTorch Tensor or a NumPy array.")

        # Log basic information about the data
        min_val = np.min(data)
        max_val = np.max(data)
        L.debug("Whisper Preprocessing...")
        L.debug(f"Data shape before processing: {data.shape}, Min: {min_val}, Max: {max_val}")
    
        total_length_s = len(data) / self.sample_rate
        chunk_length_s = 20
        stride_length_s = 5

        num_chunks = int((total_length_s - chunk_length_s) / (chunk_length_s - stride_length_s) + 1)
        L.info(f"Total number of chunks: {num_chunks}")

        if segments is not None:
            secs = np.array(range(len(segments))) * 0.5 + 0.1 / 2.0
            cur_start = 0
            cur_spk = segments[0]

            for indx, i in zip(secs, segments):
                if i != cur_spk:
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
            "language": self.lang,
            "max_length": 448
        }

        if self.lang == "Cantonese":
            config = {
                "repetition_penalty": 1.001,
                "generation_config": self.__config,
                "task": "transcribe",
                "language": self.lang,
                "max_length": 448
            }

        current_chunk_index = 0
        chunk_size = int(chunk_length_s * self.sample_rate)
        stride_size = int(stride_length_s * self.sample_rate)

        for start_idx in range(0, len(data) - chunk_size + 1, chunk_size - stride_size):
            end_idx = start_idx + chunk_size
            chunk_data = data[start_idx:end_idx]

            current_chunk_index += 1
            L.info(f"Processing chunk {current_chunk_index}/{num_chunks}, Start time: {start_idx/self.sample_rate:.2f}s, End time: {end_idx/self.sample_rate:.2f}s")


            if np.mean(np.abs(chunk_data)) < 1e-5:
                L.warning(f"Skipping chunk {current_chunk_index} due to low amplitude (near-zero signal).")
                continue

            try:
                L.debug(f"Transcribing chunk {current_chunk_index}...")
                words = self.pipe(chunk_data, batch_size=1, generate_kwargs=config)
                L.debug(f"Finished transcribing chunk {current_chunk_index}.")
            except Exception as e:
                L.error(f"Error during transcription of chunk {current_chunk_index}: {e}")
                continue

            if "chunks" in words:
                words = words["chunks"]
            else:
                L.warning(f"No 'chunks' key found in transcription result for chunk {current_chunk_index}. Skipping...")
                continue

            words = list(filter(lambda x: x["timestamp"] != (0.0, 0.0), words))

            transcript_text = " ".join([word["text"] for word in words])

            for word in words:
                group_item = {
                    "type": "text",
                    "start": word["timestamp"][0],
                    "end": word["timestamp"][1],
                    "payload": word["text"]
                }
                groups.append(group_item)

        groups = list(sorted(groups, key=lambda x: x["start"]))
        print(groups)

        turns = []
        current_speaker = 0
        current_turn = []

        if groups:
            current_segment = groups.pop(0)
        else:
            current_segment = None

        while groups:
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
                        current_turn.append(text)
            elif element["type"] == "segment" and current_speaker != element["payload"]:
                turns.append({
                    "elements": current_turn,
                    "speaker": current_speaker[0] if isinstance(current_speaker, tuple) else current_speaker
                })
                current_speaker = element["payload"],
                current_turn = []

        turns.append({
            "elements": current_turn,
            "speaker": current_speaker[0] if isinstance(current_speaker, tuple) else current_speaker
        })
        L.debug("Whisper Done.")
        return {"monologues": turns}
