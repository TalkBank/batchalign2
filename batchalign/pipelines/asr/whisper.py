from torchaudio import transforms as T
from torchaudio import load
from pyAudioAnalysis.audioSegmentation import speaker_diarization
import numpy as np 

from transformers import pipeline

from dataclasses import dataclass

from collections import defaultdict
from pathlib import Path

import torch
from transformers import WhisperProcessor, WhisperTokenizer, GenerationConfig

from nltk import sent_tokenize

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *

import logging
L = logging.getLogger("batchalign")

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
# PYTORCH_ENABLE_MPS_FALLBACK=1
# pretrained model path
# # PRETRAINED = "openai/whisper-small"
# PRETRAINED = "talkbank/CHATWhisper-en-large-v1"
# # PRETRAINED = "openai/whisper-large-v2"
# # FILE = "./data/test.wav"
# FILE = "../talkbank-alignment/testing_playground_2/input/test.wav"
# # FILE = "../talkbank-alignment/broken2/input/53.wav"

@dataclass
class ASRAudioFile:
    file : str
    tensor : torch.Tensor
    rate : int

    def chunk(self,begin_ms, end_ms):
        """Get a chunk of the audio.

        Parameters
        ----------
        begin_ms : int
            Milliseconds of the start of the slice.
        end_ms : int
            Milliseconds of the end of the slice.

        Returns
        -------
        torch.Tensor
            The returned chunk to supply to the ASR engine.
        """

        data = self.tensor[int(round((begin_ms/1000)*self.rate)):
                           int(round((end_ms/1000)*self.rate))]

        return data

    def all(self):
        """Get the audio in its entirety

        Notes
        -----
        like `chunk()` but all of the audio
        """

        return self.tensor

# inference engine
class WhisperPipeline(object):
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

    def __init__(self, model, base="openai/whisper-large-v2", language="english", target_sample_rate=16000):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=WhisperTokenizer.from_pretrained(base),
            chunk_length_s=30,
            stride_length_s=5,
            device=DEVICE,
            return_timestamps="word",
        )
        self.__config = GenerationConfig.from_pretrained(base)
        self.__config.no_repeat_ngram_size = 3
        processor = WhisperProcessor.from_pretrained(base)

        # force decoder IDs to create language
        self.lang = language
        self.__prompt_ids = processor.get_prompt_ids("I um am going to uh pause a uh lot.")

        # save the target sample rate
        self.sample_rate = target_sample_rate

    def load(self, f, num_speakers):
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

        # perform diarization
        if num_speakers == 1:
            dia_cls = None
        else:
            L.info("Whisper Diarizing...")
            dia_cls = speaker_diarization(f, num_speakers, mid_step=0.1, lda_dim=5)[0]

        # and return the audio file
        return ASRAudioFile(f, resampled, self.sample_rate), dia_cls

    def __call__(self, data, segments):
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
                              "repetition_penalty": 1.03,
                              "generation_config": self.__config,
                              "prompt_ids": self.__prompt_ids,
                              "task": "transcribe",
                              "language": self.lang
                          })
                                             # "do_sample": True,
                                             # "temperature": 0.1
                                             # })
                                             # "temperature": 0,
  #"temperature": 0.75,
                                             # })
        # to filter out the two word prompt
        words = words["chunks"][10:]

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
                current_turn.append({
                    "type": "text",
                    "ts": element["start"],
                    "end_ts": element["end"] if element["end"] else element["start"]+1,
                    "value": element["payload"].strip(),
                })
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

class WhisperEngine(BatchalignEngine):
    capabilities = [ BAEngineType.GENERATE ]

    def __init__(self, model=None, language="eng", num_speakers=2):

        if model == None and language == "eng":
            model = "talkbank/CHATWhisper-en-large-v1"
        elif model == None:
            model = "openai/whisper-large-v2"
            
        self.__whisper = WhisperPipeline(model, language=language)
        self.__lang = language
        self.__num_speakers = num_speakers

    def generate(self, source_path):
        audio,segs = self.__whisper.load(source_path, self.__num_speakers)
        res = self.__whisper(audio.all(), segs)
        doc = process_generation(res, self.__lang)

        # define media tier
        media = Media(type=MediaType.AUDIO, name=Path(source_path).stem, url=source_path)
        doc.media = media

        return doc


    # model="openai/whisper-large-v2", language="english"

# e = WhisperEngine()
# tmp = e.generate("./batchalign/tests/pipelines/asr/support/test.mp3", 1)
# tmp.model_dump()
# file = "./batchalign/tests/pipelines/asr/support/test.mp3"


