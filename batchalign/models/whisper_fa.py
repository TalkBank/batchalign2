from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration

import torch
from torchaudio import load
from torchaudio import transforms as T
from scipy.ndimage import median_filter
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as dtw
from transformers.models.whisper.generation_whisper import _median_filter as median_filter

from batchalign.models import ASRAudioFile

from batchalign.models.utils import _extract_token_timestamps as ett
from batchalign.models.utils import attn_dynamic_timewarp

WhisperForConditionalGeneration._extract_token_timestamps = ett
import numpy as np

import logging
L = logging.getLogger("batchalign")

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
TIME_PRECISION = 0.02

# inference engine
class WhisperFAModel(object):
    """An Forced Alignment engine built out of whisper

    Parameters
    ----------
    model : str
        The model path to load from.
    target_sample_rate : optional, int
        The sample rate to cast to. Defaults 16000 by Whisper.

    Example
    -------
    >>> engine = WhisperFAModel()
    >>> file = engine.load("./data/myfile.wav")
    >>> timestamps = engine(audio=file.chunk(0, 1500), text="this is my transcript") # FA
    """

    def __init__(self, model="openai/whisper-large-v2", target_sample_rate=16000):
        L.debug("Initializing whisper FA model...")
        self.__model = WhisperForConditionalGeneration.from_pretrained(model, attn_implementation="eager").to(DEVICE)
        self.__model.eval()
        L.debug("Done, initalizing processor and config...")
        self.__processor = WhisperProcessor.from_pretrained(model)
        L.debug("Whisper FA initialization done.")

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

    def __call__(self, audio, text):
        L.debug("Whisper Preprocessing...")
        # input features
        features = self.__processor(audio=audio, text=text,
                                    sampling_rate=self.sample_rate,
                                    return_tensors='pt')
        tokens = features["labels"][0]

        L.debug("Running inference...")
        # perform inference to get cached qs
        with torch.inference_mode():
            output = self.__model(**features.to(DEVICE), output_attentions=True)

        # dtw time!
        jump_times = attn_dynamic_timewarp(output,
                                           self.__model.generation_config.alignment_heads,
                                           self.__model.config.median_filter_width)

        # align jumps against transcript and decode
        timestamped_tokens = [(self.__processor.decode(i),j) for i,j in zip(tokens, jump_times)]
        # TODO: 50200 is the locations of special tokens

        L.debug("Whisper FA done.")
        # we now return the ruslts for later processing
        return timestamped_tokens

