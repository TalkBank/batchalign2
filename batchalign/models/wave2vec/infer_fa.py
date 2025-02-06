from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration

import torch
from torchaudio import load
from torchaudio import transforms as T
from batchalign.models.utils import ASRAudioFile

import torchaudio
bundle = torchaudio.pipelines.MMS_FA
import torchaudio.functional as AF

import numpy as np

import logging
L = logging.getLogger("batchalign")

# DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')
TIME_PRECISION = 0.02

# inference engine
class Wave2VecFAModel(object):
    """An Forced Alignment engine built out of whisper

    Parameters
    ----------
    model : str
        The model path to load from.
    target_sample_rate : optional, int
        The sample rate to cast to. Defaults 16000 by Whisper.

    Example
    -------
    >>> engine = Wave2VecFAModel()
    >>> file = engine.load("./data/myfile.wav")
    >>> timestamps = engine(audio=file.chunk(0, 1500), text="this is my transcript") # FA
    """

    def __init__(self, target_sample_rate=16000):
        L.debug("Initializing Wave2vec FA model")
        self.model = bundle.get_model().to(DEVICE)
        L.debug("Wave2Vec FA initialization done.")

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
        """Run forced alignment on the audio file.

        Arguments
        ----------
        audio : tensor
            The audio file to process.
        text : str
            The transcript to align to.

        Returns
        -------
        List[Tuple[str, Tuple[int, int]]]
                A list of speaker segments
        """

        L.debug("Running Wav2Vec word-level forced alignment...")

        # complete the call function, don't write anything else
        L.debug("Running Wav2Vec word-level forced alignment...")

        # Move audio to device and normalize
        audio = audio.to(DEVICE)

        # Get emission matrix from model
        emission, _ = self.model(audio.unsqueeze(0))
        emission = emission.cpu().detach()

        # Get tokens and transcript
        dictionary = bundle.get_dict()

        # Convert text to tokens 
        transcript = torch.tensor([dictionary.get(c, dictionary["*"])
                                   for word in text
                                   for c in word.lower()])

        # Run forced alignment
        path, scores = AF.forced_align(emission, transcript.unsqueeze(0))
        alignments, scores = path[0], scores[0]
        scores = scores.exp()

        # Merge repeated tokens and remove blanks
        path = AF.merge_tokens(alignments, scores)

        def unflatten(list_, lengths):
            assert len(list_) == sum(lengths)
            i = 0
            ret = []
            for l in lengths:
                ret.append(list_[i : i + l])
                i += l
            return ret

        # Unflatten to get character-level alignments
        word_spans = unflatten(path, [len(word) for word in text])
        ratio = audio.size(0)/emission.size(1)
        word_spans = [(int(((spans[0].start*ratio)/self.sample_rate)*1000),
                       int(((spans[-1].end*ratio)/self.sample_rate)*1000)) for spans in word_spans]

        return list(zip(text, word_spans))
