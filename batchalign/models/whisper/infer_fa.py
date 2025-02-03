from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration

import torch
from torchaudio import load
from torchaudio import transforms as T
from scipy.ndimage import median_filter
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as dtw
from transformers.models.whisper.generation_whisper import _median_filter as median_filter

from batchalign.models.utils import ASRAudioFile

from batchalign.models.utils import _extract_token_timestamps as ett

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

    def __call__(self, audio, text, pauses=False):
        L.debug("Whisper Preprocessing...")
        # input features
        features = self.__processor(audio=audio, text=(" ".join(list(text)) if pauses else text),
                                    sampling_rate=self.sample_rate,
                                    return_tensors='pt')
        tokens = features["labels"][0]

        L.debug("Running inference...")
        # perform inference to get cached qs
        with torch.inference_mode():
            output = self.__model(**features.to(DEVICE), output_attentions=True)

        L.debug("Collecting and normalizing activations...")
        # get decoder layer across attentions
        # which has shape layers x heads x output_tokens x input_frames
        cross_attentions = torch.cat(output.cross_attentions).cpu()

        # get the attention of alignment heads we care about only
        weights = torch.stack([cross_attentions[l][h]
                            for l, h in self.__model.generation_config.alignment_heads])

        # normalize the attentino activations
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
        weights = (weights - mean) / std

        L.debug("Applying median filter...")

        # perform smoothing on attention activations + scale them
        weights = median_filter(weights, self.__model.config.median_filter_width)
        # average weights across heads
        matrix = weights.mean(axis=0)
        matrix[0] = matrix.mean() # jank way of fixing weird 0th token output
        # see: https://media.discordapp.net/attachments/870073176380563460/1185486042753679390/image.png?ex=658fc8e9&is=657d53e9&hm=28ba60b6035fd8976e44f3e53628558d59d5f578917d36ad22c3d63b5563582e&=&format=webp&quality=lossless&width=1050&height=1164
        # essentially, the 0th token (<sos>) gets attention jammed across the entire rest of the sequence
        # because its padding; which screws everything else up

        L.debug("Applying dynamic time warping...")

        # its dynamic time warping time
        text_idx, time_idx = dtw(-matrix)
        jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
        jump_times = time_idx[jumps] * 0.02
        # align jumps against transcript and decode
        timestamped_tokens = [(self.__processor.decode(i),j) for i,j in zip(tokens, jump_times)]
        # TODO: 50200 is the locations of special tokens

        L.debug("Whisper FA done.")
        # we now return the ruslts for later processing
        return timestamped_tokens

