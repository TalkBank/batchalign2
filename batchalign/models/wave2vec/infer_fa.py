from batchalign.models.utils import ASRAudioFile

import numpy as np

import logging
L = logging.getLogger("batchalign")

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
        import torch
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        from batchalign.utils.device import force_cpu_preferred
        if force_cpu_preferred():
            device = torch.device('cpu')
        else:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu')

        L.debug("Initializing Wave2vec FA model")
        self.model = bundle.get_model().to(device)
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
        import torch
        import torchaudio
        from torchaudio import transforms as T

        # function: load and resample audio (lazy by default)
        try:
            info = torchaudio.info(f)
            sample_rate = info.sample_rate
            lazy_audio = ASRAudioFile.lazy(f, sample_rate)
        except Exception:
            audio_arr, rate = torchaudio.load(f)
            if rate != self.sample_rate:
                audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)
            resampled = torch.mean(audio_arr.transpose(0,1), dim=1)
            return ASRAudioFile(f, resampled, self.sample_rate)

        if sample_rate != self.sample_rate:
            audio_arr, rate = torchaudio.load(f)
            audio_arr = T.Resample(rate, self.sample_rate)(audio_arr)
            resampled = torch.mean(audio_arr.transpose(0,1), dim=1)
            return ASRAudioFile(f, resampled, self.sample_rate)

        return lazy_audio

    def __call__(self, audio, text):
        import torch
        import torchaudio.functional as AF
        from torchaudio.pipelines import MMS_FA as bundle
        
        device = next(self.model.parameters()).device

        L.debug("Running Wav2Vec word-level forced alignment...")

        # Move audio to device and normalize
        audio = audio.to(device)

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
