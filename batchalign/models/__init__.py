# from .utterance import BertUtteranceModel, BertCantoneseUtteranceModel
# from .whisper import WhisperASRModel, WhisperFAModel
# from .speaker import NemoSpeakerModel
# from .utils import ASRAudioFile
from .resolve import resolve
# from .wave2vec import Wave2VecFAModel

def __getattr__(name):
    if name == 'BertUtteranceModel':
        from .utterance import BertUtteranceModel
        return BertUtteranceModel
    if name == 'BertCantoneseUtteranceModel':
        from .utterance import BertCantoneseUtteranceModel
        return BertCantoneseUtteranceModel
    if name == 'WhisperASRModel':
        from .whisper import WhisperASRModel
        return WhisperASRModel
    if name == 'WhisperFAModel':
        from .whisper import WhisperFAModel
        return WhisperFAModel
    if name == 'NemoSpeakerModel':
        from .speaker import NemoSpeakerModel
        return NemoSpeakerModel
    if name == 'ASRAudioFile':
        from .utils import ASRAudioFile
        return ASRAudioFile
    if name == 'resolve':
        from .resolve import resolve
        return resolve
    if name == 'Wave2VecFAModel':
        from .wave2vec import Wave2VecFAModel
        return Wave2VecFAModel
    raise AttributeError(f"module {__name__} has no attribute {name}")
