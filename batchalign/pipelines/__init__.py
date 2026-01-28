# from .pipeline import BatchalignPipeline
# from .base import BatchalignEngine
# from .asr import WhisperEngine, RevEngine, WhisperXEngine, OAIWhisperEngine

# from .morphosyntax import StanzaEngine, CorefEngine
# from .cleanup import NgramRetraceEngine, DisfluencyReplacementEngine
# from .speaker import NemoSpeakerEngine

# from .fa import WhisperFAEngine, Wave2VecFAEngine
# from .utr import WhisperUTREngine, RevUTREngine

# from .analysis import EvaluationEngine
# from .utterance import StanzaUtteranceEngine

# from .translate import SeamlessTranslationModel, GoogleTranslateEngine
# from .avqi import AVQIEngine

# from .diarization import PyannoteEngine
# from .opensmile import OpenSMILEEngine

def __getattr__(name):
    if name == 'BatchalignPipeline':
        from .pipeline import BatchalignPipeline
        return BatchalignPipeline
    if name == 'BatchalignEngine':
        from .base import BatchalignEngine
        return BatchalignEngine
    
    # Add common engines for those still using from batchalign.pipelines import ...
    if name == 'StanzaEngine':
        from .morphosyntax import StanzaEngine
        return StanzaEngine
    if name == 'StanzaUtteranceEngine':
        from .utterance import StanzaUtteranceEngine
        return StanzaUtteranceEngine

    raise AttributeError(f"module {__name__} has no attribute {name}")
