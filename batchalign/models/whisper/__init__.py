# from .infer_asr import WhisperASRModel
# from .infer_fa import WhisperFAModel

def __getattr__(name):
    if name == 'WhisperASRModel':
        from .infer_asr import WhisperASRModel
        return WhisperASRModel
    if name == 'WhisperFAModel':
        from .infer_fa import WhisperFAModel
        return WhisperFAModel
    raise AttributeError(f"module {__name__} has no attribute {name}")
