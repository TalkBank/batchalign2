# from .infer import NemoSpeakerModel

def __getattr__(name):
    if name == 'NemoSpeakerModel':
        from .infer import NemoSpeakerModel
        return NemoSpeakerModel
    raise AttributeError(f"module {__name__} has no attribute {name}")
