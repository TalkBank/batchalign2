# from .chat import CHATFile
# from .textgrid import TextGridFile

def __getattr__(name):
    if name == 'CHATFile':
        from .chat import CHATFile
        return CHATFile
    if name == 'TextGridFile':
        from .textgrid import TextGridFile
        return TextGridFile
    raise AttributeError(f"module {__name__} has no attribute {name}")
