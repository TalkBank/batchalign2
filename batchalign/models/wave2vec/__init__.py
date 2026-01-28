# from .infer_fa import Wave2VecFAModel

def __getattr__(name):
    if name == 'Wave2VecFAModel':
        from .infer_fa import Wave2VecFAModel
        return Wave2VecFAModel
    raise AttributeError(f"module {__name__} has no attribute {name}")
