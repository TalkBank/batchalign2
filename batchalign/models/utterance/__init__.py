# from .infer import BertUtteranceModel
# from .cantonese_infer import BertCantoneseUtteranceModel

def __getattr__(name):
    if name == 'BertUtteranceModel':
        from .infer import BertUtteranceModel
        return BertUtteranceModel
    if name == 'BertCantoneseUtteranceModel':
        from .cantonese_infer import BertCantoneseUtteranceModel
        return BertCantoneseUtteranceModel
    raise AttributeError(f"module {__name__} has no attribute {name}")


