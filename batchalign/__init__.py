import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)

import logging

# clear all of nemo's loggers
logging.getLogger().handlers.clear()
logging.getLogger('nemo_logger').handlers.clear()
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('nemo_logger').disabled = True

from .document import *
from .constants import *
from .errors import *

# Defer slow imports
# from .formats import *
# from .pipelines import *
# from .models import *
# from .cli import batchalign as cli

def __getattr__(name):
    if name == 'cli':
        from .cli import batchalign
        return batchalign
    if name == 'BatchalignPipeline':
        from .pipelines import BatchalignPipeline
        return BatchalignPipeline
    if name == 'CHATFile':
        from .formats.chat import CHATFile
        return CHATFile
    # Add other common engines if needed for dispatch.py
    if name in ['WhisperEngine', 'WhisperFAEngine', 'StanzaEngine', 'RevEngine',
                'NgramRetraceEngine', 'DisfluencyReplacementEngine', 'WhisperUTREngine',
                'RevUTREngine', 'EvaluationEngine', 'WhisperXEngine', 'NemoSpeakerEngine',
                'StanzaUtteranceEngine', 'CorefEngine', 'Wave2VecFAEngine', 'SeamlessTranslationModel',
                'GoogleTranslateEngine', 'OAIWhisperEngine', 'PyannoteEngine']:
        from .pipelines import dispatch
        # This is a bit recursive, let's just let dispatch import them locally
        # which it already does now.
        import importlib
        # We need to find which subpackage it's in. 
        # Actually, if we use local imports in dispatch.py, we don't need these here.
        pass

    raise AttributeError(f"module {__name__} has no attribute {name}")

logging.getLogger('nemo_logger').disabled = False
