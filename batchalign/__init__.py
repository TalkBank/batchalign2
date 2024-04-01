import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = str(1)

import logging

# clear all of nemo's loggers
logging.getLogger().handlers.clear()
logging.getLogger('nemo_logger').handlers.clear()
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger('nemo_logger').disabled = True

from .document import *
from .formats import *
from .pipelines import *
from .models import *
from .cli import batchalign as cli
from .constants import *

logging.getLogger('nemo_logger').disabled = False
