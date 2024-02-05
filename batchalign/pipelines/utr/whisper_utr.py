from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel

from batchalign.pipelines.utr.utils import bulletize_doc

import warnings 

import pycountry

import logging
L = logging.getLogger("batchalign")

class WhisperUTREngine(BatchalignEngine):
    tasks = [ Task.UTTERANCE_TIMING_RECOVERY ]

    def __init__(self, model=None, lang="eng"):

        if model == None and lang == "eng":
            model = "talkbank/CHATWhisper-en-large-v1"
        elif model == None:
            model = "openai/whisper-large-v2"

        language = pycountry.languages.get(alpha_3=lang).name
            
        self.__whisper = WhisperASRModel(model, language=language)
        self.__lang = lang

    def process(self, doc, **kwargs):
        # check that the document has a media path to align to
        assert doc.media != None and doc.media.url != None, f"We cannot add utterance timings to something that doesn't have a media path! Provided media tier='{doc.media}'"

        # check and if there are existing utterance timings, warn
        if any([i.alignment for i in doc.content if isinstance(i, Utterance)]):
            warnings.warn(f"We found existing utterance timings in the document with {doc.media.url}! Skipping rough utterance alignment.")
            return doc

        L.debug(f"Whisper ASR is loading url {doc.media.url}...")
        res = self.__whisper(self.__whisper.load(doc.media.url).all())

        return bulletize_doc(res, doc)

