from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel, BertUtteranceModel

import pycountry

import logging
L = logging.getLogger("batchalign")

from batchalign.utils.utils import correct_timing



POSTPROCESSOR_LANGS = {'eng': "talkbank/CHATUtterance-en"}

class WhisperEngine(BatchalignEngine):
    tasks = [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]

    def __init__(self, model=None, lang="eng"):

        if model == None and lang == "eng" or lang == "jpn":
            model = "talkbank/CHATWhisper-en-large-v1"
        elif model == None:
            model = "openai/whisper-large-v2"

        language = pycountry.languages.get(alpha_3=lang).name
            
        self.__whisper = WhisperASRModel(model, language=language)
        self.__lang = lang

        if POSTPROCESSOR_LANGS.get(self.__lang) != None:
            L.debug("Initializing utterance model...")
            self.__engine = BertUtteranceModel(POSTPROCESSOR_LANGS.get(self.__lang))
            L.debug("Done.")
        else:
            self.__engine = None

    def generate(self, source_path, **kwargs):
        res = self.__whisper(self.__whisper.load(source_path).all())
        doc = process_generation(res, self.__lang, utterance_engine=self.__engine)

        # define media tier
        media = Media(type=MediaType.AUDIO, name=Path(source_path).stem, url=source_path)
        doc.media = media

        return correct_timing(doc)


    # model="openai/whisper-large-v2", language="english"

# e = WhisperEngine()
# tmp = e.generate("./batchalign/tests/pipelines/asr/support/test.mp3", 1)
# tmp.model_dump()
# file = "./batchalign/tests/pipelines/asr/support/test.mp3"


