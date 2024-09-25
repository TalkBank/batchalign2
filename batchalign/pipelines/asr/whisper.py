from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel, BertUtteranceModel

import pycountry

import logging
L = logging.getLogger("batchalign")

from batchalign.utils.utils import correct_timing
from batchalign.models import resolve


class WhisperEngine(BatchalignEngine):

    @property
    def tasks(self):
        # if there is no utterance segmentation scheme, we only
        # run ASR
        if self.__engine:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR ]

    def __init__(self, lang="eng"):
        # try to resolve our internal model
        res = resolve("whisper", lang)
        if res:
            _, base = res
        else:
            # Set default model and base based on language
            if lang == "yue":
                # model = "alvanlii/whisper-small-cantonese"
                base = "alvanlii/whisper-small-cantonese"
            else:
                # model = "openai/whisper-large-v3"
                base = "openai/whisper-large-v3"

        language = pycountry.languages.get(alpha_3=lang).name
        if language == "Yue Chinese":
            language = "Cantonese"
        if "greek" in language.lower():
            language = "Greek"  
        
        self.__whisper = WhisperASRModel(base=base, language=language)
        self.__lang = lang

        if resolve("utterance", self.__lang) is not None:
            L.debug("Initializing utterance model...")
            self.__engine = BertUtteranceModel(resolve("utterance", self.__lang))
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


