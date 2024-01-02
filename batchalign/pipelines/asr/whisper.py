from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.models import WhisperASRModel, BertUtteranceModel

import pycountry

import logging
L = logging.getLogger("batchalign")



POSTPROCESSOR_LANGS = {'eng': "talkbank/CHATUtterance-en"}

class WhisperEngine(BatchalignEngine):
    tasks = [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]

    def __init__(self, model=None, lang="eng"):

        if model == None and lang == "eng":
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

    def generate(self, source_path):
        res = self.__whisper(self.__whisper.load(source_path).all())
        doc = process_generation(res, self.__lang, utterance_engine=self.__engine)

        # define media tier
        media = Media(type=MediaType.AUDIO, name=Path(source_path).stem, url=source_path)
        doc.media = media

        # correct the utterance-level timings
        last_end = 0
        for i in doc.content:
            # bump time forward
            if i.alignment:
                time = list(i.alignment)
                if i.alignment[0] < last_end:
                    time[0] = last_end
                last_end = time[1]
                i.time = tuple(time)
                # if the time has been squished to nothing, we clear time
                if i.alignment[1] <= i.alignment[0]:
                    i.time = None
                    for j in i.content:
                        j.time = None
                # otherwise, we remove impossible timestamps
                else:
                    for j in i.content:
                        if j.time and j.time[1] <= j.time[0]:
                            j.time = None
        return doc


    # model="openai/whisper-large-v2", language="english"

# e = WhisperEngine()
# tmp = e.generate("./batchalign/tests/pipelines/asr/support/test.mp3", 1)
# tmp.model_dump()
# file = "./batchalign/tests/pipelines/asr/support/test.mp3"


