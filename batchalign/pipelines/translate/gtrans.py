from batchalign.models import WhisperFAModel
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils import *
from batchalign.utils.dp import *
from batchalign.constants import *
from batchalign.pipelines.translate.utils import run_coroutine_sync

from googletrans import Translator

import logging
L = logging.getLogger("batchalign")

import re

# !uv pip install sentencepiece

import pycountry
import warnings
import time

import asyncio

class GoogleTranslateEngine(BatchalignEngine):
    tasks = [ Task.TRANSLATE ]

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def __init__(self):
        self.status_hook = None

    async def translate(self, text):
        translator = Translator()
        return await translator.translate(text)

    def process(self, doc:Document, **kwargs):

        for indx, i in enumerate(doc.content):
            if not isinstance(i, Utterance):
                continue
            if i.translation:
                continue
            
            text = i.strip(join_with_spaces=False, include_retrace=True, include_fp=True)
            if "yue" in doc.langs or "zho" in doc.langs:
                text = text.replace(" ","")
                text = text.replace(".","。")

            translated_text_from_text = run_coroutine_sync(self.translate(text)).text
            translated_text_from_text = translated_text_from_text.replace("。", ".")
            translated_text_from_text = translated_text_from_text.replace("’", "'")
            translated_text_from_text = translated_text_from_text.replace("\t", " ")

            i.translation = translated_text_from_text
            for j in MOR_PUNCT + ENDING_PUNCT:
                i.translation = i.translation.replace(j, " "+j)

            if self.status_hook != None:
                self.status_hook(indx+1, len(doc.content))
            time.sleep(1.5)

        return doc


