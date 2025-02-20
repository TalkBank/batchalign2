from batchalign.models import WhisperFAModel
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils import *
from batchalign.utils.dp import *
from batchalign.constants import *

from transformers import AutoProcessor, SeamlessM4TModel

import logging
L = logging.getLogger("batchalign")

import re

# !uv pip install sentencepiece

import pycountry
import warnings

class SeamlessTranslationModel(BatchalignEngine):
    tasks = [ Task.TRANSLATE ]

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def __init__(self):
        self.status_hook = None
        self.processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
        self.model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

    def process(self, doc:Document, **kwargs):

        for indx, i in enumerate(doc.content):
            if not isinstance(i, Utterance):
                continue
            if i.translation:
                continue
            
            text = i.strip(join_with_spaces=False, include_retrace=True, include_fp=True)
            text_inputs = self.processor(text=text, src_lang=doc.langs[0] if doc.langs[0] != "zho" else "cmn", return_tensors="pt")
            output_tokens = self.model.generate(**text_inputs, tgt_lang="eng", generate_speech=False)
            translated_text_from_text = self.processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

            i.translation = translated_text_from_text
            for j in MOR_PUNCT + ENDING_PUNCT:
                i.translation = i.translation.replace(j, " "+j)

            if self.status_hook != None:
                self.status_hook(indx+1, len(doc.content))

        return doc


