import os
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.pipelines.utr.utils import bulletize_doc
from batchalign.pipelines.asr.funaudio import FunAudioEngine

from opencc import OpenCC
cc = OpenCC('s2hk')

import warnings 

import pycountry

import logging
L = logging.getLogger("batchalign")

class FunAudioUTREngine(BatchalignEngine):
    tasks = [ Task.UTTERANCE_TIMING_RECOVERY ]

    def __init__(self, model=None, lang="yue"):
        model = "FunAudioLLM/SenseVoiceSmall"

        language = pycountry.languages.get(alpha_3=lang).name
            
        self.__funaudio = FunAudioEngine(model, lang="yue")
        self.__lang = lang

    def process(self, doc, **kwargs):
        # bring language code into the stack to access
        lang = doc.langs[0]

        # check and if there are existing utterance timings, warn
        if any([i.alignment for i in doc.content if isinstance(i, Utterance)]):
            warnings.warn(f"We found existing utterance timings in the document with {doc.media.url}! Skipping rough utterance alignment.")
            return doc

        f = kwargs.get("extra_info", {}).get("extra_input")

        if not f:
            assert doc.media != None and doc.media.url != None, f"We cannot add utterance timings to something that doesn't have a media path! Provided media tier='{doc.media}'"

        f = f if f else doc.media.url
        
        res = self.__funaudio.generate(
            audio_file_path=doc.media.url
        )

        element_lit = res.content
        res_content = []
        for item in element_lit:
            text = item[0]
            text = str(text)
            match = re.search(r"text='(.*?)'.*time=\((\d+),\s*(\d+)\)", text)
            if match:
                text = match.group(1) 
                start_ms = int(match.group(2)) 
                end_ms = int(match.group(3)) 
                print(type(end_ms))
            
            res_content.append({
            "value": text,
            "ts": start_ms / 1000.0,  
            "end_ts": end_ms / 1000.0  
            })

            
            turns = []
            turns.append({
                "elements": res_content,
                "speaker": "unknown"  
            })
            
            element = {"monologues": turns}           
            
        return bulletize_doc(element, doc)
