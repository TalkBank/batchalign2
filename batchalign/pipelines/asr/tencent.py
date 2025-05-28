"""
rev.py
Support for Rev.ai, a commerical ASR service
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.utils.config import config_read

from batchalign.errors import *

from batchalign.models import BertUtteranceModel, BertCantoneseUtteranceModel, resolve

from opencc import OpenCC
cc = OpenCC('s2hk')

import time
import pathlib
import tempfile
import pycountry
import numpy as np
import soundfile as sf
# from pydub import AudioSegment
# from pydub.effects import normalize
import base64
from tencentcloud.common.credential import Credential
from tencentcloud.asr.v20190614.asr_client import AsrClient, models

import asyncio
import tempfile
import os
# from pydub import AudioSegment
# from pydub.effects import normalize
# from pydub.exceptions import CouldntDecodeError


import logging
L = logging.getLogger("batchalign")

class TencentEngine(BatchalignEngine):

    @property
    def tasks(self):
        if self.__engine:
            return [ Task.ASR, Task.SPEAKER_RECOGNITION, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR, Task.SPEAKER_RECOGNITION ]

    def __init__(self, key:str=None, lang="eng", num_speakers=2):

        if key == None or key.strip() == "":
            config = config_read()
            try:
                id = config["asr"]["engine.tencent.id"] 
                key = config["asr"]["engine.tencent.key"] 
            except KeyError:
                raise ConfigError("No Tencent Cloud key found. Tencent Cloud was not set up! Please write one yourself and place it at ~/.batchalign.ini.")

        self.__lang_code = lang
        self.__num_speakers = num_speakers

        if lang == "yue":
            self.__lang = "yue"
        else:
            self.__lang = pycountry.languages.get(alpha_3=lang).alpha_2

        cred = Credential(id, key)
        self.__client = AsrClient(cred, "ap-hongkong")

        if resolve("utterance", lang) != None:
            L.debug("Initializing utterance model...")
            if lang != "yue":
                self.__engine = BertUtteranceModel(resolve("utterance", lang))
            else:
                self.__engine = BertCantoneseUtteranceModel(resolve("utterance", lang))
            L.debug("Done.")
        else:
            self.__engine = None
        
    def replace_cantonese_words(self, word):
        """Function to replace Cantonese words with custom replacements."""
        word_replacements = {
            "系": "係", 
            "唔系": "唔係",
            "噶": "㗎",
            "咧": "呢",
            "嗬": "喎",
            "只": "隻",
            "咯": "囉",
            "嚇": "吓",
            "飲": "飲",
            "喐": "郁",
            "食": "食",
            "啫": "咋",
            "哇": "嘩",
            "着": "著",
            "中意": "鍾意",
            "嘞": "喇",
            "啵": "噃",
            "遊水": "游水",
            "羣組": "群組",
            "古仔": "故仔",
            "甕": "㧬",
            "牀": "床",
            "松": "鬆",
            "較剪": "鉸剪",
            "吵": "嘈",
            "衝涼": "沖涼",
            "分鍾": "分鐘",
            "重復": "重複"
        }
        return word_replacements.get(word, word)

    @staticmethod
    def is_roman(x):
        """check if x contains only roman characters"""
        return all(c.isalpha() and ord(c) < 128 for c in x if not c.isspace())
    
    def generate(self, f, **kwargs):
        lang = self.__lang
        client = self.__client

        # processed_path = self.__preprocess_audio(f)
        # audio = AudioSegment.from_file(processed_path)
        
        L.info(f"Uploading '{pathlib.Path(f).stem}'...")
        # we will send the file for processing
        if not str(f).startswith("http"):
            with open(f, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

        req = models.CreateRecTaskRequest()
        if lang in {'zho', 'yue', 'wuu', 'nan','hak'}:
            req.EngineModelType = "16k_zh_large"
        else:
            req.EngineModelType = f"16k_{lang}"
        req.ResTextFormat = 1
        req.SpeakerDiarization = 1
        req.ChannelNum = 1
        if not str(f).startswith("http"):
            req.Data = encoded_string.decode('ascii')
            req.SourceType = 1
        else:
            req.Url = f
            req.SourceType = 0
        resp = client.CreateRecTask(req)

        L.info(f"Tencent is transcribing '{pathlib.Path(f).stem}'...")
        req = models.DescribeTaskStatusRequest()
        req.TaskId = resp.Data.TaskId

        res = client.DescribeTaskStatus(req)
        while res.Data.Status not in [2, 3]:
            time.sleep(15)
            res = client.DescribeTaskStatus(req)

        if res.Data.Status in ["3", 3]:
            raise RuntimeError(f"Tencent reports job failed! error='{res.Data.ErrorMsg}'")

        turns = []
        for i in res.Data.ResultDetail:
            turn = []
            start = i.StartMs
            roman_cache = ""
            roman_cache_start = i.StartMs
            roman_cache_end = i.StartMs
            for j in i.Words:
                word = j.Word
                if self.__lang == "yue":
                    word = cc.convert(word)

                    word = self.replace_cantonese_words(word)

                if self.is_roman(word):
                    if roman_cache == "":
                        roman_cache_start = (j.OffsetStartMs + start)
                    roman_cache = roman_cache + word
                    roman_cache_end = (j.OffsetEndMs + start)
                else:
                    if roman_cache != "":
                        turn.append({
                            "type": "text",
                            "ts": roman_cache_start / 1000,
                            "end_ts": roman_cache_end / 1000,
                            "value": roman_cache
                        })
                    roman_cache = ""
                    turn.append({
                        "type": "text",
                        "ts": (j.OffsetStartMs + start) / 1000,
                        "end_ts": (j.OffsetEndMs + start) / 1000,
                        "value": word
                    })

            if roman_cache != "":
                turn.append({
                    "type": "text",
                    "ts": roman_cache_start / 1000,
                    "end_ts": roman_cache_end / 1000,
                    "value": roman_cache
                })

            turns.append({
                "elements": turn,
                "speaker": i.SpeakerId
            })
        L.debug(f"Tencent done.")

        # Extract the text from the small volume parts for translation

        doc = process_generation({"monologues": turns},
                                self.__lang_code, 
                                utterance_engine=self.__engine)
        media = Media(type=MediaType.AUDIO, name=Path(f).stem, url=f)
        doc.media = media
        return doc

