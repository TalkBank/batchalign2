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

import asyncio
import tempfile
import os
# from pydub import AudioSegment
# from pydub.effects import normalize
# from pydub.exceptions import CouldntDecodeError

import logging
L = logging.getLogger("batchalign")


import batchalign.extern.nls as nls
import threading
import wave
import time

import os
import time
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

class Runner:
    def __init__(self, TOKEN, APPKEY, test_file):
        self.__TOKEN = TOKEN
        self.__APPKEY = APPKEY

        self.__id = -1
        self.__test_file = test_file
        self.__sample_rate = None
        self.__sentences = []
   
    def loadfile(self, filename):
        with wave.open(filename, "rb") as f:
            self.__data = f.readframes(f.getnframes())
            self.__sample_rate = f.getframerate()
    
    def start(self):
        self.loadfile(self.__test_file)
        return self.__run()
        # self.__th.start()
        # self.__th.join()

    def test_on_sentence_begin(self, message, *args):
        ...
        # print("test_on_sentence_begin:{}".format(message))

    def test_on_sentence_end(self, message, *args):
        self.__sentences.append(json.loads(message)["payload"]["words"])

    def test_on_start(self, message, *args):
        ...
        # print("test_on_start:{}".format(message))

    def test_on_error(self, message, *args):
        raise ValueError(message)

    def test_on_close(self, *args):
        ...
        # print("on_close: args=>{}".format(args))

    def test_on_result_chg(self, message, *args):
        ...
        # print("test_on_chg:{}".format(message))

    def test_on_completed(self, message, *args):
        ...
        # print("on_completed:args=>{} message=>{}".format(args, message))
        # 

    def __run(self):
        sr = nls.NlsSpeechTranscriber(
                    url="wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1",
                    token=self.__TOKEN,
                    appkey=self.__APPKEY,
                    on_sentence_begin=self.test_on_sentence_begin,
                    on_sentence_end=self.test_on_sentence_end,
                    on_start=self.test_on_start,
                    on_result_changed=self.test_on_result_chg,
                    on_completed=self.test_on_completed,
                    on_error=self.test_on_error,
                    on_close=self.test_on_close
                )

        # print("{}: session start".format(self.__id))
        r = sr.start(aformat="pcm",
                     enable_intermediate_result=True,
                     enable_punctuation_prediction=False,
                     enable_inverse_text_normalization=False,
                     sample_rate=self.__sample_rate)

        self.__slices = zip(*(iter(self.__data),) * 640)
        for i in self.__slices:
            sr.send_audio(bytes(i))
            time.sleep(0.01)

        sr.ctrl(ex={"test":"tttt"})
        r = sr.stop()

        return self.__sentences

class AliyunEngine(BatchalignEngine):

    @property
    def tasks(self):
        if self.__engine:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]

    def __init__(self, lang="yue", num_speakers=2):

        config = config_read()
        try:
            self.__id = config["asr"]["engine.aliyun.ak_id"] 
            self.__key = config["asr"]["engine.aliyun.ak_secret"] 
            self.__appkey = config["asr"]["engine.aliyun.ak_appkey"] 
        except KeyError:
            raise ConfigError("No Alibaba key found. Aliyun was not set up! Please write one yourself and place it at ~/.batchalign.ini.")

        assert lang == "yue", "Alibaba currently only considers Cantonese due to how its set up, we can enable more in the future!"
        if lang == "yue":
            self.__lang = "yue"
        self.__lang_code = lang


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

        L.info(f"Obtaining session token...")

        client = AcsClient(
            self.__id,
            self.__key,
            "ap-southeast-1"
        )

        request = CommonRequest()
        request.set_method('POST')
        request.set_domain('nlsmeta.ap-southeast-1.aliyuncs.com')
        request.set_version('2019-07-17')
        request.set_action_name('CreateToken')

        response = client.do_action_with_exception(request)

        jss = json.loads(response)
        if 'Token' in jss and 'Id' in jss['Token']:
            token = jss['Token']['Id']
            expireTime = jss['Token']['ExpireTime']

        TOKEN = token
        APPKEY = self.__appkey
        URL = "wss://nls-gateway-ap-southeast-1.aliyuncs.com/ws/v1"

        L.info(f"Uploading and transcribing '{pathlib.Path(f).stem}'...")
        results = Runner(TOKEN, APPKEY, f).start()
        L.info(f"Aliyun Done, postprocessing...")

        turns = []
        for sentence in results:
            turn = []
            for word in sentence:
                turn.append({
                    "type": "text",
                    "ts": word["startTime"]/1000,
                    "end_ts": word["endTime"]/1000,
                    "value": self.replace_cantonese_words(word["text"])
                })
            turns.append({
                "elements": turn,
                "speaker": 0
            })

        # postprocess step
        doc = process_generation({"monologues": turns},
                                 self.__lang_code, 
                                 utterance_engine=self.__engine)
        L.info(f"Postprocessing Done...")
        media = Media(type=MediaType.AUDIO, name=Path(f).stem, url=f)
        doc.media = media
        return doc

