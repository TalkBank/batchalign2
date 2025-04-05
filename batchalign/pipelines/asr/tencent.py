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
import pycountry

import base64
from tencentcloud.common.credential import Credential
from tencentcloud.asr.v20190614.asr_client import AsrClient, models

import logging
L = logging.getLogger("batchalign")

class TencentEngine(BatchalignEngine):

    @property
    def tasks(self):
        # if there is no utterance segmentation scheme, we only
        # run ASR
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
                raise ConfigError("No Tencent Cloud key found. Tencent Cloud was not set up! Please write one yourself and place it at `~/.batchalign.ini`.")

        self.__lang_code = lang
        self.__num_speakers = num_speakers

        if lang == "yue":
            self.__lang = "yue"
        else:
            self.__lang = pycountry.languages.get(alpha_3=lang).alpha_2

        cred = Credential(id,key)
        self.__client = AsrClient(cred, "ap-hongkong")

        if resolve("utterance", lang) != None:
            L.debug("Initializing utterance model...")
            if lang != "yue":
                self.__engine = BertUtteranceModel(resolve("utterance", lang))
            else:
                # we have special inference procedure for cantonese
                self.__engine = BertCantoneseUtteranceModel(resolve("utterance", lang))
            L.debug("Done.")
        else:
            self.__engine = None


    def generate(self, f, **kwargs):
        # bring language code into the stack to access
        lang = self.__lang
        client = self.__client

        L.info(f"Uploading '{pathlib.Path(f).stem}'...")
        # we will send the file for processing
        if not str(f).startswith("http"):
            with open(f, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

        req = models.CreateRecTaskRequest()
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
        while res.Data.Status not in [2,3]:
            time.sleep(15)
            res = client.DescribeTaskStatus(req)

        # if failed, raise
        if res.Data.Status == "3" or res.Data.Status == 3:
            raise RuntimeError(f"Tencent reports job failed! error='{res.Data.ErrorMsg}'")

        turns = []
        for i in res.Data.ResultDetail:
            turn = []
            start = i.StartMs
            for j in i.Words:
                turn.append({
                    "type": "text",
                    "ts": (j.OffsetStartMs+start)/1000,
                    "end_ts": (j.OffsetEndMs+start)/1000,
                    "value": cc.convert(j.Word)
                })
            turns.append({
                "elements": turn,
                "speaker": i.SpeakerId
            })
        L.debug(f"Tencent done.")

        # postprocess the output and define media tier
        doc = process_generation({"monologues": turns},
                                 self.__lang_code, utterance_engine=self.__engine)
        media = Media(type=MediaType.AUDIO, name=Path(f).stem, url=f)
        doc.media = media
        return doc
