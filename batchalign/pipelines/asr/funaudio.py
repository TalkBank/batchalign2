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
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

import logging
L = logging.getLogger("batchalign")

class FunAudioEngine(BatchalignEngine):

    @property
    def tasks(self):
        if self.__engine:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR ]
        
    def __init__(self, model="FunAudioLLM/SenseVoiceSmall", lang="yue"):

        self.model_dir = model
        self.__lang = "yue"
        
        self.model = AutoModel(
            model=self.model_dir,
            output_timestamps=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",  # GPU
            hub="hf",
            cache={},
            language="yue", 
            use_itn=True,
            batch_size_s=60,
            output_timestamp=True,
            ban_emo_unk =False,
            merge_vad=True,
            merge_length_s=15,
           
        )

        if resolve("utterance", self.__lang) != None:
            L.debug("Initializing utterance model...")
            if lang != "yue":
                self.__engine = BertUtteranceModel(resolve("utterance", lang))
            else:
                # we have special inference procedure for cantonese
                self.__engine = BertCantoneseUtteranceModel(resolve("utterance", lang))
            L.debug("Done.")
        else:
            self.__engine = None
            
    def replace_cantonese_words(self, text):
        """Function to replace Cantonese words with custom replacements."""
        word_replacements = {
            "系": "係", 
            "繫": "係",
            "聯係": "聯繫",
            "系啊": "係啊",
            "真系": "真係",
            "唔系": "唔係",
            "呀": "啊",
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
        sorted_keys = sorted(word_replacements.keys(), key=len, reverse=True)
        pattern = re.compile('|'.join(re.escape(key) for key in sorted_keys))
        
        def replace_word(match):
            matched_text = match.group(0)  # Extract the matched word
            return word_replacements.get(matched_text, matched_text)  # Replace or return the original word

        return pattern.sub(replace_word, text)

    @staticmethod
    def is_roman(x):
        """check if x contains only roman characters"""
        return all(c.isalpha() and ord(c) < 128 for c in x if not c.isspace())
    
    def generate(self, audio_file_path):
        """
        Generate transcription from an audio file using the FunAudio model.
        :param audio_file_path: Path to the audio file to be transcribed.
        :return: A Document object containing the transcription and metadata.
        """
        res = self.model.generate(
            input=audio_file_path,
            cache={},
            language=self.__lang, 
            output_timestamps=True,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000},
            ban_emo_unk=False,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
            output_timestamp=True,
            spk_model="cam++"
        )
        
        turns = []

        for segment in res:  # segment is a dictionary with keys "text" and "timestamp"
            print("segment:", segment)
            print(type(segment))

            # Extracting text and timestamps from the segment
            text = segment["text"]
            print(text)
            timestamps = segment["timestamp"]

            # Check if timestamps is a list of tuples
            utterances = []
            current_utterance = []
            for part in text.split("<|yue|>"):
                if not part.strip():
                    continue
                parts = part.strip().split("<|withitn|>", 1)
                if len(parts) > 1:
                    emotion = parts[0].strip()
                    content = parts[1].strip()
             
                    current_utterance.append(content)
                    print(f"current_utterance:{current_utterance}")

            large_string = ''.join(current_utterance)
            print(f"Large string: {large_string}")
        
            turn = []
                
            # process Cantonese differently
            if self.__lang == "yue":
                content = cc.convert(large_string)
                content = self.replace_cantonese_words(content)
                content = content.replace("「", "").replace("」", "").replace("。", "").replace("，", "").replace("！", "").replace("？", "")
                print(f"Processed Cantonese content: {content}")

               
                items = list(content)  
            else:                    
                items = large_string.split()
                
            turn = []

            num_items = len(items)  
            print("Number of items:", num_items)              
            for index, item in enumerate(items):
                print(f"Processing item {index + 1}/{num_items}: {item}")
                item_start, item_end = timestamps[index]
                

                turn.append({
                    "type": "text",
                    "ts": item_start / 1000,
                    "end_ts": item_end / 1000,
                    "value": item  
                })


            turns.append({
                "elements": turn,
                "speaker": 0
            })

        L.debug(f"Funaudio done.")

        doc = process_generation({"monologues": turns},
                                self.__lang, 
                                utterance_engine=self.__engine)
        media = Media(type=MediaType.AUDIO, name=Path(audio_file_path).stem, url=audio_file_path)
        doc.media = media
        return doc


