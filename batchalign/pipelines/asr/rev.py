"""
rev.py
Support for Rev.ai, a commerical ASR service
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.utils.config import config_read

from batchalign.errors import *

from batchalign.models import BertUtteranceModel

import time
import pathlib
import pycountry

from rev_ai import apiclient, JobStatus

import logging
L = logging.getLogger("batchalign")

POSTPROCESSOR_LANGS = {'en': "talkbank/CHATUtterance-en"}

class RevEngine(BatchalignEngine):
    tasks = [ Task.ASR, Task.SPEAKER_RECOGNITION, Task.UTTERANCE_SEGMENTATION ]

    def __init__(self, key:str=None, lang="eng", num_speakers=2):

        if key == None or key.strip() == "":
            config = config_read()
            try:
                key = config["asr"]["engine.rev.key"] 
            except KeyError:
                raise ConfigError("No Rev.AI key found. Rev.AI was not set up! Please set up Rev.ai through the initial setup process by running 'batchalign setup' in the command line to generate one, or write one yourself and place it at `~/.batchalign.ini`.")

        self.__lang_code = lang
        self.__num_speakers = num_speakers
        self.__lang = pycountry.languages.get(alpha_3=lang).alpha_2
        self.__client = apiclient.RevAiAPIClient(key)
        if POSTPROCESSOR_LANGS.get(self.__lang) != None:
            L.debug("Initializing utterance model...")
            self.__engine = BertUtteranceModel(POSTPROCESSOR_LANGS.get(self.__lang))
            L.debug("Done.")
        else:
            self.__engine = None


    def generate(self, f, **kwargs):
        # bring language code into the stack to access
        lang = self.__lang
        client = self.__client


        L.info(f"Uploading '{pathlib.Path(f).stem}'...")
        # we will send the file for processing
        job = client.submit_job_local_file(f,
                                           metadata=f"batchalign2_{pathlib.Path(f).stem}",
                                           language=lang,
                                           # some languages don't have postprocessors, so this option
                                           # raises an exception
                                           skip_postprocessing=(True if lang in POSTPROCESSOR_LANGS.keys()
                                                                else False),
                                           speakers_count=self.__num_speakers)

        # we will wait untitl job finishes
        status = client.get_job_details(job.id).status
        L.info(f"Rev.AI is transcribing '{pathlib.Path(f).stem}'...")

        # check status, sleeping every so often and check again
        while status == JobStatus.IN_PROGRESS:
            time.sleep(15)
            status = client.get_job_details(job.id).status

        # if we failed, report failure and give up
        if status == JobStatus.FAILED:
            err = client.get_job_details(job.id).failure_detail
            raise RuntimeError(f"Rev.AI reports job failed! file='{pathlib.Path(f).stem}', error='{err}'.")

        # and now, we extract result and postprocess it
        transcript_json = client.get_transcript_json(job.id)
        L.debug(f"Rev.AI done.")

        # postprocess the output and define media tier
        doc = process_generation(transcript_json, self.__lang_code, utterance_engine=self.__engine)
        media = Media(type=MediaType.AUDIO, name=Path(f).stem, url=f)
        doc.media = media
        return doc
