"""
rev.py
Support for Rev.ai, a commerical ASR service
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *

from batchalign.models.utterance import BertModel

import time
import pathlib
import pycountry

from rev_ai import apiclient, JobStatus

import logging
L = logging.getLogger("batchalign")

POSTPROCESSOR_LANGS = {'en': "talkbank/CHATUtterance-en"}

class RevEngine(BatchalignEngine):
    capabilities = [ BAEngineType.GENERATE ]

    def __init__(self, key:str, lang_code="eng", num_speakers=2):

        self.__lang_code = lang_code
        self.__num_speakers = num_speakers
        self.__lang = pycountry.languages.get(alpha_3=lang_code).alpha_2
        self.__client = apiclient.RevAiAPIClient(key)
        if POSTPROCESSOR_LANGS.get(self.__lang) != None:
            L.debug("Initializing utterance model...")
            self.__engine = BertModel(POSTPROCESSOR_LANGS.get(self.__lang))
            L.debug("Done.")
        else:
            self.__engine = None


    def generate(self, f):
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
            time.sleep(30)
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