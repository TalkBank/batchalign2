"""
rev.py
Support for Rev.ai, a commerical ASR service
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.utils.config import config_read

from batchalign.pipelines.utr.utils import bulletize_doc

from batchalign.errors import *
import warnings 

import time
import pathlib
import pycountry

from rev_ai import apiclient, JobStatus

import logging
L = logging.getLogger("batchalign")

class RevUTREngine(BatchalignEngine):
    tasks = [ Task.UTTERANCE_TIMING_RECOVERY ]

    def __init__(self, key:str=None, lang="eng"):

        if key == None or key.strip() == "":
            config = config_read()
            try:
                key = config["asr"]["engine.rev.key"] 
            except KeyError:
                raise ConfigError("No Rev.AI key found. Rev.AI was not set up! Please set up Rev.ai through the initial setup process by running 'batchalign setup' in the command line to generate one, or write one yourself and place it at `~/.batchalign.ini`.")

        self.__lang_code = lang
        self.__lang = pycountry.languages.get(alpha_3=lang).alpha_2
        self.__client = apiclient.RevAiAPIClient(key)


    def process(self, doc, **kwargs):
        # bring language code into the stack to access
        lang = self.__lang
        try:
            lang = pycountry.languages.get(alpha_3=doc.langs[0]).alpha_2
        except:
            # some languages don't have alpha 2
            pass

        if lang == "zh":
            lang = "cmn"

        client = self.__client

        assert doc.media != None and doc.media.url != None, f"We cannot add utterance timings to something that doesn't have a media path! Provided media tier='{doc.media}'"

        # check and if there are existing utterance timings, warn
        if any([i.alignment for i in doc.content if isinstance(i, Utterance)]):
            warnings.warn(f"We found existing utterance timings in the document with {doc.media.url}! Skipping rough utterance alignment.")
            return doc

        f = doc.media.url

        L.info(f"Uploading '{pathlib.Path(f).stem}'...")
        # we will send the file for processing
        job = client.submit_job_local_file(f,
                                           metadata=f"batchalign2_{pathlib.Path(f).stem}",
                                           language=lang,
                                           # some languages don't have postprocessors, so this option
                                           # raises an exception
                                           skip_postprocessing=(lang in "en"))

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
        L.debug(f"bulletizing...")

        res =  bulletize_doc(transcript_json, doc)
        L.debug(f"done...")

        return res


