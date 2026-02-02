"""
rev.py
Support for Rev.ai, a commerical ASR service
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.utils.config import config_read

from batchalign.errors import *

from batchalign.models import BertUtteranceModel, BertCantoneseUtteranceModel
from batchalign.models.resolve import resolve

import time
import pathlib
import pycountry

# removed top-level rev_ai and model imports

import logging
L = logging.getLogger("batchalign")

class RevEngine(BatchalignEngine):

    @property
    def tasks(self):
        # if there is no utterance segmentation scheme, we only
        # run ASR
        if self.__engine:
            return [ Task.ASR, Task.UTTERANCE_SEGMENTATION ]
        else:
            return [ Task.ASR ]

    def __init__(self, key: str | None = None, lang="eng", num_speakers=2):

        if key == None or key.strip() == "":
            config = config_read()
            try:
                key = config["asr"]["engine.rev.key"] 
            except KeyError:
                raise ConfigError("No Rev.AI key found. Rev.AI was not set up! Please set up Rev.ai through the initial setup process by running 'batchalign setup' in the command line to generate one, or write one yourself and place it at `~/.batchalign.ini`.")

        from batchalign.models import BertUtteranceModel, BertCantoneseUtteranceModel
        from batchalign.models.resolve import resolve
        from rev_ai import apiclient

        self.__lang_code = lang
        self.__num_speakers = num_speakers
        self.__lang = pycountry.languages.get(alpha_3=lang).alpha_2
        if self.__lang == "zh":
            self.__lang = "cmn"
        # apiclient is a module, but mypy treats RevAiAPIClient as non-callable from stubs.
        self.__client = apiclient.RevAiAPIClient(key)  # type: ignore[operator]
        utterance_model_res = resolve("utterance", lang)
        utterance_model: str | None = utterance_model_res if isinstance(utterance_model_res, str) else None
        if utterance_model is not None:
            L.debug("Initializing utterance model...")
            if lang != "yue":
                self.__engine = BertUtteranceModel(utterance_model)
            else:
                # we have special inference procedure for cantonese
                self.__engine = BertCantoneseUtteranceModel(utterance_model)
            L.debug("Done.")
        else:
            self.__engine = None


    def generate(self, source_path: str, **kwargs) -> Document:
        from rev_ai import JobStatus
        
        # bring language code into the stack to access
        lang = self.__lang
        client = self.__client


        L.info(f"Uploading '{pathlib.Path(source_path).stem}'...")
        # we will send the file for processing
        if "en" not in lang and "es" not in lang:
            job = client.submit_job_local_file(source_path,
                                               metadata=f"batchalign2_{pathlib.Path(source_path).stem}",
                                               language=lang)
        else:
            job = client.submit_job_local_file(source_path,
                                               metadata=f"batchalign2_{pathlib.Path(source_path).stem}",
                                               language=lang,
                                               skip_postprocessing=(True if resolve("utterance", self.__lang_code) is not None and lang in ["en", "fr"] else False),
                                               speakers_count=self.__num_speakers)

        # we will wait untitl job finishes
        status = client.get_job_details(job.id).status
        L.info(f"Rev.AI is transcribing '{pathlib.Path(source_path).stem}'...")

        # check status, sleeping every so often and check again
        while status == JobStatus.IN_PROGRESS:
            time.sleep(15)
            status = client.get_job_details(job.id).status
            L.debug(f"Rev.AI got '{status}'...")

        # if we failed, report failure and give up
        if status == JobStatus.FAILED:
            err = client.get_job_details(job.id).failure_detail
            raise RuntimeError(f"Rev.AI reports job failed! file='{pathlib.Path(source_path).stem}', error='{err}'.")

        # and now, we extract result and postprocess it
        transcript_json = client.get_transcript_json(job.id)
        L.debug(f"Rev.AI done.")

        # postprocess the output and define media tier
        doc = process_generation(transcript_json, self.__lang_code, utterance_engine=self.__engine)
        media = Media(type=MediaType.AUDIO, name=Path(source_path).stem, url=source_path)
        doc.media = media
        return doc
