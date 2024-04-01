from batchalign.models import NemoSpeakerModel
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils import *
from batchalign.utils.dp import *
from batchalign.constants import *

import logging
L = logging.getLogger("batchalign")

import re

import pycountry
import warnings

class NemoSpeakerEngine(BatchalignEngine):
    tasks = [ Task.SPEAKER_RECOGNITION ]

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def __init__(self, num_speakers=None):

        self.status_hook = None
        self.num_speakers = num_speakers
        self.__model = NemoSpeakerModel()

    def process(self, doc:Document, **kwargs):
        # check that the document has a media path to align to
        assert doc.media != None and doc.media.url != None, f"We cannot speaker ID something that doesn't have speaker information path! Provided media tier='{doc.media}'"
        # assume num speaker is class default, or 2 if there isn't
        num_speakers = kwargs.get("num_speakers")
        if num_speakers == None:
            num_speakers = self.num_speakers
        if num_speakers == None:
            L.warning("Assuming two speakers as `num_speakers` arg was not passed.")
            num_speakers = 2

        # load the audio file
        L.info(f"Running speaker ID on {doc.media.url} with {num_speakers} speaker(s)...")
        speaker_ids = self.__model(doc.media.url)
        L.debug(f"Speaker ID finished.")

        # grab a list of speakers and generate tiers for them
        speakers = list(set([i[2] for i in speaker_ids]))
        speaker_tier_map = {
            i:Tier(lang=doc.langs[0], corpus="corpus_name",
                   id=f"PAR{i}", name=f"Participant")
            for i in speakers
        }

        # run two pointers line to assign speaker; choosing the LAST good diarization
        # which the start time of an utterance is a part
        spk = speaker_ids.pop(0)[-1]

        for ut in doc.content:
            if isinstance(ut, Utterance):
                ut_start = ut.alignment[0]
                # peek and check if the next tier starts before
                # ut_start
                if len(speaker_ids) > 0 and speaker_ids[0][0] <= ut_start:
                    spk = speaker_ids.pop(0)[-1]
                ut.tier = speaker_tier_map[spk]

        return doc
