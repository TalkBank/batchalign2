# system utils
import glob, os, re
from itertools import groupby

# pathing tools
from pathlib import Path

# UD tools
import stanza

import copy

from stanza.utils.conll import CoNLL
from stanza import Document, DownloadMethod
from stanza.models.common.doc import Token
from stanza.pipeline.core import CONSTITUENCY
from stanza import DownloadMethod
from torch import heaviside

from stanza.pipeline.processor import ProcessorVariant, register_processor_variant
from stanza.resources.common import download_resources_json, load_resources_json, get_language_resources

# the loading bar
from tqdm import tqdm

from bdb import BdbQuit

from nltk import word_tokenize
from collections import defaultdict

import warnings

from stanza.utils.conll import CoNLL

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob.glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, pathlib.Path(file_path).name)


from batchalign.document import *
from batchalign.constants import *
from batchalign.pipelines.base import *
from batchalign.formats.chat.parser import chat_parse_utterance

from batchalign.utils.dp import *

from pyannote.audio import Pipeline

import logging
L = logging.getLogger("batchalign")

import pycountry

class PyannoteEngine(BatchalignEngine):
    tasks = [ Task.SPEAKER_RECOGNITION ]
    status_hook = None

    def __init__(self, num_speakers=2):
        self.pipe = Pipeline.from_pretrained("talkbank/dia-fork")
        self.num_speakers = num_speakers

    def process(self, doc):
        assert doc.media != None and doc.media.url != None, f"We cannot diarize something that doesn't have a media path! Provided media tier='{doc.media}'"
        res = self.pipe(doc.media.url, num_speakers=self.num_speakers)

        speakers = list(set([int(i[-1].split("_")[-1])
                            for i in res.itertracks(yield_label=True)]))
        corpus = doc.tiers[0].corpus
        lang = doc.tiers[0].lang
        tiers = {
            i:
            Tier(
                lang=lang, corpus=corpus,
                id="PAR"+str(i), name="Participant",
                birthday="",
            )
            for i in speakers
        }

        for i in doc.content:
            if not isinstance(i, Utterance):
                continue
            if i.alignment is None:
                continue
            start,end = i.alignment
            if start is None or end is None:
                continue

            for (a,b),_,speaker in res.itertracks(yield_label=True):
                speaker_id = int(speaker.split("_")[-1])
                tier = tiers.get(speaker_id)
                # we set the end time of the utterance as the
                # *LAST* segment it ends before
                # i.e. [seg_end, ....., ut_end]
                # like that 
                if b <= end/1000 and tier:
                    i.tier = tier

        # doc.tiers = list(tiers.values())

        return doc


