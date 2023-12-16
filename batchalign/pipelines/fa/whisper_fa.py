from batchalign.models.fa import WhisperFAModel
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.dp import *
from batchalign.utils import *
from batchalign.constants import *

import logging
L = logging.getLogger("batchalign")

import re

import pycountry
import warnings

class WhisperFAEngine(BatchalignEngine):
    tasks = [ Task.FORCED_ALIGNMENT ]

    def __init__(self, model=None):

        if model == None:
            model = "openai/whisper-large-v2"

        self.__whisper = WhisperFAModel(model)

    def process(self, doc:Document):
        # check that the document has a media path to align to
        assert doc.media != None and doc.media.url != None, f"We cannot forced-align something that doesn't have a media path! Provided media tier='{doc.media}'"

        # load the audio file
        L.debug(f"Whisper FA is loading url {doc.media.url}...")
        f = self.__whisper.load(doc.media.url)
        L.debug(f"Whisper FA finished loading media.")

        # collect utterances 30 secondish segments to be aligned for whisper
        # we have to do this because whisper does poorly with very short segments
        groups = []
        group = []
        seg_start = 0

        L.debug(f"Whisper FA finished loading media.")

        for i in doc.content:
            if i.alignment == None:
                warnings.warn("Cannot force-align an utterance without utterance-level segmentation; please run a pipeline that includes `Task.UTTERANCE_TIMING_RECOVERY` (\"bulletize\") before running forced-alignment.")

            # pop the previous group onto the stack
            if (i.alignment[-1] - seg_start) > 28*1000:
                groups.append(group)
                group = []
                seg_start = i.alignment[0]

            # append the contents to the running group
            for word in i.content:
                group.append((word, i.alignment))

        groups.append(group)

        L.debug(f"Begin Whisper Inference...")

        for indx, grp in enumerate(groups):
            L.info(f"Whisper FA processing segment {indx+1}/{len(groups)}...")

            # perform alignment
            # we take a 2 second buffer in each direction
            res = self.__whisper(audio=f.chunk(grp[0][1][0], grp[-1][1][1]),
                                 text=detokenize(word[0].text for word in grp))

            # create reference backplates, which are the word ids to set the timing for
            ref_targets = []
            for indx, (word, _) in enumerate(grp):
                for char in word.text:
                    ref_targets.append(ReferenceTarget(char, payload=indx))
            # create target backplates for the timings
            payload_targets = []
            timings = []
            for indx, (word, time) in enumerate(res):
                timings.append(time)
                for char in word:
                    payload_targets.append(PayloadTarget(char, payload=indx))
            # alignment!
            alignments = align(payload_targets, ref_targets, tqdm=False)

            # set the ids back to the text ids
            # we do this BACKWARDS because we went to have the first timestamp
            # we get about a word first
            alignments.reverse()
            for elem in alignments:
                if isinstance(elem, Match):
                    grp[elem.reference_payload][0].time = (int(round((timings[elem.payload]*1000 +
                                                                      grp[0][1][0]))),
                                                           int(round((timings[elem.payload]*1000 +
                                                                      grp[0][1][0]))))

        L.debug(f"Correcting text...")

        # we now set the end alignment of each word to the start of the next
        for ut in doc.content:
            # correct each word by bumping it forward
            # and if its not a word we remove the timing
            for indx, w in enumerate(ut.content):
                if w.type in [TokenType.PUNCT, TokenType.FEAT, TokenType.ANNOT]:
                    w.time = None
                elif indx == len(ut.content)-1 and w.text in ENDING_PUNCT:
                    w.time = None
                elif indx != len(ut.content)-1:
                    w.time = (w.time[0], ut.content[indx+1].time[0])
            # clear any built-in timing (i.e. we should use utterance-derived timing)
            ut.time = None
            # correct the text 
            if ut.alignment:
                ut.text = re.sub("\x15\d+_\d+\x15", f"\x15{ut.alignment[0]}_{ut.alignment[1]}\x15", ut.text).strip()
            else:
                ut.text = re.sub("\x15\d+_\d+\x15", f"", ut.text).strip()

        return doc
