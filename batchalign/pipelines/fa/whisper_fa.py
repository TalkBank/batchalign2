from batchalign.models import WhisperFAModel
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

class WhisperFAEngine(BatchalignEngine):
    tasks = [ Task.FORCED_ALIGNMENT ]

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def __init__(self, model=None):

        self.status_hook = None

        if model == None:
            model = "openai/whisper-large-v2"

        self.__whisper = WhisperFAModel(model)

    def process(self, doc:Document, **kwargs):
        # check if pauses
        pauses = kwargs.get("pauses", False)

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
            if not isinstance(i, Utterance):
                continue
            if i.alignment == None:
                warnings.warn("We found at least one utterance without utterance-level alignment; this is usually not an issue, but if the entire transcript is unaligned, it means that utterance level timing recovery (which is fuzzy using ASR) failed due to the audio clarity. On this transcript, before running forced-alignment, please supply utterance-level links.")
                continue

            # pop the previous group onto the stack
            if (i.alignment[-1] - seg_start) > 20*1000:
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
            if self.status_hook != None:
                self.status_hook(indx+1, len(groups))

            # perform alignment
            # we take a 2 second buffer in each direction
            try:
                detokenized = detokenize(word[0].text for word in grp)
                # replace ANY punctuation
                for i in MOR_PUNCT + ENDING_PUNCT:
                    detokenized = detokenized.replace(i, "").strip()
                # to ensure that combined words are pased correctly 
                detokenized = detokenized.replace("_", " ")
                # if "noone's" in detokenized:
                    # breakpoint()
                res = self.__whisper(audio=f.chunk(grp[0][1][0], grp[-1][1][1]),
                                     text=detokenized, pauses=pauses)
            except IndexError:
                # utterance contains nothing
                continue

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
            for indx,elem in enumerate(alignments):
                if isinstance(elem, Match):
                    if pauses:
                        next_elem = indx - 1 # remember this is backwards, see above
                        while next_elem >= 0 and alignments[next_elem].payload == elem.payload:
                            next_elem -= 1
                        if next_elem < 0:
                            next_elem = None
                        else:
                            next_elem = alignments[next_elem]
                        grp[elem.reference_payload][0].time = (int(round((timings[elem.payload]*1000 +
                                                                        grp[0][1][0]))),
                                                            int(round((timings[elem.payload]*1000 +
                                                                        grp[0][1][0])))+500 if next_elem == None else
                                                            int(round((timings[next_elem.payload]*1000 +
                                                                        grp[0][1][0]))))
                    else:
                        grp[elem.reference_payload][0].time = (int(round((timings[elem.payload]*1000 +
                                                                          grp[0][1][0]))),
                                                               int(round((timings[elem.payload]*1000 +
                                                                          grp[0][1][0]))))

        L.debug(f"Correcting text...")

        # we now set the end alignment of each word to the start of the next
        for doc_ut, ut in enumerate(doc.content):
            if not isinstance(ut, Utterance):
                continue

            # correct each word by bumping it forward
            # and if its not a word we remove the timing
            for indx, w in enumerate(ut.content):
                if w.type in [TokenType.PUNCT, TokenType.FEAT, TokenType.ANNOT]:
                    w.time = None
                elif indx == len(ut.content)-1 and w.text in ENDING_PUNCT:
                    w.time = None
                elif indx != len(ut.content)-1:
                    # search forward for the next compatible time
                    tmp = indx+1
                    while tmp < len(ut.content)-1 and ut.content[tmp].time == None:
                        tmp += 1
                    if w.time == None:
                        continue
                    if ut.content[tmp].time == None:
                        # seek forward one utterance to find their start time
                        next_ut = doc_ut + 1 
                        while next_ut < len(doc.content)-1 and (not isinstance(doc.content, Utterance) or doc.content[next_ut].alignment == None):
                            next_ut += 1
                        if next_ut < len(doc.content) and isinstance(doc.content, Utterance) and doc.content[next_ut].alignment:
                            w.time = (w.time[0], doc.content[next_ut].alignment[0])
                        else:
                            w.time = (w.time[0], w.time[0]+500) # give half a second because we don't know
                    elif not pauses:
                        w.time = (w.time[0], ut.content[tmp].time[0])

                    # just in case, bound the time by the utterance derived timings
                    if ut.alignment and ut.alignment[0] != None:
                        w.time = (max(w.time[0], ut.alignment[0]), min(w.time[1], ut.alignment[1]))
                    # if we ended up with timings that don't make sense, drop it
                    if w.time and w.time[0] >= w.time[1]:
                        w.time = None

            # clear any built-in timing (i.e. we should use utterance-derived timing)
            ut.time = None
            # correct the text 
            if ut.alignment and ut.text != None:
                if '\x15' not in ut.text:
                    ut.text = (ut.text+f" \x15{ut.alignment[0]}_{ut.alignment[1]}\x15").strip()
                else:
                    ut.text = re.sub(r"\x15\d+_\d+\x15",
                                     f"\x15{ut.alignment[0]}_{ut.alignment[1]}\x15", ut.text).strip()
            elif ut.text != None:
                ut.text = re.sub(r"\x15\d+_\d+\x15", f"", ut.text).strip()

        return doc
