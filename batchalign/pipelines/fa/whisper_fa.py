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
        group: list = []
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

        # Initialize cache infrastructure
        from batchalign.pipelines.cache import (
            CacheManager, AlignmentCacheKey, _get_batchalign_version
        )
        cache = CacheManager()
        key_gen = AlignmentCacheKey()
        ba_version = _get_batchalign_version()
        engine_version = "whisper-fa-v1" # engine identifier
        override_cache = kwargs.get("override_cache", False)

        for indx, grp in enumerate(groups):
            L.info(f"Whisper FA processing segment {indx+1}/{len(groups)}...")
            if self.status_hook != None:
                self.status_hook(indx+1, len(groups))

            # Cache at the group level
            group_key = None
            try:
                # Use a dummy utterance representing the group for key generation
                group_text = detokenize(word[0].text for word in grp)
                group_start = grp[0][1][0]
                group_end = grp[-1][1][1]

                # Tiny audio fingerprint for the group
                audio_hash = f.hash_chunk(group_start, group_end)

                # Combine into group key
                group_key = key_gen.generate_key(group_text, audio_hash, pauses)

                if not override_cache:
                    cached = cache.get(group_key, "forced_alignment", engine_version)
                    if cached is not None:
                        # Re-apply cached word timings to the group
                        cached_timings = key_gen.deserialize_output(cached)
                        if len(cached_timings) == len(grp):
                            for (word, _), t in zip(grp, cached_timings):
                                word.time = tuple(t) if t else None
                            continue # Skip Whisper
            except Exception as e:
                L.debug(f"Cache check failed for group {indx}: {e}")

            # perform alignment
            # we take a 2 second buffer in each direction
            try:
                detokenized = detokenize(word[0].text for word in grp)
                # replace ANY punctuation
                for p in MOR_PUNCT + ENDING_PUNCT:
                    detokenized = detokenized.replace(p, "").strip()
                # to ensure that combined words are pased correctly 
                detokenized = detokenized.replace("_", " ")
                res = self.__whisper(audio=f.chunk(grp[0][1][0], grp[-1][1][1]),
                                     text=detokenized, pauses=pauses)
            except IndexError:
                # utterance contains nothing
                continue

            # create reference backplates, which are the word ids to set the timing for
            ref_targets = []
            for i_idx, (word, _) in enumerate(grp):
                for char in word.text:
                    ref_targets.append(ReferenceTarget(char, payload=i_idx))
            # create target backplates for the timings
            payload_targets = []
            timings = []
            for i_idx, (word, time) in enumerate(res):
                timings.append(time)
                for char in word:
                    payload_targets.append(PayloadTarget(char, payload=i_idx))
            # alignment!
            alignments = align(payload_targets, ref_targets, tqdm=False)

            # set the ids back to the text ids
            # we do this BACKWARDS because we went to have the first timestamp
            # we get about a word first
            alignments.reverse()
            for i_idx,elem in enumerate(alignments):
                if isinstance(elem, Match):
                    if pauses:
                        next_elem: Match | None = None
                        next_elem_idx: int = i_idx - 1 # remember this is backwards, see above
                        while next_elem_idx >= 0 and alignments[next_elem_idx].payload == elem.payload:
                            next_elem_idx -= 1
                        if next_elem_idx < 0:
                            next_elem = None
                        else:
                            next_elem = alignments[next_elem_idx]
                        if next_elem is None:
                            next_time = timings[elem.payload]
                        else:
                            next_time = timings[next_elem.payload]
                        grp[elem.reference_payload][0].time = (
                            int(round((timings[elem.payload] * 1000 + grp[0][1][0]))),
                            int(round((next_time * 1000 + grp[0][1][0]))) + (500 if next_elem is None else 0),
                        )
                    else:
                        grp[elem.reference_payload][0].time = (int(round((timings[elem.payload]*1000 +
                                                                          grp[0][1][0]))),
                                                               int(round((timings[elem.payload]*1000 +
                                                                          grp[0][1][0]))))
            
            # Cache the results for this group
            if group_key is not None:
                try:
                    group_timings = [word[0].time for word in grp]
                    cache.put(group_key, "forced_alignment", engine_version, ba_version, {"timings": group_timings})
                except Exception as e:
                    L.debug(f"Failed to cache group {indx}: {e}")

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
                            next_alignment = doc.content[next_ut].alignment
                            if next_alignment is not None:
                                w.time = (w.time[0], next_alignment[0])
                        else:
                            w.time = (w.time[0], w.time[0]+500) # give half a second because we don't know
                    elif not pauses:
                        next_time = ut.content[tmp].time
                        if next_time is not None:
                            w.time = (w.time[0], next_time[0])

                    # just in case, bound the time by the utterance derived timings
                    if ut.alignment is not None and w.time is not None:
                        alignment_start, alignment_end = ut.alignment
                        if alignment_start is not None and alignment_end is not None:
                            w.time = (max(w.time[0], alignment_start), min(w.time[1], alignment_end))
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
