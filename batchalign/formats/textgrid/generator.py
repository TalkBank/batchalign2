from batchalign.document import *
from batchalign.constants import *
from collections import defaultdict

import re
from praatio import textgrid
from praatio.utilities.constants import Interval, Point

from warnings import warn

def _extract_tiers(doc, by_word = True):
    tiers = defaultdict(list)

    # we throw away any special tiers
    utterances = list(filter(lambda x:isinstance(x, Utterance), doc.content))
    for indx_ut, ut in enumerate(utterances):
        # if its by word do that
        if by_word:
            for indx, w in enumerate(ut.content):
                # we skip punctuation
                if w.text in ENDING_PUNCT or w.text in MOR_PUNCT:
                    continue

                if w.time == None:
                    # infer alignments by interpolating
                    prev_indx = indx -1
                    next_indx = indx + 1

                    while prev_indx > 0 and ut.content[prev_indx].time == None:
                        prev_indx -= 1
                    while (next_indx < len(ut.content) and ut.content[next_indx].time == None):
                        next_indx += 1

                    if prev_indx < 0 or next_indx >= len(ut.content):
                        warn(f"Encountered a word with no alignment and no way to interpolate an alignment! Praat requires every form to be aligned, and so we must skip it. Utterance='{ut.strip()}', word='{w.text}'.")
                    else:
                        tiers[ut.tier.id].append((w.text,
                                                ut.content[prev_indx].time[-1],
                                                ut.content[next_indx].time[0]))
                else:
                    tiers[ut.tier.id].append((w.text,
                                            w.time[0],
                                            w.time[1]))

        else:
            if ut.alignment == None:
                # infer alignments by interpolating
                prev_indx = indx_ut -1
                next_indx = indx_ut + 1

                while prev_indx > 0 and utterances[prev_indx].alignment == None:
                    prev_indx -= 1
                while next_indx < len(utterances) and utterances[next_indx].alignment == None:
                    next_indx += 1

                if prev_indx < 0 or next_indx >= len(utterances):
                    warn(f"Encountered a utterance with no alignment and no way to interpolate an alignment! Praat requires every form to be aligned, and so we must skip it. Utterance='{ut.strip()}'.")
                    continue

                tiers[ut.tier.id].append((ut.strip(False, True, True),
                                        utterances[prev_indx].alignment[-1],
                                        utterances[next_indx].alignment[0]))
            else:
                tiers[ut.tier.id].append((ut.strip(False, True, True),
                                          ut.alignment[0],
                                          ut.alignment[1]))

    return dict(tiers)

def dump_textgrid(doc, by_word=True):
    tg = textgrid.Textgrid()

    # extract the tiers from the doc, meannk
    tiers = _extract_tiers(doc, by_word)

    # convert each field to intervals
    intervals = {k:[Interval(start/1000, end/1000, word) for word, start, end in v]
                for k,v in tiers.items()}
    # tiers
    for k,v in intervals.items():
        tg.addTier(textgrid.IntervalTier(k, v, v[0].start, v[-1].end))

    return tg
