from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils.dp import *
from batchalign.constants import *

import logging
L = logging.getLogger("batchalign")


class CantoneseSegmentationEngine(BatchalignEngine):
    tasks = [Task.WORD_SEGMENTATION]

    def __init__(self):
        import pycantonese
        self.__segment = pycantonese.segment

    def process(self, doc: Document, **kwargs):
        # no-op for non-Cantonese
        if "yue" not in doc.langs:
            L.debug("CantoneseSegmentation: skipping non-Cantonese document.")
            return doc

        for ut in doc.content:
            if not isinstance(ut, Utterance):
                continue

            # separate regular words from punctuation/features at the end
            regular_forms = []
            trailing = []
            for f in ut.content:
                if f.type in (TokenType.PUNCT, TokenType.FEAT, TokenType.ANNOT):
                    trailing.append(f)
                else:
                    # if we already started collecting trailing punct,
                    # but hit another regular token, flush trailing back
                    if trailing:
                        regular_forms.extend(trailing)
                        trailing = []
                    regular_forms.append(f)

            if not regular_forms:
                continue

            # combine all regular token text (no spaces for CJK)
            original_text = "".join(f.text for f in regular_forms)
            if not original_text.strip():
                continue

            # run pycantonese segmentation
            segmented_words = self.__segment(original_text)

            # use character-level DP alignment to map segmented words
            # back onto original tokens
            #
            # reference: chars from original tokens, payload = index into regular_forms
            ref_targets = []
            for i, f in enumerate(regular_forms):
                for ch in f.text:
                    ref_targets.append(ReferenceTarget(ch, payload=i))

            # payload: chars from segmented words, payload = index into segmented_words
            pay_targets = []
            for i, word in enumerate(segmented_words):
                for ch in word:
                    pay_targets.append(PayloadTarget(ch, payload=i))

            alignments = align(pay_targets, ref_targets, tqdm=False)

            # build a mapping: segmented word index -> list of original token indices
            seg_to_orig: dict[int, list[int]] = {}
            for elem in alignments:
                if isinstance(elem, Match):
                    seg_to_orig.setdefault(elem.payload, []).append(elem.reference_payload)

            # construct new Form objects from segmented words
            new_forms = []
            for seg_idx, word in enumerate(segmented_words):
                orig_indices = seg_to_orig.get(seg_idx, [])

                # inherit timing: start from first orig, end from last orig
                time = None
                if orig_indices:
                    first_time = regular_forms[orig_indices[0]].time
                    last_time = regular_forms[orig_indices[-1]].time
                    if first_time is not None and last_time is not None:
                        time = (first_time[0], last_time[1])
                    elif first_time is not None:
                        time = first_time
                    elif last_time is not None:
                        time = last_time

                # determine token type from the first matched original
                token_type = TokenType.REGULAR
                if orig_indices:
                    token_type = regular_forms[orig_indices[0]].type

                new_forms.append(Form(
                    text=word,
                    time=time,
                    type=token_type,
                ))

            # re-append trailing punctuation/features
            new_forms.extend(trailing)
            ut.content = new_forms
            # clear cached text so _detokenize() regenerates from new Forms
            ut.text = None

        return doc
