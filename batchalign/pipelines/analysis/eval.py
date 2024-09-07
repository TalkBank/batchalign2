"""
eval.py
Engines for transcript evaluation
"""

from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.pipelines.asr.utils import *
from batchalign.utils.config import config_read

from batchalign.utils.dp import align, ExtraType, Extra, Match

import logging
L = logging.getLogger("batchalign")

class EvaluationEngine(BatchalignEngine):
    tasks = [ Task.WER ]

    @staticmethod
    def __compute_wer(doc, gold):
        # get the text of the document and get the text of the gold
        forms = [ j.text.lower() for i in doc.content for j in i.content if isinstance(i, Utterance)]
        gold_forms = [ j.text.lower() for i in gold.content for j in i.content if isinstance(i, Utterance)]

        forms = [i for i in forms if i.strip() not in MOR_PUNCT+ENDING_PUNCT]
        gold_forms = [i for i in gold_forms if i.strip() not in MOR_PUNCT+ENDING_PUNCT]

        # dp!
        alignment = align(forms, gold_forms, False)

        # calculate each type of error
        sub = 0
        dl = 0
        ins = 0
        prev_error = None

        # we consider each pair of different extra-type a substitution
        # ie: if we have <extra.payload> <extra.reference> +> substitution
        #     but if we have <extra.reference> <extra.reference> this is 2 insertions

        cleaned_alignment = []

        for i in alignment:

            if isinstance(i, Extra):
                if len(cleaned_alignment) > 0 and i.extra_type == ExtraType.REFERENCE and "name" in i.key and i.key[:4] != "name":
                    cleaned_alignment.pop(-1)
                    cleaned_alignment.append(Match(i.key, None, None))
                    continue

                if prev_error != None and prev_error != i.extra_type:
                    # this is a substitution: we have different "extra"s in
                    # reference vs. playload
                    sub += 1

                    # we need to subtract the addition caused by the
                    # previous entry (which we now know is a substitution)
                    if prev_error == ExtraType.REFERENCE:
                        ins -= 1
                    else:
                        dl -= 1

                    prev_error = None
                else:
                    # otherwise, we need to consider what type of error
                    # it is to know where to add
                    prev_error = i.extra_type
                    if i.extra_type == ExtraType.REFERENCE:
                        ins += 1
                    else:
                        dl += 1
            else:
                prev_error = None

            cleaned_alignment.append(i)

        diff = []
        for i in alignment:
            if isinstance(i, Extra):
                diff.append(f"{'+' if i.extra_type == ExtraType.REFERENCE else '-'} {i.key}")
            else:
                diff.append(f"  {i.key}")
                
        # wer = (S+D+I)/N
        return (sub+dl+ins)/len(gold_forms), "\n".join(diff)

    def analyze(self, doc, **kwargs):
        gold = kwargs.get("gold")

        # check that the gold transcript is what we want
        if not gold or not isinstance(gold, Document):
            raise ValueError(f"Unexpected format for gold transcript. Expected batchalign.Document, got '{type(gold)}'")

        wer, diff = self.__compute_wer(doc, gold)

        return {
            "wer": wer,
            "diff": diff
        }






