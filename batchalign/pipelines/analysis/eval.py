"""
eval.py
Engines for transcript evaluation
"""

import re
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

        forms = [i.replace("-", "") for i in forms if i.strip() not in MOR_PUNCT+ENDING_PUNCT]
        gold_forms = [i.replace("-", "") for i in gold_forms if i.strip() not in MOR_PUNCT+ENDING_PUNCT]

        forms = [re.sub(r"\((.*)\)",r"", i) for i in forms]
        gold_forms = [re.sub(r"\((.*)\)",r"", i) for i in gold_forms]

        # if there are single letter frames, we combine them tofgether
        # until the utterance is done or there isn't any left
        forms_finished = []

        single_sticky = ""
        is_single = False

        for i in forms:
            if len(i) == 1 and ("zho" not in doc.langs):
                single_sticky += i
            else:
                if single_sticky != "":
                    forms_finished.append(single_sticky)
                    single_sticky = ""
                forms_finished.append(i)

        if single_sticky != "":
            forms_finished.append(single_sticky)
            single_sticky = ""

        # special Chinese processing
        gold_final = []
        forms_final = []

        if "zho" in doc.langs:
            for i in gold_forms:
                for j in i:
                    gold_final.append(j)
            for i in forms_finished:
                for j in i:
                    forms_final.append(j)
        else:
            gold_final = gold_forms
            forms_final = forms_finished

        # dp!
        alignment = align(forms_final, gold_final, False)

        # calculate each type of error
        sub = 0
        dl = 0
        ins = 0
        prev_error = None

        # we consider each pair of different extra-type a substitution
        # ie: if we have <extra.payload> <extra.reference> +> substitution
        #     but if we have <extra.reference> <extra.reference> this is 2 insertions

        cleaned_alignment = []
        # whether we had a "firstname" in reference document and hence are
        # anticipating a payload for it (the actual name) in the next entry in the
        # alignment
        anticipating_payload = False

        for i in alignment:

            if isinstance(i, Extra):

                if i.extra_type == ExtraType.REFERENCE and "name" in i.key and i.key[:4] != "name":
                    if (isinstance(cleaned_alignment[-1], Extra) and
                        cleaned_alignment[-1].extra_type ==  ExtraType.PAYLOAD and
                        len(cleaned_alignment) > 0):
                        cleaned_alignment.pop(-1)
                    else:
                        anticipating_payload = True
                    cleaned_alignment.append(Match(i.key, None, None))
                    continue
                elif i.extra_type == ExtraType.PAYLOAD and anticipating_payload:
                    anticipating_payload = False
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
        for i in cleaned_alignment:
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
            "diff": diff,
            "doc": doc
        }






