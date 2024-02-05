from batchalign.pipelines.base import *
from batchalign.document import *
from batchalign.pipelines.cleanup.parse_support import _mark_utterance

import re

class DisfluencyReplacementEngine(BatchalignEngine):

    tasks = [ Task.DISFLUENCY_ANALYSIS ]

    def process(self, doc, **kwargs):
        primary_language = doc.langs[0]
        for ut in doc.content:
            _mark_utterance(ut, "filled_pauses", TokenType.FP, primary_language)
        for ut in doc.content:
            _mark_utterance(ut, "replacements", TokenType.REGULAR, primary_language)
        return doc

