from batchalign.pipelines.base import *
from batchalign.document import *
from batchalign.pipelines.cleanup.parse_support import _mark_utterance

import warnings

import re

RETRACE_WARNING = "Retracing analysis detected that an utterance has an existing text line, for instance from an original input file. This has been removed!\n\nBatchalign will re-render the utterance text on its own accord if retracing or disfluency analysis is needed, instead of attempting to modify the original line.\n\nHint: You should only run Batchalign retracing engine on a newly minted document for instance from ASR."

class NgramRetraceEngine(BatchalignEngine):

    tasks = [ Task.RETRACE_ANALYSIS ]

    def process(self, doc, **kwargs):
        has_text = False

        for ut in doc.content:
            # get only the main content (we don't want to double-mark retrace)
            content = []
            for i in ut.content:
                if i.type in [TokenType.REGULAR, TokenType.PUNCT, TokenType.FP]:
                    content.append(i)
            # scan for n-gram retraces
            for n in range(1, len(content)):
                begin = 0
                while begin < len(content)-(n):
                    # get the n gram info; we convert it to
                    # a tuple to make it hashable
                    gram = tuple([i.text for i in content[begin:begin+n]])
                    # and compare forward, marking retraces 
                    # as needed. recall that content[] contain
                    # *pointers* to forms so we can just modify them
                    # in place
                    root = begin 
                    # we compare the same slice of n forward, and check
                    # if the slice of n forward is the same as the slice
                    # of n back. recall the last instance of the occurance
                    # is not a retrace: <hello pie hello pie> [/] hello pie .
                    # so that's why if we check the next ngram against the
                    # DOUBLE next
                    while tuple([i.text for i in content[root+n:root+2*n]]) == gram:
                        for j in content[begin:begin+n]:
                            if j.type != TokenType.FP:
                                j.type = TokenType.RETRACE
                                for p in ENDING_PUNCT + MOR_PUNCT:
                                    j.text = j.text.replace(p, "").strip()
                        root = root+n
                    # we scan grams forward one by one
                    begin += 1

            # give up
            if ut.text:
                has_text = True
                ut.text = None

        if has_text:
            warnings.warn(RETRACE_WARNING)

        return doc


