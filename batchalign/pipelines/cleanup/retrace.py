from batchalign.pipelines.base import *
from batchalign.document import *
from batchalign.pipelines.cleanup.parse_support import _mark_utterance

import re

class NgramRetraceEngine(BatchalignEngine):

    tasks = [ Task.RETRACE_ANALYSIS ]

    def process(self, doc):

        for ut in doc.content:
            # get only the main content (we don't want to double-mark retrace)
            content = []
            for i in ut.content:
                if i.type in [TokenType.REGULAR, TokenType.PUNCT]:
                    content.append(i)
            # scan for n-gram retraces
            for n in range(1, len(content)):
                begin = 0
                while begin < len(content)-(n+1):
                    # get the n gram info; we convert it to
                    # a tuple to make it hashable
                    gram = tuple(content[begin:begin+n])
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
                    while tuple(content[root+n:root+2*n]) == gram:
                        for j in content[begin:begin+n]:
                            j.type = TokenType.RETRACE
                        root = root+n
                        # we bump begin forward until AFTER the retrace
                        # and the main content
                        begin = root-1
                    # we scan grams forward one by one
                    begin += 1
            breakpoint()


