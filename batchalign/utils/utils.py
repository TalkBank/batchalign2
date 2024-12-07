import re
import nltk
from nltk import word_tokenize as WT
from nltk import sent_tokenize as ST
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import TweetTokenizer
from batchalign.constants import *
from contextlib import contextmanager

import os

def word_tokenize(str):
    """Tokenize a string by word

    Parameters
    ----------
    str : str
        input string.

    Returns
    -------
    List[str]
        Word tokens.
    """

    tmp = TweetTokenizer()
    
    try:
        return tmp.tokenize(str)
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        return tmp.tokenize(str)

def sent_tokenize(str):
    """Tokenize a string by sentence

    Parameters
    ----------
    str : str
        input string.

    Returns
    -------
    List[str]
        Sentence tokens.
    """
 
    try:
        return ST(str)
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        return ST(str)

def detokenize(tokens):
    """Merge tokenized words.

    Parameters
    ----------
    tokens : List[str]
        input tokens.

    Returns
    -------
    str
        Result strings.
    """
 
    try:
        return TreebankWordDetokenizer().detokenize(tokens)
    except LookupError:
        nltk.download("punkt")
        nltk.download("punkt_tab")
        return TreebankWordDetokenizer().detokenize(tokens)

def correct_timing(doc):
    """Correct the timings of ASR.

    Parameters
    ----------
    doc : Document
        The Document to correct ASR output timings of.

    Returns
    -------
    Document
        The document that has the utterance timings corrected.
    """
    
    # correct the utterance-level timings
    last_end = 0
    for i in doc.content:
        # bump time forward
        if i.alignment:
            time = list(i.alignment)
            if i.alignment[0] < last_end:
                time[0] = last_end
            last_end = time[1]
            i.time = tuple(time)
            # if the time has been squished to nothing, we clear time
            if i.alignment[1] <= i.alignment[0]:
                i.time = None
                for j in i.content:
                    j.time = None
            # otherwise, we remove impossible timestamps
            else:
                for j in i.content:
                    if j.time and j.time[1] <= j.time[0]:
                        j.time = None
    return doc



@contextmanager
def silence():
    # silence all outputs
    fd0 = os.dup(0)
    fd1 = os.dup(1)
    fd2 = os.dup(2)

    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), 0)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

    # yield to call
    yield

    # Restore the original sys.stdout file descriptor
    os.dup2(fd0, 0)
    os.dup2(fd1, 1)
    os.dup2(fd2, 2)
