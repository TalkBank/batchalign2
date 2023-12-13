import re
import nltk
from nltk import word_tokenize as WT
from nltk import sent_tokenize as ST
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import TweetTokenizer
from batchalign.constants import *

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
        return TreebankWordDetokenizer().detokenize(tokens)

