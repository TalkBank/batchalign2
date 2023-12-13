"""
parse_support.py
Read cleanup support wordlists
""" 

import os 
import pathlib
from dataclasses import dataclass

from functools import cache

from batchalign.pipelines.base import *
from batchalign.document import *

import re

@dataclass
class Replacement:
    original: str
    main_line_replacement: str
    lemma_replacement: str

@cache
def parse(name):
    # parse support fiel by name, ignoring all lines that
    # start with #
    dir = pathlib.Path(__file__).parent.resolve()
    try:
        with open(os.path.join(dir, "support", name), 'r') as df:
            lines = df.readlines()
            lines = [i.strip() for i in lines if i[0] != "#"]
    except FileNotFoundError:
        return {}
    # split the file by space
    lines = [i.split(" ") for i in lines]

    # serialize into replacement objects
    result = {}
    for original, main, lemma in lines:
        result[original] = Replacement(original=original,
                                       main_line_replacement=main,
                                       lemma_replacement=lemma)

    return result

def _mark_utterance(utterance:Utterance, support:str, type:TokenType, lang:str="eng"):
    """Mark an utterance for some property.

    Parameters
    ----------
    utterance : Utterance
        The utterance to mark.
    support : str
        The support file to find.
    type : TokenType
        The token type to mark matches.
    lang : str
        The language of matches.

    This function performs *IN PLACE* modifications.
    """
    
    data = parse(f"{support}.{lang}")

    # things to replace on the text level
    text_replace = []

    # replace everything in the lemma level
    for i in utterance.content:
        replacement = data.get(i.text.lower())
        if replacement:
            i.text = replacement.lemma_replacement
            text_replace.append(replacement)
            i.type = type

    # collect text level replacements together
    if utterance.text:
        for i in text_replace:
            orig = re.compile(re.escape(i.original), re.IGNORECASE)
            utterance.text = orig.sub(i.main_line_replacement, utterance.text)

