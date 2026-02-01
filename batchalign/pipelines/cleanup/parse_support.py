"""
parse_support.py
Read cleanup support wordlists
""" 

import os 
import pathlib
from dataclasses import dataclass
from typing import Dict

from functools import cache

from batchalign.pipelines.base import BatchalignEngine
from batchalign.document import Utterance, TokenType, Task

import re

@dataclass
class Replacement:
    original: str
    main_line_replacement: str
    lemma_replacement: str

@cache
def parse(name: str) -> Dict[str, Replacement]:
    # parse support file by name, ignoring all lines that
    # start with #
    dir = pathlib.Path(__file__).parent.resolve()
    try:
        with open(os.path.join(dir, "support", name), 'r') as df:
            lines = df.readlines()
            lines = [i.strip() for i in lines if i[0] != "#"]
    except FileNotFoundError:
        return {}
    
    # serialize into replacement objects
    result: Dict[str, Replacement] = {}
    for line in lines:
        parts = line.split(" ")
        if len(parts) >= 3:
            original, main, lemma = parts[0], parts[1], parts[2]
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
        for repl in text_replace:
            orig = re.compile(re.escape(repl.original), re.IGNORECASE)
            utterance.text = orig.sub(repl.main_line_replacement, utterance.text)

