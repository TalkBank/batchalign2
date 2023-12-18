from batchalign.document import *
from batchalign.constants import *
from collections import defaultdict

import copy
import re
from praatio import textgrid
from praatio.utilities.constants import Interval, Point

from warnings import warn

def load_textgrid_word(tg, lang, corpus_name):
    """Load TextGrid files where each interval is an utterance.

    Parameters
    ----------
    tg : TextGrid
        TextGrid file to load.
    lang : str
        3 letter language code.
    corpus_name : str
        Corpus name.

    Returns
    -------
    Document
        The document created by treating each TextGrid element as a
        word.
    """
    
    # collect all the forms together
    forms = []

    for tier in tg.tiers:
        # collect all the forms into one place
        for entry in tier.entries:
            forms.append((tier.name, Form(text=entry.label,
                                        time=(int(entry.start*1000),
                                                int(entry.end*1000)))))

    # sorts all the forms
    forms = sorted(forms, key=lambda x:x[1].time[0])

    # if the textgrid returned no forms (which shouldn't be the case
    # because return_empty_forms is false), we just return in kind an empty doc
    if len(forms) == 0:
        doc = Document()
        doc.lang = lang
        return doc

    # create utterances based on if the participant changed
    prev_par = forms[0][0]

    turns = []
    turn = []
    while len(forms) > 0:
        par, form = forms.pop(0)

        if par == prev_par:
            turn.append(form)
        else:
            turns.append((par, copy.deepcopy(turn)))
            turn = []
            prev_par = par
    turns.append((par, copy.deepcopy(turn)))

    # cache the tiers
    tiers = {}
    utterances = []

    for spk, turn in turns:
        # get or create tier
        tier = tiers.get(spk)
        if tier == None:
            tier = Tier(lang=lang, corpus=corpus_name, id=spk)
            tiers[spk] = tier
        # and create utterance
        utterances.append(Utterance(tier=tier, content=turn))

    # create the document
    doc = Document(content=utterances, lang=[lang])

    return doc

def load_textgrid_utterance(tg, lang, corpus_name):
    """Load TextGrid files where each interval is an utterance.

    Parameters
    ----------
    tg : TextGrid
        TextGrid file to load.
    lang : str
        3 letter language code.
    corpus_name : str
        Corpus name.

    Returns
    -------
    Document
        The document created by treating each TextGrid element as a
        word.
    """

    # collect all the forms together
    forms = []
    tiers = {}

    for tier in tg.tiers:
        # collect all the forms into one place
        for entry in tier.entries:
            # get or create tier
            t = tiers.get(tier.name)
            if t == None:
                t = Tier(lang=lang, corpus=corpus_name, id=tier.name)
                tiers[tier.name] = t

            forms.append(Utterance(content=entry.label,
                                time=(int(entry.start*1000), int(entry.end*1000)),
                                tier=t))
    # create the document
    return Document(content=forms, lang=[lang])

def load_textgrid(tg, lang="eng", corpus_name="corpus_name", by_word = True):
    """Load TextGrid files

    Parameters
    ----------
    tg : TextGrid
        TextGrid file to load.
    lang : str
        3 letter language code.
    corpus_name : str
        Corpus name.
    by_word : bool
        Whether the TextGrids's intervals are words (True) or utterances False

    Returns
    -------
    Document
        The document created by treating each TextGrid element as a
        word.
    """
 
    if by_word:
        return load_textgrid_word(tg, lang, corpus_name)
    else:
        return load_textgrid_utterance(tg, lang, corpus_name)

