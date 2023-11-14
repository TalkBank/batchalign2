import os 
import json
import pytest
import pathlib

from batchalign.formats.chat.utils import *
from batchalign.formats.chat.parser import *
from batchalign.document import *
from batchalign.errors import *

# tests typical utterance with full alignment and one missing bullet
STANDARD_UTTERANCE = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1~aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    '1|3|NSUBJ 2|3|AUX 3|18|ROOT 4|5|MARK 5|3|XCOMP 6|3|PUNCT',
    "I'm \x152530_2720\x15 going \x152720_2910\x15 to read \x153000_3200\x15 .",
    []
]
PARSED_STANDARD_UTTERANCE = [Form(text="I'm", time=(2530, 2720), morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=[Dependency(id=1, dep_id=3, dep_type='NSUBJ'), Dependency(id=2, dep_id=3, dep_type='AUX')]), Form(text='going', time=(2720, 2910), morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=[Dependency(id=3, dep_id=18, dep_type='ROOT')]), Form(text='to', time=None, morphology=[Morphology(lemma='to', pos='part', feats='')], dependency=[Dependency(id=4, dep_id=5, dep_type='MARK')]), Form(text='read', time=(3000, 3200), morphology=[Morphology(lemma='read', pos='verb', feats='Inf')], dependency=[Dependency(id=5, dep_id=3, dep_type='XCOMP')]), Form(text='.', time=None, morphology=[Morphology(lemma='.', pos='PUNCT', feats='')], dependency=[Dependency(id=6, dep_id=3, dep_type='PUNCT')])]

STANDARD_NOTIME_NOMOR = [
    "I'm going to read .",
    None,
    '1|3|NSUBJ 2|3|AUX 3|18|ROOT 4|5|MARK 5|3|XCOMP 6|3|PUNCT',
    None,
    []
]
PARSED_NOTIME_NOMOR = [Form(text="I'm", time=None, morphology=None, dependency=None), Form(text='going', time=None, morphology=None, dependency=None), Form(text='to', time=None, morphology=None, dependency=None), Form(text='read', time=None, morphology=None, dependency=None), Form(text='.', time=None, morphology=None, dependency=None)]

STANDARD_NOGRA = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1~aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    None,
    None,
    []
]
PARSED_NOGRA = [Form(text="I'm", time=None, morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=None), Form(text='going', time=None, morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=None), Form(text='to', time=None, morphology=[Morphology(lemma='to', pos='part', feats='')], dependency=None), Form(text='read', time=None, morphology=[Morphology(lemma='read', pos='verb', feats='Inf')], dependency=None), Form(text='.', time=None, morphology=[Morphology(lemma='.', pos='PUNCT', feats='')], dependency=None)]


MISALIGNED_UTTERANCE_MOR = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1 aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    '1|3|NSUBJ 2|3|AUX 3|18|ROOT 4|5|MARK 5|3|XCOMP 6|3|PUNCT',
    "I'm \x152530_2720\x15 going \x152720_2910\x15 to read \x153000_3200\x15 .",
    []
]
MISALIGNED_UTTERANCE_CHOP_GRA = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1~aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    '1|3|NSUBJ 2|3|AUX 6|3|PUNCT',
    "I'm \x152530_2720\x15 going \x152720_2910\x15 to read \x153000_3200\x15 .",
    []
]
MISALIGNED_UTTERANCE_TOOMUCH_GRA = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1~aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    '1|3|NSUBJ 2|3|AUX 3|18|ROOT 4|5|MARK 4|5|MARK 4|5|MARK 6|3|PUNCT',
    "I'm \x152530_2720\x15 going \x152720_2910\x15 to read \x153000_3200\x15 .",
    []
]

# test various forms of normal parses
def test_parse_utterance():
    res, delim = chat_parse_utterance(*STANDARD_UTTERANCE)
    assert delim == "."
    assert res == PARSED_STANDARD_UTTERANCE
def test_parse_utterance_missing_mor_time():
    res, delim = chat_parse_utterance(*STANDARD_NOTIME_NOMOR)
    assert delim == "."
    assert res == PARSED_NOTIME_NOMOR
def test_parse_utterance_missing_gra():
    res, delim = chat_parse_utterance(*STANDARD_NOGRA)
    assert delim == "."
    assert res == PARSED_NOGRA

# tests various forms of alignment errors
def test_parse_utterance_misaligned():
    with pytest.raises(CHATValidationException):
        res, delim = chat_parse_utterance(*MISALIGNED_UTTERANCE_MOR)
def test_parse_utterance_misaligned_chopped_gra():
    with pytest.raises(CHATValidationException):
        res, delim = chat_parse_utterance(*MISALIGNED_UTTERANCE_CHOP_GRA)
def test_parse_utterance_misaligned_toomuch_gra():
    with pytest.raises(CHATValidationException):
        res, delim = chat_parse_utterance(*MISALIGNED_UTTERANCE_TOOMUCH_GRA)
