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
PARSED_STANDARD_UTTERANCE = [Form(text="I'm", time=(2530, 2720), morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=[Dependency(id=1, dep_id=3, dep_type='NSUBJ'), Dependency(id=2, dep_id=3, dep_type='AUX')]), Form(text='going', time=(2720, 2910), morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=[Dependency(id=3, dep_id=18, dep_type='ROOT')]), Form(text='to', time=None, morphology=[Morphology(lemma='to', pos='part', feats='')], dependency=[Dependency(id=4, dep_id=5, dep_type='MARK')]), Form(text='read', time=(3000, 3200), morphology=[Morphology(lemma='read', pos='verb', feats='Inf')], dependency=[Dependency(id=5, dep_id=3, dep_type='XCOMP')]), Form(text='.', time=None, morphology=[Morphology(lemma='.', pos='PUNCT', feats='')], dependency=[Dependency(id=6, dep_id=3, dep_type='PUNCT')], type=5)]

STANDARD_NOTIME_NOMOR = [
    "I'm going to read .",
    None,
    '1|3|NSUBJ 2|3|AUX 3|18|ROOT 4|5|MARK 5|3|XCOMP 6|3|PUNCT',
    None,
    []
]
PARSED_NOTIME_NOMOR = [Form(text="I'm", time=None, morphology=None, dependency=None), Form(text='going', time=None, morphology=None, dependency=None), Form(text='to', time=None, morphology=None, dependency=None), Form(text='read', time=None, morphology=None, dependency=None), Form(text='.', time=None, morphology=None, dependency=None, type=5)]

STANDARD_NOGRA = [
    "I'm going to read . \x152530_6490\x15",
    'pron|I-Prs-Nom-S1~aux|be-Fin-Ind-1-Pres verb|go-Part-Pres part|to verb|read-Inf .',
    None,
    None,
    []
]
PARSED_NOGRA = [Form(text="I'm", time=None, morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=None), Form(text='going', time=None, morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=None), Form(text='to', time=None, morphology=[Morphology(lemma='to', pos='part', feats='')], dependency=None), Form(text='read', time=None, morphology=[Morphology(lemma='read', pos='verb', feats='Inf')], dependency=None), Form(text='.', time=None, morphology=[Morphology(lemma='.', pos='PUNCT', feats='')], dependency=None, type=5)]


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

LINE_WITH_COLON = [
    "@Begin",
    "@Situation:\tlol:colon.",
    "@End",
]

LINE_WITH_BRAKET_GROUPS = "mais il [= the tap] <est> [% sch=va] trop fort ."
PARSED_BRAKET_GROUPS = ([Form(text='mais'),
                         Form(text='il'),
                         Form(text='est'),
                         Form(text='trop'),
                         Form(text='fort'),
                         Form(text='.', type=5)],
                        '.')

# email dec202023 7:19pm
LINE_WITH_SKIPPING_BRAKETS = "pas <des> [% sch=de] [/] des [/] des formes ."
PARSED_LINE_WITH_SKIPPING_BRAKETS = ([Form(text='pas'),
                                      Form(text='des', type=1),
                                      Form(text='des', type=1),
                                      Form(text='des', type=0),
                                      Form(text='formes'),
                                      Form(text='.', type=5)],
                                     '.')

# general test for all types of group parses
CONTINOUS_GROUPS = "Humph without [=? what ] xxx <    bateau alo > [: à l'eau] [?] i <je fais > [=? j'ai fait] <I am> [=? this is hilarious    ] [/] lmfmao this [: mmm] [/] yo <ahaha ha> [/] what is [/] happening ."
PARSED_CONTINOUS_GROUPS = ([Form(text='Humph'), Form(text='without'), Form(text='à'), Form(text="l'eau"),
                            Form(text='i'), Form(text='je'), Form(text='fais'), Form(text='I', type=1),
                            Form(text='am', type=1), Form(text='lmfmao'), Form(text='mmm', type=1), Form(text='yo'),
                            Form(text='ahaha', type=1), Form(text='ha', type=1), Form(text='what'),
                            Form(text='is', type =1), Form(text='happening'), Form(text='.', type=5)], '.')

# general test for recursive groups parses as well as special symbols
# emali DEC232023 0723
RECURSIVE_GROUPS = "+< <(es)tá (a)brien(d)o [= la puerta del salón] [: lol psych this is a retrace] [/] (.) ábrele sometimes> [?] no it [=! susurrando] [/] isn't, or is [/] it +... [=a tmp]"
PARSED_RECURSIVE_GROUPS = ([Form(text='está', type=0), Form(text='lol', type=1), Form(text='psych', type=1),
                            Form(text='this', type=1), Form(text='is', type=1), Form(text='a', type=1),
                            Form(text='retrace', type=1), Form(text='ábrele', type=0), Form(text='sometimes', type=0),
                            Form(text='no', type=0), Form(text='it', type=1), Form(text="isn't", type=0),
                            Form(text=',', type=5), Form(text='or', type=0), Form(text='is', type=1),
                            Form(text='it', type=0), Form(text='+...', type=5)], '+...')

# nested groups
# Email DEC142024 1208
NESTED_GROUPS = "<chiquitito <se pu> [//] <no quier> [///] se quería bañar> [=! cantando] +..."
PARSED_NESTED_GROUPS = ([Form(text='chiquitito'), Form(text='se', type=1), Form(text='pu', type=1),
                         Form(text='no', type=1), Form(text='quier', type=1), Form(text='se'),
                         Form(text='quería'), Form(text='bañar'), Form(text='+...', type=5)], '+...')

EDGE_CASES = [
    # email dec192023 10:31AM: wack ending
    [
        "quand j'aurai quarante ans +...",
        "sconj|quand pron|moi-Prs-Acc-S1~verb|avoir-Fin-Ind-1-Pres num|quarante noun|an-Masc-Plur +...", 
        "1|3|MARK 2|3|NSUBJ 3|5|ROOT 4|5|NUMMOD 5|3|OBJ 6|3|PUNCT"
    ], [
        # email dec192023 10:31AM: xxx
        "xxx; Grégoire .",
        "noun|Grégoire-ComNeut .", 
        "1|1|ROOT 2|1|PUNCT"
    ],
    # ], [
    #     # email dec192023 10:31AM: wack group
    #     "fouf@c [= moufle]; moufle .",
    #     "x|fouf noun|moufle-Fem .", 
    #     "1|2|FLAT 2|1|NMOD 3|1|PUNCT"
    # ],
    [
        # email dec192023 10:31AM: group whose beginning ends with an ENDING_PUNCT
        "je fais [=? j'ai fait] amadouadou@si dehors avec une (pe)tite fille de trois ans .",
        "pron|moi-Prs-Acc-S1 verb|faire-Fin-Ind-1-Pres x|amadouadou adv|dehors adp|avec det|un-Ind-Art adj|petit-Pos-S-1 noun|fille-Fem adp|de num|trois noun|an-Masc-Plur .", 
        "1|2|NSUBJ 2|11|ROOT 3|2|FLAT 4|2|ADVMOD 5|8|CASE 6|8|DET 7|8|AMOD 8|2|OBL-MOD 9|11|CASE 10|11|NUMMOD 11|8|NMOD 12|2|PUNCT"
    ],
    [
        # email dec192023 10:31AM: group which has spaces near the group markers
        "< (ç)a coule > [?] .",
        "pron|ça-Dem-Acc-S3 verb|couler-Fin-Ind-3-Pres .", 
        "1|2|NSUBJ 2|2|ROOT 3|2|PUNCT"
    ]
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

# email dec192023 10:31AM
def test_special_line_with_colon():
    tmp = chat_parse_doc(LINE_WITH_COLON)

def test_edge_cases():
    for i,j,k in EDGE_CASES:
        chat_parse_utterance(i, j, k, None, None)

# email dec202023 10:02AM
def test_group_markers():
    assert (chat_parse_utterance(LINE_WITH_BRAKET_GROUPS, None, None, None, None) ==
            PARSED_BRAKET_GROUPS)

# email dec202023 7:19pm
def test_group_markers():
    assert (chat_parse_utterance(LINE_WITH_SKIPPING_BRAKETS, None, None, None, None) ==
            PARSED_LINE_WITH_SKIPPING_BRAKETS)

def test_continous_groups():
    assert (chat_parse_utterance(CONTINOUS_GROUPS, None, None, None, None) ==
            PARSED_CONTINOUS_GROUPS)

# email dec232023 7:23pm
def test_recursive_groups():
    assert (chat_parse_utterance(RECURSIVE_GROUPS, None, None, None, None) ==
            PARSED_RECURSIVE_GROUPS)

# email dec242023 12:08PM
def test_nested_groups():
    assert (chat_parse_utterance(NESTED_GROUPS, None, None, None, None) ==
            PARSED_NESTED_GROUPS)
def test_ca_marks():
    assert len(chat_parse_utterance("⌊-=tararea⌋ . •458470_458480•", None, None, None, None)[0]) == 1
def test_repeated_marks_in_same_lex():
    assert len(chat_parse_utterance("[- eng] <mind if I borrow xxx>[<][*] ?", None, None, None, None)[0]) == 5




