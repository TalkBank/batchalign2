import pytest

from batchalign.formats.chat.utils import *
from batchalign.document import *
from batchalign.errors import *

GRA_VALID = "5|3|XCOMP"
GRA_VALID_DOC = Dependency(id=5, dep_id=3, dep_type="XCOMP")

MOR_VALID = "mod|do-3S~neg|not"
MOR_VALID_DOC = [Morphology(lemma='do', pos='mod', feats='3S'),
                 Morphology(lemma='not', pos='neg', feats='')]
MOR_VALID_MULTIFEAT = "mod|do-3S-2n"
MOR_VALID_MULTIFEAT_DOC = [Morphology(lemma='do', pos='mod', feats='3S-2n')]
MOR_INVALID = "moddo-3S"

STR_TO_CLEAN = "d-⌋ʔwhat&=um"
STR_CLEANED = "dwhatum"

# chat_parse_gra
def test_parse_gra():
    assert chat_parse_gra(GRA_VALID) == GRA_VALID_DOC

# parse_mor
def test_parse_mor():
    assert chat_parse_mor(MOR_VALID) == MOR_VALID_DOC
    assert chat_parse_mor(MOR_VALID_MULTIFEAT) == MOR_VALID_MULTIFEAT_DOC
def test_parse_mor_throw():
    with pytest.raises(CHATValidationException):
        chat_parse_mor(MOR_INVALID)

# annotation_clean
def test_annotation_clean():
    assert annotation_clean(STR_TO_CLEAN) == STR_CLEANED


