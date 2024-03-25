from batchalign.pipelines.cleanup import NgramRetraceEngine, DisfluencyReplacementEngine
from batchalign.document import *

import copy
import pytest

BOTH = 'this &-um is all <so crazy so crazy so so crazy> [/] so crazy , everybody [/] everybody seem [/] seem so famous [/] famous I am a big scary dinosaur I am a big &-um &-um &-um &-um scary dinosaur I am a big scary dinosaur .'
DISF = 'this &-um is all so crazy so crazy so so crazy so crazy , everybody everybody seem seem so famous famous I am a big scary dinosaur I am a big &-um &-um &-um &-um scary dinosaur I am a big scary dinosaur .'
RET = 'this um is all <so crazy so crazy so so crazy> [/] so crazy , everybody [/] everybody seem [/] seem so famous [/] famous I am a big scary dinosaur I am a big um [/] um [/] um [/] um scary dinosaur I am a big scary dinosaur .'
SRC = "this um is all so crazy so crazy so so crazy so crazy, everybody everybody seem seem so famous famous I am a big scary dinosaur I am a big um um um um scary dinosaur I am a big scary dinosaur."

RET_WITH_DISFLUENCY = 'um this is this is a retrace'

BEG_WITH_DISFLUENCY = 'um this is um this is a retrace'

@pytest.fixture(scope="module")
def doc():
    return Document.new(SRC)

@pytest.fixture(scope="module")
def nr():
    return NgramRetraceEngine()

@pytest.fixture(scope="module")
def dr():
    return DisfluencyReplacementEngine()

def test_ngram_retrace(doc, nr):
    # tests the application of only the ngram engine
    d = copy.deepcopy(doc)
    dp = nr(d)


    assert str(doc) == str(d)
    assert str(dp) == RET

def test_disfluency(doc, dr):
    # tests the application of only the disfluency engine
    d = copy.deepcopy(doc)
    dp = dr(d)

    assert str(doc) == str(d)
    assert str(dp) == DISF

def test_retrace_and_disf(doc, dr, nr):
    # tests that the application of both engines are sound
    # in whichever order 
    d1 = copy.deepcopy(doc)
    d2 = copy.deepcopy(doc)

    dp1 = dr(nr(doc))
    dp2 = nr(dr(doc))

    assert str(doc) == str(d1)
    assert str(doc) == str(d2)

    assert str(dp1) == BOTH
    assert str(dp2) == BOTH

# regression #idk
def test_retrace_with_disfluency(nr):
    doc = Document.new(RET_WITH_DISFLUENCY)
    docp = nr(doc)

    assert str(docp) == "um <this is> [/] this is a retrace"

# regression email 01/01/2024 10:01AM
def test_beg_with_disfluency(nr):
    doc = Document.new(BEG_WITH_DISFLUENCY)

    docp = nr(doc)
    assert str(docp) == "<um this is> [/] um this is a retrace"

# regression email 03/25/2024 2:21PM
def test_nested_retrace(nr):
    doc = Document.new("ice ice ice ice ice cream ice cream")

    docp = nr(doc)

    assert str(docp) == "ice [/] ice [/] ice [/] ice [/] <ice cream> [/] ice cream"


