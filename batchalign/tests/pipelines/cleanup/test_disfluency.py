from batchalign.pipelines.cleanup import NgramRetraceEngine, DisfluencyReplacementEngine
from batchalign.document import *

import copy
import pytest

BOTH = 'this &-um is all <so crazy> [/] so crazy <so so crazy> [/] so crazy , everybody [/] everybody seem [/] seem so famous [/] famous I am a big scary dinosaur I am a big &-um &-um &-um &-um scary dinosaur I am a big scary dinosaur .'
DISF = 'this &-um is all so crazy so crazy so so crazy so crazy , everybody everybody seem seem so famous famous I am a big scary dinosaur I am a big &-um &-um &-um &-um scary dinosaur I am a big scary dinosaur .'
RET = 'this um is all <so crazy> [/] so crazy <so so crazy> [/] so crazy , everybody [/] everybody seem [/] seem so famous [/] famous I am a big scary dinosaur I am a big <um um um> [/] um scary dinosaur I am a big scary dinosaur .'

@pytest.fixture(scope="module")
def doc():
    return Document.new("this um is all so crazy so crazy so so crazy so crazy, everybody everybody seem seem so famous famous I am a big scary dinosaur I am a big um um um um scary dinosaur I am a big scary dinosaur.")

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

