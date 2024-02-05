from batchalign.pipelines.analysis.eval import EvaluationEngine
from batchalign.document import *
from copy import deepcopy

import warnings

import pytest

# eval engine fixure
@pytest.fixture(scope="module")
def eval_engine():
    return EvaluationEngine()

def test_wer(en_doc, eval_engine):
    # create a broken doc
    d_alt = deepcopy(en_doc)
    d_alt[3].content[3].text="chicken"
    del d_alt.content[0]
    del d_alt.content[5]

    # create an analysis engine
    analysis = eval_engine(d_alt, gold=en_doc)

    # check for the right WER
    assert analysis["wer"] == 0.06382978723404255
   
