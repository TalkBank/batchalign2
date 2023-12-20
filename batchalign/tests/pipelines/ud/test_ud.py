from batchalign.pipelines.morphosyntax.ud import morphoanalyze
from batchalign.document import *

import warnings

CEST_TAGGED = [{'lemma': 'ce', 'pos': 'pron', 'feats': 'Dem-S3'},
               {'lemma': 'être', 'pos': 'aux', 'feats': 'Fin-Ind-3-Pres'}]

JUSQU_AU_TAGGED = {'text': "jusqu'au",
                   'time': None,
                   'morphology': [{'lemma': 'jusque', 'pos': 'adp', 'feats': ''},
                                  {'lemma': 'au', 'pos': 'adv', 'feats': ''}],
                   'dependency': [{'id': 4, 'dep_id': 5, 'dep_type': 'CASE'},
                                  {'id': 5, 'dep_id': 3, 'dep_type': 'ADVMOD'}],
                   'type': 0}

def test_ud_pipeline(en_doc):
    assert morphoanalyze(en_doc) == en_doc
    
# email dec092023-10:31
def test_ud_cest_mwt():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("c'est du réglisse .", lang="fra"))

        assert [i.model_dump() for i in res[0][0].morphology] == CEST_TAGGED

# email dec202023-10:02
def test_ud_jusqu():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("tu vas aller jusqu'au.", lang="fra"))
        assert res[0][-2].model_dump() == JUSQU_AU_TAGGED



