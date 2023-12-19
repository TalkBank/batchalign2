from batchalign.pipelines.morphosyntax.ud import morphoanalyze
from batchalign.document import *

import warnings

CEST_TAGGED = [{'lemma': 'ce', 'pos': 'pron', 'feats': 'Dem-S3'},
               {'lemma': 'être', 'pos': 'aux', 'feats': 'Fin-Ind-3-Pres'}]

def test_ud_pipeline(en_doc):
    assert morphoanalyze(en_doc) == en_doc
    
# email dec092023-10:31
def test_ud_cest_mwt():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("c'est du réglisse .", lang="fra"))

        assert [i.model_dump() for i in res[0][0].morphology] == CEST_TAGGED

