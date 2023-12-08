from batchalign.pipelines.ud import morphoanalyze

def test_ud_pipeline(en_doc):
    assert morphoanalyze(en_doc) == en_doc
    

