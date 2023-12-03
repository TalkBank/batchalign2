from batchalign.pipelines.ud import morphoanalyze

def test_ud_pipeline(gold_en):
    assert morphoanalyze(gold_en) == gold_en
    

