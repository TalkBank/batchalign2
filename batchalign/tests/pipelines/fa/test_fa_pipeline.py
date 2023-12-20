from batchalign.pipelines import BatchalignPipeline

def test_whisper_fa_pipeline(en_doc):
    whisper = BatchalignPipeline.new("fa", lang="eng", num_speakers=1, fa="whisper_fa")
    doc = whisper(en_doc)

    # TODO we won't check this accuracy for now
    


