from batchalign.pipelines import BatchalignPipeline

DOC = "hello . \x150_420\x15\nthis is a test of batchline . \x15880_2320\x15\nI'm going to read some random crap as I see on the screen just to test batchline . \x152500_6480\x15\nthe primary area for recording , editing and arranging audio . \x156960_10040\x15\nMIDI and drummer regions divided into different track types . \x1510200_12820\x15\npress command slash for more info . \x1513220_14820\x15\ntest [/] test . \x1515360_15940\x15\nI don't know what to say but &-um here's some retracing . \x1516400_19180\x15\nso just for fun . \x1519420_20160\x15\n&-um <I like I like> [/] I like beans . \x1520800_24140\x15\nbeans are fun . \x1524380_25080\x15\nthank you very much . \x1525280_25940\x15"

def test_whisper_asr_pipeline(en_doc):
    whisper = BatchalignPipeline.new("asr", lang_code="eng", num_speakers=1, asr="whisper")
    doc = whisper(en_doc)

    assert str(doc) == DOC


