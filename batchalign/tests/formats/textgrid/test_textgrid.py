import os
import json
import pytest
import pathlib

from batchalign.document import *
from batchalign.formats.textgrid import TextGridFile

@pytest.fixture(scope="module")
def doc():
    dir = pathlib.Path(__file__).parent.resolve()
    with open(os.path.join(dir, "support", "test.json"), 'r') as df:
        doc = Document.model_validate(json.load(df))

        return doc

@pytest.fixture(scope="module")
def utterance_tg_path():
    dir = pathlib.Path(__file__).parent.resolve()
    return os.path.join(dir, "support", "utterance.TextGrid")

@pytest.fixture(scope="module")
def word_tg_path():
    dir = pathlib.Path(__file__).parent.resolve()
    return os.path.join(dir, "support", "word.TextGrid")

@pytest.fixture(scope="module")
def utterance_tg_data(utterance_tg_path):
    with open(utterance_tg_path, 'r') as df:
        return df.read().strip()

@pytest.fixture(scope="module")
def word_tg_data(word_tg_path):
    with open(word_tg_path, 'r') as df:
        return df.read().strip()
    
WORD_DOC = "hello \x150_420\x15\nis a test of batchline \x151280_2320\x15\ngoing to read some random crap as I see on the screen just to test batchline the primary area for recording editing and arranging audio \x152740_10040\x15\nand drummer regions divided into different track types \x1510340_12820\x15\ncommand slash for more info test test I don't know what to say but um here's some retracing \x1513520_19180\x15\njust for fun um I like I like I like beans \x1519540_24140\x15\nare fun thank you very much \x1524600_25940\x15"

UTT_DOC = "hello . \x150_420\x15\nI'm going to read some random crap as I see on the screen just to test batchline . \x152500_6480\x15\nthe primary area for recording , editing and arranging audio . \x156960_10040\x15\npress command slash for more info . \x1513220_14820\x15\ntest test . \x1515380_15940\x15\nI don't know what to say but um here's some retracing . \x1516400_19180\x15\nbeans are fun . \x1524380_25080\x15\nthank you very much . \x1525280_25940\x15\nthis is a test of batchline . \x15880_2320\x15\nMIDI and drummer regions divided into different track types . \x1510200_12820\x15\nso just for fun . \x1519420_20160\x15\num I like I like I like beans . \x1520800_24140\x15"

#################

def test_textgrid_generation(doc, word_tg_data, utterance_tg_data):
    assert str(TextGridFile("word", doc=doc)).strip() == word_tg_data.strip()
    assert str(TextGridFile("utterance", doc=doc)).strip() == utterance_tg_data.strip()

def test_textgrid_parsing(doc, word_tg_path, utterance_tg_path):
    assert WORD_DOC == str(TextGridFile("word", path=word_tg_path).doc)
    assert UTT_DOC == str(TextGridFile("utterance", path=utterance_tg_path).doc)


