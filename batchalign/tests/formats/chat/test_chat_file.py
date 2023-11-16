import os 
import json
import pytest
import pathlib

from batchalign.formats.chat import *
from batchalign.document import *
from batchalign.errors import *
from batchalign.constants import *

# tests header and metadata parsing 
def test_whole_file():
    dir = pathlib.Path(__file__).parent.resolve()
    c = CHATFile(os.path.join(dir, "support", "test.cha"))
    with open(os.path.join(dir, "support", "success.json"), 'r') as df:
        correct = Document.model_validate_json(df.read())
        assert correct == c.doc

# tests header and metadata parsing 
def test_media_link():
    dir = pathlib.Path(__file__).parent.resolve()
    unlinked = CHATFile(os.path.join(dir, "support", "unlinked.cha"))
    linked = CHATFile(os.path.join(dir, "support", "test.cha"))
    missing = CHATFile(os.path.join(dir, "support", "media_missing.cha"))

    assert unlinked.doc.media.type == MediaType.UNLINKED_AUDIO
    assert linked.doc.media.type == MediaType.AUDIO
    assert missing.doc.media == None



