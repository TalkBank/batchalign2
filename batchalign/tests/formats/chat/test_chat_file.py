import os 
import json
import pytest
import pathlib

from batchalign.formats.chat import *
from batchalign.document import *
from batchalign.errors import *

# tests header and metadata parsing 
def test_whole_file():
    dir = pathlib.Path(__file__).parent.resolve()
    c = CHATFile(os.path.join(dir, "support", "test.cha"))
    with open(os.path.join(dir, "support", "success.json"), 'r') as df:
        correct = Document.model_validate_json(df.read())
        assert correct == c.doc


