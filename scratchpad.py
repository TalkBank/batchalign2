from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
L.getLogger("stanza").setLevel(L.ERROR)
L.getLogger('batchalign').setLevel(L.DEBUG)

########

from batchalign import *

doc = Document.new("Hello my name Jack. How uh are you doing today?")
doc


# ud = UDEngine()


# # CHATFile(path="./tmp.cha").doc

# whisper = WhisperEngine(num_speakers=1)
# doc = whisper.generate("./batchalign/tests/support/test.mp3")


# doc = ud(doc)

# doc[0][0].type

# doc.model_dump()
