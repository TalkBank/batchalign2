from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
L.getLogger("stanza").setLevel(L.ERROR)
L.getLogger('batchalign').setLevel(L.DEBUG)

########

from batchalign import *

# doc = Document.new("This represents apple's remarkable innovation, and deep collaboration between different teams, and reaffirms our commitment to making the world a better place")

rev = WhisperEngine(num_speakers=1)
tmp = rev("./batchalign/tests/support/test.mp3")
tmp

# doc[0][0]
# from nltk.tokenize import word_tokenize
# tmp = TweetTokenizer()
# word_tokenize("¡por supuesto, maestro!")

disf = DisfluencyEngine()
# # ore = OrthographyReplacementEngine()
doc = disf(tmp)
doc
# # oc = disf(ore(doc))


# ud = UDEngine()
# doc = ud(doc)
# doc[0][0]
# doc[0][0]



# # CHATFile(path="./tmp.cha").doc

# whisper = WhisperEngine(num_speakers=1)
# doc = whisper.generate("./batchalign/tests/support/test.mp3")


# doc = ud(doc)

# doc[0][0].type

# doc.model_dump()
