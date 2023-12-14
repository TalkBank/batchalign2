from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
L.getLogger("stanza").setLevel(L.ERROR)
L.getLogger('batchalign').setLevel(L.DEBUG)

########

from batchalign import *

doc = Document.new("howdy partner, what's your name?")


whisper = WhisperEngine()
ud = UDEngine()

nlp = BatchalignPipeline(ud, whisper)

res = nlp("./batchalign/tests/support/test.mp3")

nlp.tasks

# tmp

# - engine capabilities: 
#  -- revise to a list of tasks each engine performs
#  -- and the pipeline takes a series of engines, and orders them in the sensical order based on their tasks
#  -- each engine can perform multiple tasks

# doc[0][0]
# from nltk.tokenize import word_tokenize
# tmp = TweetTokenizer()
# word_tokenize("¡por supuesto, maestro!")

# disf = DisfluencyEngine()
# # ore = OrthographyReplacementEngine()
# doc = disf(tmp)
# doc
# # oc = disf(ore(doc))
# from transformers import WhisperForConditionalGeneration
# tmp = WhisperForConditionalGeneration.from_pretrained("talkbank/CHATWhisper-en-large-v1")


# ud = UDEngine()
# doc = ud(doc)
# doc[0][0]
# doc[0][0]
# doc[0][0]



# # CHATFile(path="./tmp.cha").doc

# whisper = WhisperEngine(num_speakers=1)
# doc = whisper.generate("./batchalign/tests/support/test.mp3")


# doc = ud(doc)

# doc[0][0].type

# doc.model_dump()
