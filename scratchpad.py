from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.WARNING)
L.getLogger('batchalign').setLevel(L.DEBUG)

whisper = WhisperEngine(num_speakers=1)
pipeline = BatchalignPipeline(generator=whisper)

result = pipeline("./extern/test.wav")

result

result.media
CHATFile(doc=result).write("./tmp.cha")


# c = CHATFile("./extern/test.cha")

# doc = c.doc

# doc[3].content
# doc[3].text

# doc = Document.from_media("./extern/test.wav")





