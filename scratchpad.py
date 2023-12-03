from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
L.getLogger("stanza").setLevel(L.ERROR)
L.getLogger('batchalign').setLevel(L.DEBUG)

########

from batchalign import *

# lead input chat files
input = CHATFile(path="./extern/tmp/test.cha")
doc = input.doc

import json
print(json.dumps(doc.model_dump(), indent=2))

# Document

# doc.model_dump()


# for i in doc[0].content:
    # print(i.type)
# doc[0].strip()
# doc[0][0]

# load ASR and morphosyntax engine
# whisper = WhisperEngine(num_speakers=1)
ud = UDEngine()

# cosntruct and a pipeline with the engines
pipeline = BatchalignPipeline(processors=[ud])
result = pipeline(input.doc)

# write output
CHATFile(doc=result).write("./extern/tmp.cha")

# result
# result.media

# doc = CHATFile(path="./test.cha").doc
# doc
# from batchalign.pipelines.ud import morphoanalyze

# doc = morphoanalyze(doc)
# CHATFile(doc=doc).write("./tmp.cha")






