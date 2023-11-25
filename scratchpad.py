from batchalign import *
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.WARNING)
L.getLogger('batchalign').setLevel(L.DEBUG)



########

tmp = CHATFile(path="./input.cha")
tmp.write("./test.cha")


ud = UDEngine()
pipeline = BatchalignPipeline(processors=[ud])
result = pipeline(tmp.doc)

CHATFile(doc=result).write("./test.cha")

# result
# result.media

# doc = CHATFile(path="./test.cha").doc
# doc
# from batchalign.pipelines.ud import morphoanalyze

# doc = morphoanalyze(doc)
# CHATFile(doc=doc).write("./tmp.cha")






