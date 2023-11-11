from batchalign import CHATFile
import json


c = CHATFile("./extern/test.cha")
c.doc.content[0].model_dump()


