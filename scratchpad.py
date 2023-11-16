from batchalign import CHATFile
import json

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.WARNING)
L.getLogger('batchalign').setLevel(L.DEBUG)

#### TODOS ####
# - angle brackets for stringification
# - interpolate time backwards if the last form doesn't have a time

c = CHATFile("./extern/test.cha")
document = c.doc

document

# document[0].model_json_schema()
# document

# document[0][0]
# document[0][2]

# document.transcript(strip=True)
# document.model_copy()

