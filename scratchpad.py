from batchalign import CHATFile
import json

c = CHATFile("./extern/test.cha")
document = c.doc


document.transcript(strip=True)

