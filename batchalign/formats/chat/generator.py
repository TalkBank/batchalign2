from batchalign.document import *
from batchalign.formats import CHATFile

c = CHATFile("./extern/test.cha")
document = c.doc
utterance = document[0]

# def utterance_to_chat(utterance: Utterance):
main_line = str(utterance)


