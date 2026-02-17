from batchalign import *
import json
from glob import glob
from pathlib import Path
from rich.console import Console
import copy
import os

import logging as L 

LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
L.getLogger("stanza").setLevel(L.ERROR)
L.getLogger('batchalign').setLevel(L.DEBUG)


########


from batchalign import *
from batchalign.formats.chat import CHATFile

doc = CHATFile(path="../talkbank-alignment/output/short.cha").doc
doc

# from batchalign.formats.chat.parser import chat_parse_utterance
# from batchalign.formats.chat import CHATFile

# f = CHATFile(path="../talkbank-alignment/testing_playground/input/test.cha")
# doc = f.doc

doc[1].content[0].time = (4,2)
doc[1].content[2].time = (12425, 12825)
# doc[3].content[4].time = (14425, 14525)
f = CHATFile(doc=doc)

print(f._CHATFile__generate(f._CHATFile__doc, f._CHATFile__special_mor, write_wor=True))


# doc = Document.new("I've been feeling like a rap god")
# doc[0][0]

# # 1+1
# text = "Hello are you the f b i ?"
# langs = ["eng"]
# parsed, delim = chat_parse_utterance(text, None, None, None, None)
# ut = Utterance(content=parsed, delim=delim, text=text)
# doc = Document(content=[ut], langs=langs)

# # pipe = BatchalignPipeline.new("morphosyntax", lang="jpn")
# # doc_out = pipe(doc, retokenize=True)

# cf = CHATFile(path="/Users/houjun/Documents/Projects/talkbank-alignment/comma/mm.cha")
# print(str(cf))
# print(str())

# 1+1


# doc = CHATFile(path="../talkbank-alignment/input/011116.cha").doc
# newdoc = Document(content=[doc[4]], langs=["heb"])
# pipe = StanzaEngine()
# res = pipe(newdoc)
# print(CHATFile(doc=res))

# ours = BatchalignPipeline.new("asr", lang="eng", asr="rev")
# doc = Document.new(media_path="../talkbank-alignment/input/test.mp3", lang="eng")
# doc = ours(doc)
# CHATFile(doc=doc).write("../talkbank-alignment/input/test.cha")

# from pyannote.audio import Pipeline
# pipe = Pipeline.from_pretrained("talkbank/dia-fork")
# res = pipe("../talkbank-alignment/input/test.mp3", num_speakers=2)

# speakers = list(set([int(i[-1].split("_")[-1])
#                      for i in res.itertracks(yield_label=True)]))
# corpus = doc.tiers[0].corpus
# lang = doc.tiers[0].lang
# tiers = {
#     i:
#     Tier(
#         lang=lang, corpus=corpus,
#         id="PAR"+str(i), name="Participant",
#         birthday="",
#     )
#     for i in speakers
# }

# for i in doc.content:
#     if not isinstance(i, Utterance):
#         continue
#     if i.alignment is None:
#         continue
#     start,end = i.alignment
#     if start is None or end is None:
#         continue

#     for (a,b),_,speaker in res.itertracks(yield_label=True):
#         speaker_id = int(speaker.split("_")[-1])
#         tier = tiers.get(speaker_id)
#         # we set the end time of the utterance as the
#         # *LAST* segment it ends before
#         # i.e. [seg_end, ....., ut_end]
#         # like that 
#         if b <= end/1000 and tier:
#             i.tier = tier

