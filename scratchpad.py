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
from batchalign.formats.chat.parser import chat_parse_utterance

# (.*)([a-z]) ([a-z])
# \t\1\2\3

# text = "啊 哈哈哈哈 是的 t h i s i s a t h i n g 啊哈哈哈哈 是的 ."
# langs = ["yue"]
# parsed, delim = chat_parse_utterance(text, None, None, None, None)
# ut = Utterance(content=parsed, delim=delim, text=text)
# doc = Document(content=[ut], langs=langs)

# doc

# print(str(CHATFile(doc=doc)))

# pipe = BatchalignPipeline.new("morphosyntax", lang="jpn")
# doc_out = pipe(doc, retokenize=True)




# doc = CHATFile(path="../talkbank-alignment/input/Untitled.cha").doc



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

