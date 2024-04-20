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

########### The Batchalign Core Test Harness ###########
# from batchalign.formats.chat.parser import chat_parse_utterance

# ng = NgramRetraceEngine()
# disf = DisfluencyReplacementEngine()
# doc = Document.new("um I'm seeing I'm seeing light and dark to create uh uh to create uh to create uh time", lang="eng")
# pipe = BatchalignPipeline(ng, disf)
# tmp = pipe(doc)
# tmp


# tmp[0].content


# import stanza
# nlp = stanza.Pipeline("en", packages="tokenize,pos,constituency")
# tree = nlp("I am dead, and he is also dead, but Josh isn't dead because \"only the brave die young and he knows the best is yet to come\".").sentences[0].constituency
# i am dead and he is also dead but Josh isn't dead because only the brave die young and he knows the best is yet to come.

# parse_tree(tree)


# def rollout_to_leaf(tree):
#     """Extract the leaf nodes from a subtree via dfs"""

#     try:
#         children = tree.children
#     except AttributeError:
#         breakpoint()
#     leafs = []

#     for c in children:
#         if c.is_leaf():
#             leafs.append(c.label)
#         else:
#             leafs += rollout_to_leaf(c)

#     return leafs

# from batchalign.models import BertUtteranceModel
# tmp = CHATFile(path="./extern/Untitled.cha").doc
# tmp
# tmp1 = sue(tmp)
# tmp1
# tmp


# pipe = BatchalignPipeline.new("asr", num_speakers=2, asr="whisper", lang="spa")
# tmp = pipe("../talkbank-alignment/test_harness/input/Untitled.wav")
# tmp

# CHATFile(doc=tmp).write("./extern/Untitled.cha")


# ut = BertUtteranceModel("../talkbank-alignment/train/models/utterance/mandarin/utterance_mandarin")

# ut("早上好 中国 我在吃一个 冰淇淋 这个 冰淇淋 很 好吃 但是 我不知道")

# text = "ice ice cream ice cream"

# function = "morphosyntax"
# lang = "cym"
# num_speakers = 1

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)

# ut = Document(content=[utterance], langs=[lang])

# pipeline = BatchalignPipeline.new(function, lang=lang)
# res = pipeline(ut)

# print(str(CHATFile(doc=res)))


########### The Batchalign Individual Engine Harness ###########

# text = "I love chicken pie I love chicken pie I love chicken pie "
# text = "ice ice cream ice ice cream ice ice cream"

# ice ice cream ice cream
# ice [/] <ice cream> [/] ice cream
# ice cream ice cream ice cream ice ice cream cream

# lang = "eng"

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)
# ut = Document(content=[utterance], langs=[lang])

# retrace = NgramRetraceEngine()
# pipe = BatchalignPipeline(retrace)

# doc = pipe(ut)
# # doc[0].content

# print(str(CHATFile(doc=doc)))

########### The Batchalign String Test Harness ###########
# from batchalign.formats.chat.parser import chat_parse_utterance
 
# file = "/Users/houjun/Documents/Projects/talkbank-alignment/test_harness/input/Untitled.wav"

# function = "asr"
# lang = "spa"
# num_speakers = 1

# ut = Document.new(media_path=file, lang=lang)

# pipeline = BatchalignPipeline.new(function, lang=lang, num_speakers=num_speakers)
# doc = pipeline(ut)
# doc.content
# # doc[0][-1]
# # doc[0][-2].model_dump()

# # doc[0].content[-2]

# print(str(CHATFile(doc=doc)))

########### The Batchalign Parser Harness ###########
# from batchalign.formats.chat import CHATFile

# in_dir = "../talkbank-alignment/test_harness/input"

# in_files = glob(str(Path(in_dir)/"*.cha"))
# parent, _, files = zip(*list(os.walk(in_dir)))

# in_files = [os.path.join(a,i)
#             for a,b in zip(parent, files)
#             for i in b
#             if ".cha" in i]

# for file in in_files:
#     try:
#         f = CHATFile(path=os.path.abspath(file))
#     except Exception as e:
#         print(file)
#         raise e

# f.doc[12]

# d = f.doc
# wer = EvaluationEngine()
# wer(d_alt, gold=d)

# dp time
# f.doc[1][1]
# f.doc.media

# doc = f.doc
# doc[-382][1]

########### The Batchalign CLI Harness ###########
# from batchalign.cli.dispatch import _dispatch

# in_dir = "../talkbank-alignment/test_harness/input/"
# out_dir = "../talkbank-alignment/test_harness/output/"
# in_format = "cha"

# function = "morphotag"
# lang = "hrv"
# num_speakers = 1

# class Context:
#     obj = {"verbose": 3}

# def loader(file):
#     return CHATFile(path=os.path.abspath(file)).doc

#     # return file

# def writer(doc, output):
#     CHATFile(doc=doc).write(output)
#     # CHATFile(doc=doc).write(output
#     #                         .replace(".wav", ".cha")
#     #                         .replace(".mp4", ".cha")
#     #                         .replace(".mp3", ".cha"))

# _dispatch(function, lang, num_speakers, [in_format], Context(),
          
#             in_dir, out_dir,
#             loader, writer, Console())

########## The Batchalign CHAT Test Tarness ##########

# from batchalign.formats.chat.parser import chat_parse_utterance
# from batchalign.formats.chat.lexer import lex

# from batchalign.formats.chat.utils import annotation_clean

# main = "+\" and then some Indians came and took something away from him and he said +\"/. [+ dia] •884995_892418•"
# mor = None
# gra = None

# chat_parse_utterance(main, mor, gra, None, None)



