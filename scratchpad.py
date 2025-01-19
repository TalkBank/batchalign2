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

# from batchalign.models.utterance import infer

# engine = infer.BertUtteranceModel("talkbank/CHATUtterance-zh_CN")
# engine("我 现在 想 听 你说 一些 你 自己 经 历 过 的 故 事 好不好 然后 呢 我们 会 一起 讨 论 有 六 种 不同 的 情 景 然后 在 每 一个 情 景 中 都 需要 你 去 讲 一个 关 于 你 自己 的 一个 故 事 小 故 事")

# doc = Document.new(media_path="/Users/houjun/Downloads/trial.mp3", lang="zho")
# print(doc)
# pipe = BatchalignPipeline.new("asr", lang="zho", num_speakers=2, engine="rev")
# res = pipe(doc)

# # with open("schema.json", 'w') as df:
# #     json.dump(Document.model_json_schema(), df, indent=4)

# ########### The Batchalign Core Test Harness ###########
from batchalign.formats.chat.parser import chat_parse_utterance

# print(str(CHATFile(doc=ut)))
# doc = CHATFile(path="../talkbank-alignment/input/barry.cha").doc
# doc[3][0]
# て
# print(str(CHATFile(doc=res)))

                
# # j.coref_chains) for j in i.words] for i in coref_chains]
# # dir(coref_chains[0][0][1][0])
# # coref_chains[0][0][1][0].chain.index
# # coref_chains[0][0][1][0].is_start
# # coref_chains[0][0][1][0].is_end

# # coref_chains[0].word







# # ng = NgramRetraceEngine()
# # # disf = DisfluencyReplacementEngine()
# doc = Document.new("I am a very large chicken indeed.", lang="eng")

# forms, delim = chat_parse_utterance("I am a very large chicken indeed.", None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)
# gold = Document(content=[utterance], langs=["eng"])

# pipeline = BatchalignPipeline(EvaluationEngine())
# result = pipeline(doc, gold=gold)

# # pipeline = BatchalignPipeline.new("morphosyntax")
# # result2 = pipeline(gold)

# # print(str(CHATFile(doc=result2)))


# result
# print(result["diff"])


# # # # doc[0].content[4].text = "maman,"
# # # # doc[0].content[5].text = "maman,"
# # pipe = BatchalignPipeline(ng, disf)
# # tmp = pipe(doc)
# # tmp


# # tmp[0].content


# # import stanza
# # nlp = stanza.Pipeline("en", packages="tokenize,pos,constituency")
# # tree = nlp("I am dead, and he is also dead, but Josh isn't dead because \"only the brave die young and he knows the best is yet to come\".").sentences[0].constituency
# # i am dead and he is also dead but Josh isn't dead because only the brave die young and he knows the best is yet to come.

# # parse_tree(tree)


# # def rollout_to_leaf(tree):
# #     """Extract the leaf nodes from a subtree via dfs"""

# #     try:
# #         children = tree.children
# #     except AttributeError:
# #         breakpoint()
# #     leafs = []

# #     for c in children:
# #         if c.is_leaf():
# #さっき[* s] 食べた事ある.             leafs.append(c.label)
# #         else:
# #             leafs += rollout_to_leaf(c)

# #     return leafs

# from batchalign.models import BertUtteranceModel
# from batchalign.pipelines import BatchalignPipeline
# pipe = BatchalignPipeline.new("morphosyntax", "fra")
# res = pipe(ut)
# print(str(CHATFile(doc=res)))


# tmp[-1].content
# tmp[-1]
# tmp[6]

# tmp
# tmp1 = sue(tmp)
# # tmp1
# # tmp


# # control x . oh yeah nine x . so we're on this side .
# pipe = BatchalignPipeline.new("asr", num_speakers=2, asr="whisper", lang="spa")
# uts = Document.new(media_path="../talkbank-alignment/test_harness/input/1576.mp3")

# # pipe(uts)

# # tmp = pipe("../talkbank-alignment/test_harness/input/crop.mp3")
# # tmp[8]

# # CHATFile(doc=tmp).write("../talkbank-alignment/test_harness/input/crop.cha")


# ut = BertUtteranceModel("../talkbank-alignment/train/models/utterance/mandarin/utterance_mandarin")

# ut("早上好 中国 我在吃一个 冰淇淋 这个 冰淇淋 很 好吃 但是 我不知道")

# text = "ice ice cream ice cream"

# function = "morphosyntax"
# lang = "jpn"
# num_speakers = 1


# ut = "Swimming is really fun."

# forms, delim = chat_parse_utterance(ut, None, None, None, None)
# # forms
# utterance = Utterance(content=forms, delim=delim, text=ut)

# sec = "vidiš (š)to sam lepo@d našalala [: našarala] ."

# forms, delim = chat_parse_utterance(sec, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim, text=sec)

# utterance[3].time=(1500,1600)
# ut = Document(content=[utterance], langs=["eng"])

# print(str(CHATFile(doc=ut)))


# # # # # =======
# ut = Document(content=[utterance], langs=["jpn"])

# pipeline = BatchalignPipeline.new("morphosyntax", lang="jpn")
# res = pipeline(ut, retokenize=True)


# >>>>>>> theirs


# pipeline = BatchalignPipeline.new("morphosyntax", lang="jpn")
# res = pipeline(ut, retokenize=True)
# >>>>>>> Stashed changes

# # print(str(CHA
# print(str(CHATFile(doc=ut)))
# ut = Document(content=[utterance], langs=[lang])

# pipeline = BatchalignPipeline.new("morphosyntax", lang=lang)
# res = pipeline(ut, retokenize=True)

# print(str(CHATFile(doc=res)))

# print(u"\u202bwhat up with that?")
# print("אויתאויונסתהאויסו".encode().decode("").encode().decode("utf-8"))

########### The Batchalign Individual Engine Harness ###########

# text = "We should be friends! Yes we should."
# # text = "ice ice cream ice ice cream ice ice cream"

# # ice ice cream ice cream
# # ice [/] <ice cream> [/] ice cream
# # ice cream ice cream ice cream ice ice cream cream

# lang = "eng"

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)
# ut = Document(content=[utterance], langs=[lang])

# doc = Document.new(text, lang=lang)

# retrace = StanzaEngine()
# pipe = BatchalignPipeline(retrace)

# doc = pipe(doc)
# doc

# # # doc[0].content

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



