from batchalign import *
import json
from glob import glob
from pathlib import Path
from rich.console import Console
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
 
# text = "I'm going to read some random crap as I see on the screen . 2530_4940"

# function = "morphosyntax"
# lang = "eng"
# num_speakers = 1

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)

# # utterance = Utterance(content=text)

# ut = Document(content=[utterance], langs=[lang])

# pipeline = BatchalignPipeline.new(function, lang=lang, num_speakers=num_speakers)
# doc = pipeline(ut)
# doc[0][-1]
# doc[0][-2].model_dump()

# doc[0].content[-2]

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


# pipeline = BatchalignPipeline.new("fa", lang="eng", num_speakers=len(doc.tiers))
# doc = pipeline(doc)


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
# lang = "fra"
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

# main = "+< <太 高 了> [/] 太 高 了 . •125000_126823•"
# mor = None
# gra = None

# chat_parse_utterance(main, mor, gra, None, None)



