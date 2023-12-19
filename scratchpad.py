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

########### The Batchalign Test Harness ###########
# from batchalign.formats.chat.parser import chat_parse_utterance

# text = "est ce que la +/."

# function = "morphosyntax"
# lang = "fra"
# num_speakers = 1

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)

# ut = Document(content=[utterance], langs=[lang])
# pipeline = BatchalignPipeline.new(function, lang_code=lang, num_speakers=num_speakers)

# doc = pipeline(ut, type=5

########### The Batchalign Parser Harness ###########
# from batchalign.formats.chat import CHATFile

# in_dir = "../talkbank-alignment/test_harness/input/"
# out_dir = "../talkbank-alignment/test_harness/output/"

# in_files = glob(str(Path(in_dir)/"*.cha"))

# for file in in_files:
#     try:
#         CHATFile(path=os.path.abspath(file))
#     except Exception as e:
#         print(file)
#         raise e

########### The Batchalign CLI Harness ###########
# from batchalign.cli.dispatch import _dispatch

# in_dir = "../talkbank-alignment/test_harness/input/"
# out_dir = "../talkbank-alignment/test_harness/output/"
# in_files = glob(str(Path(in_dir)/"*.cha"))

# function = "morphotag"
# lang = "fra"
# num_speakers = 1

# class Context:
#     obj = {"verbose": 3}

# def loader(file):
#     try:
#         return CHATFile(path=os.path.abspath(file)).doc
#     except:
#         breakpoint()

#     # return file

# def writer(doc, output):
#     CHATFile(doc=doc).write(output)
#     # CHATFile(doc=doc).write(output
#     #                         .replace(".wav", ".cha")
#     #                         .replace(".mp4", ".cha")
#     #                         .replace(".mp3", ".cha"))

# _dispatch(function, lang, num_speakers, in_files, Context(),
#             in_dir, out_dir,
#             loader, writer, Console())

########## The Batchalign CHAT Test Tarness ##########
# from batchalign.formats.chat.parser import chat_parse_utterance

# main = "ah oui, hein , ça c'est chic ."
# mor = "intj|ah intj|oui cm|cm intj|hein cm|cm pron|ça-Dem-Acc-S3 pron|ce-Dem-Acc-S3~aux|être-Fin-Ind-3-Pres adj|chic-Pos-S-1 ."
# gra = "1|9|DISCOURSE 2|9|DISCOURSE 3|2|PUNCT 4|9|DISCOURSE 5|4|PUNCT 6|9|DISLOCATED 7|9|NSUBJ 8|9|COP 9|9|ROOT 10|9|PUNCT"

# chat_parse_utterance(main, mor, gra, None, None)

