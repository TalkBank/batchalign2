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

# text = "c'était deux mots différents ."

# function = "morphosyntax"
# lang = "fra"
# num_speakers = 1

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)

# # utterance = Utterance(content=text)

# ut = Document(content=[utterance], langs=[lang])
# pipeline = BatchalignPipeline.new(function, lang_code=lang, num_speakers=num_speakers)

# doc = pipeline(ut)

# print(str(CHATFile(doc=doc)))

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
from batchalign.cli.dispatch import _dispatch

in_dir = "../talkbank-alignment/test_harness/input/"
out_dir = "../talkbank-alignment/test_harness/output/"
in_files = glob(str(Path(in_dir)/"*.cha"))

function = "morphotag"
lang = "fra"
num_speakers = 1

class Context:
    obj = {"verbose": 3}

def loader(file):
    try:
        return CHATFile(path=os.path.abspath(file)).doc
    except:
        breakpoint()

    # return file

def writer(doc, output):
    CHATFile(doc=doc).write(output)
    # CHATFile(doc=doc).write(output
    #                         .replace(".wav", ".cha")
    #                         .replace(".mp4", ".cha")
    #                         .replace(".mp3", ".cha"))

_dispatch(function, lang, num_speakers, in_files, Context(),
            in_dir, out_dir,
            loader, writer, Console())

########## The Batchalign CHAT Test Tarness ##########
# from batchalign.formats.chat.parser import chat_parse_utterance

# main = "ben c'est bon [?] ."
# mor = "intj|ben pron|ce-Dem-Acc-S3~aux|être-Fin-Ind-3-Pres adj|bon-Pos-S-1 ."
# gra = "1|4|DISCOURSE 2|4|NSUBJ 3|4|COP 4|4|ROOT 5|4|PUNCT"

# chat_parse_utterance(main, mor, gra, None, None)

