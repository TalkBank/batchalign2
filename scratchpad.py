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

### The Batchalign Test Harness ###
from batchalign.formats.chat.parser import chat_parse_utterance

# text = "est ce que la +/."

# function = "morphosyntax"
# lang = "fra"
# num_speakers = 1

# forms, delim = chat_parse_utterance(text, None, None, None, None)
# utterance = Utterance(content=forms, delim=delim)

# ut = Document(content=[utterance], langs=[lang])
# pipeline = BatchalignPipeline.new(function, lang_code=lang, num_speakers=num_speakers)

# doc = pipeline(ut)

### The Batchalign CLI Harness ###
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

### The Batchalign CHAT Test Tarness ####
# from batchalign.formats.chat.parser import chat_parse_utterance

# main = "xxx; crocodile , crocodile ."
# mor = "noun|crocodile&Masc cm|cm noun|crocodile&Masc ."
# gra = "1|3|ROOT 2|3|PUNCT 3|1|NMOD 4|1|PUNCT"

# chat_parse_utterance(main, mor, gra, None, None)

