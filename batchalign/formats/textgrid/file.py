from batchalign.document import *
from batchalign.utils import *
from batchalign.errors import *
from batchalign.constants import *

from batchalign.formats.base import BaseFormat
from batchalign.formats.textgrid.parser import *
from batchalign.formats.textgrid.generator import *

from praatio import textgrid
from praatio.utilities.textgrid_io import  getTextgridAsStr

import os
import re

from glob import glob
from pathlib import Path


class TextGridFile(BaseFormat):
    """TextGrid File

    Reads and handles a TextGrid file into Batchalign document format. Notably,
    there are two ways a TextGrid can be written, and the distinction between
    them could be ambiguous. In particular: a TextGrid file use intervals as
    words (default?), or utterances. Hence, the TextGrid file forces the user
    to specify this.

    Attributes
    ----------
    doc : Document
        The Batchalign document representing the file.

    Parameters
    ----------
    style : str
        'word', or 'utterance': whether each interval is a
        word or an utterance.
    path : Optional[str]
        The path of the TextGrid file to load from.
    doc : Optional[Document]
        Batchalign document to initialize a TextGrid file from.
    tg : Optional[TextGrid]
        The TextGrid file
    lang : Optional[str]
        The language of the TextGrid, 3 letters. 
    corpus_name : Optional[TextGrid]
        The TextGrid file


    Notes
    -----
    To initlize the class, choose one of two formats to seed the
    document. Either provide a path to a TextGrid file, a Batchalign
    Document, or a TextGrid object.

    Examples
    --------
    >>> c = TextGridFile(style="word", path="./extern/test.TextGrid")
    >>> transcript = c.doc.transcript()
    """

    def __init__(self, style, path=None, doc=None, tg=None, lang="eng", corpus_name="corpus_name"):
        if path == None and doc == None and tg == None:
            raise ValueError("No info about this TextGrid is provided! Provide one of `doc`, `path`, or `tg`.")

        if path:
            tg = textgrid.openTextgrid(path, False)
        if doc == None:
            doc = load_textgrid(tg, lang, corpus_name, style == "word")

        self.__doc = doc
        self.__style = style

    def write(self, path):
        """Write the CHATFile to file.

        Parameters
        ----------
        path : str
            Path of where the CHAT file should get written.
        """

        tg = dump_textgrid(self.__doc, self.__style == "word")
        tg.save(path, format="long_textgrid", includeBlankSpaces=True)

    @staticmethod
    def __tgToDictionary(tg):
        tiers = []
        for tier in tg.tiers:
            tierAsDict = {
                "class": tier.tierType,
                "name": tier.name,
                "xmin": tier.minTimestamp,
                "xmax": tier.maxTimestamp,
                "entries": tier.entries,
            }
            tiers.append(tierAsDict)

        return {"xmin": tg.minTimestamp, "xmax": tg.maxTimestamp, "tiers": tiers}

    @property
    def doc(self):
        return self.__doc

    def __str__(self):
        tg = dump_textgrid(self.__doc, self.__style == "word")
        tgAsDict = self.__tgToDictionary(tg)
        return getTextgridAsStr(tg=tgAsDict, format='long_textgrid', includeBlankSpaces=True)

