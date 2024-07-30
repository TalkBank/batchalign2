from batchalign.document import *
from batchalign.utils import *
from batchalign.errors import *
from batchalign.constants import *

from batchalign.formats.base import BaseFormat
from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import lex
from batchalign.formats.chat.parser import *
from batchalign.formats.chat.generator import *

import os
import re

from glob import glob
from pathlib import Path


class CHATFile(BaseFormat):
    """CHAT File

    Reads and handles a CHAT file into Batchalign document format.

    Attributes
    ----------
    doc : Document
        The Batchalign document representing the file.

    Parameters
    ----------
    path : Optional[str]
        The path of the CHAT file to load from.
    doc : Optional[Document]
        Batchalign document to initialize a CHAT file from.
    doc : Optional[List[str]]
        The lines of the files to load from.

    Notes
    -----
    To initlize the class, choose one of three formats to seed the
    document. Either provide a path to a CHAT file, a Batchalign
    Document, or lines to the document.

    Examples
    --------
    >>> c = CHATFile("./extern/test.cha")
    >>> transcript = c.doc.transcript()
    """

    def __init__(self, path=None, doc=None, lines=None, special_mor_=False):

        self.__special_mor = special_mor_

        if path:
            # read in the resulting file
            with open(path, "r", encoding="utf-8") as df:
                # get alignment result
                data = df.readlines()
            # conform result with tab-seperated beginnings
            raw = []
            # for each value, if its a tab seperated beginning, we
            # concate it with the previous line
            for value in data:
                if value[0] == "\t":
                    res = raw.pop()
                    res = res.strip("\n") + " " + value[1:]
                    raw.append(res)
                else:
                    raw.append(value)

            self.__doc = chat_parse_doc(raw, special_mor=special_mor_)
            # media file auto-associate
            if self.__doc.media != None:
                name = self.__doc.media.name
                dir = os.path.dirname(path)
                globs = [os.path.join(dir, i) for i in PARSABLE_MEDIA]

                # try to find the media file
                media_files_glob = sum([glob(i) for i in globs], [])
                # filter for those whose basename matches
                media_files = [i for i in media_files_glob if os.path.splitext(os.path.basename(i))[0] == name]
                # if we have no match, go by file names instead
                if len(media_files) == 0:
                    media_files = [i for i in media_files_glob
                                   if os.path.splitext(os.path.basename(i))[0] == os.path.splitext(os.path.basename(path))[0]]

                # associate
                if len(media_files) > 0:
                    self.__doc.media.url = media_files[0]
        elif lines:
            self.__doc = chat_parse_doc(lines, special_mor=special_mor_)
        else:
            self.__doc = doc


    def write(self, path, write_wor=True):
        """Write the CHATFile to file.

        Parameters
        ----------
        path : str
            Path of where the CHAT file should get str.
        """
        
        str_doc = self.__generate(self.__doc, self.__special_mor, write_wor=write_wor)

        with open(path, 'w', encoding="utf-8") as df:
            df.write(str_doc)

    @staticmethod
    def __generate(doc:Document, special=False, write_wor=True):
        utterances = doc.content

        def __get_birthdays(line):
            return isinstance(line, CustomLine) and "birth" in line.id.lower()

        pid = "" if doc.pid in [None, ""] else f"@PID:\t{doc.pid}\n"
        main = [f"@UTF8\n{pid}@Begin", generate_chat_preamble(doc, filter(__get_birthdays,
                                                                    utterances))]
        for i in utterances:
            if isinstance(i, CustomLine):
                if "birth" not in i.id.lower():
                    extra = f"@{i.id}"
                    if i.content != None:
                        extra += ":\t"+i.content
                    main.append(extra.strip())
            else:
                main.append(generate_chat_utterance(i, special and doc.langs[0] == "eng",
                                                    write_wor=write_wor))
        main.append("@End\n")

        raw = "\n".join(main)

        # correct for unicode problems
        corrected = raw
        corrected = corrected.replace(u"\u202b", u"\u200f")

        return corrected

    @property
    def doc(self):
        return self.__doc

    def __str__(self):
        return self.__generate(self.__doc, self.__special_mor)

