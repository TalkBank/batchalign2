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

    def __init__(self, path=None, doc=None, lines=None):
        if path:
            # read in the resulting file
            with open(path, "r") as df:
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

            self.__doc = chat_parse_doc(raw)
            # media file auto-associate
            if self.__doc.media != None:
                name = self.__doc.media.name
                dir = os.path.dirname(path)
                globs = [os.path.join(dir, i) for i in MEDIA_EXTENSIONS]

                # try to find the media file
                media_files = sum([glob(i) for i in globs], [])

                # associate
                if len(media_files) > 0:
                    self.__doc.media.url = media_files[0]
        elif lines:
            self.__doc = chat_parse_doc(lines)
        else:
            self.__doc = doc

    def write(self, path):
        """Write the CHATFile to file.

        Parameters
        ----------
        path : str
            Path of where the CHAT file should get str.
        """
        
        str_doc = self.__generate(self.__doc)

        with open(path, 'w') as df:
            df.write(str_doc)

    @staticmethod
    def __generate(doc:Document):
        utterances = doc.content

        def __get_birthdays(line):
            return isinstance(line, CustomLine) and "birth" in line.id.lower()

        main = ["@UTF8\n@Begin", generate_chat_preamble(doc, filter(__get_birthdays,
                                                                    utterances))]
        for i in utterances:
            if isinstance(i, CustomLine):
                if "birth" not in i.id.lower():
                    extra = f"@{i.id}"
                    if i.content != None:
                        extra += ":\t"+i.content
                    main.append(extra.strip())
            else:
                main.append(generate_chat_utterance(i))
        main.append("@End\n")

        return "\n".join(main)

    @property
    def doc(self):
        return self.__doc

    def __str__(self):
        return self.__generate(self.__doc)

