from batchalign.document import *
from batchalign.utils import *
from batchalign.errors import *
from batchalign.constants import *

from batchalign.formats.base import BaseFormat
from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import lex, ULTokenType
from batchalign.formats.chat.parser import *

import re

# FILE = "./extern/minga01a.cha"

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
        elif lines:
            self.__doc = chat_parse_doc(lines)
        else:
            self.__doc = doc

    @property
    def doc(self):
        return self.__doc

