import pytest

from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import *
from batchalign.document import *
from batchalign.errors import *

# checks for extra spaces, single and double repetation, and filed pauses
STANDARD_LEX_STR = "   Um <I like I like> [/] I like &-um beans [//] beans         .   "
STANDARD_LEX = [('Um', TokenType.REGULAR), ('I', TokenType.RETRACE), ('like', TokenType.RETRACE), ('I', TokenType.RETRACE), ('like', TokenType.RETRACE),  ('I', TokenType.REGULAR), ('like', TokenType.REGULAR), ('um', TokenType.FP), ('beans', TokenType.RETRACE), ('beans', TokenType.REGULAR), ('.', TokenType.PUNCT)]

def test_lex():
    assert lex(STANDARD_LEX_STR) == STANDARD_LEX
