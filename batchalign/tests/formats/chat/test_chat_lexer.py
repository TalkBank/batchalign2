import pytest

from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import *
from batchalign.document import *
from batchalign.errors import *

# checks for extra spaces, single and double repetation, and filed pauses
STANDARD_LEX_STR = "   Um <I like I like> [/] I like &-um beans [//] beans         .   "
STANDARD_LEX = [('Um', TokenType.REGULAR), ('I', TokenType.RETRACE), ('like', TokenType.RETRACE), ('I', TokenType.RETRACE), ('like', TokenType.RETRACE), ('[/]', TokenType.FEAT), ('I', TokenType.REGULAR), ('like', TokenType.REGULAR), ('um', TokenType.FP), ('beans', TokenType.RETRACE), ('[//]', TokenType.FEAT), ('beans', TokenType.REGULAR), ('.', TokenType.FEAT)]
# checks for groups that don't end, with the wrong group spec, or with extra spaces
BROKEN_GROUP_LEX_STR1 = "   Um <I like I like >"
BROKEN_GROUP_LEX_STR2 = "   Um <I like I like"
BROKEN_GROUP_LEX_STR3 = "   Um <I like I like> [chickens]"

def test_lex():
    assert lex(STANDARD_LEX_STR) == STANDARD_LEX
def test_lex_broken_group():
    with pytest.raises(CHATValidationException):
        lex(BROKEN_GROUP_LEX_STR1)
    with pytest.raises(CHATValidationException):
        lex(BROKEN_GROUP_LEX_STR2)
    with pytest.raises(CHATValidationException):
        lex(BROKEN_GROUP_LEX_STR3)




