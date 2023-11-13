import pytest

from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import *
from batchalign.document import *
from batchalign.errors import *

# checks for extra spaces, single and double repetation, and filed pauses
STANDARD_LEX_STR = "   Um <I like I like> [/] I like &-um beans [//] beans         .   "
STANDARD_LEX = [('Um', ULTokenType.REGULAR), ('I', ULTokenType.RETRACE), ('like', ULTokenType.RETRACE), ('I', ULTokenType.RETRACE), ('like', ULTokenType.RETRACE), ('[/]', ULTokenType.FEAT), ('I', ULTokenType.REGULAR), ('like', ULTokenType.REGULAR), ('um', ULTokenType.FP), ('beans', ULTokenType.RETRACE), ('[//]', ULTokenType.FEAT), ('beans', ULTokenType.REGULAR), ('.', ULTokenType.FEAT)]
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




