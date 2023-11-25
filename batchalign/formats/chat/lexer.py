import re
from enum import Enum

from batchalign.formats.chat.utils import *
from batchalign.constants import *

class ULTokenType(Enum):
    REGULAR = 0 # hello
    RETRACE = 1 # <I am I am> [/] 
    FEAT = 2 # (.)
    FP = 3 # &-uh
    ANNOT = 4 # &~ject &~head
    MORPUNCT = 5 # ‡„,
    CORRECTION = 6 # test [= test]

class UtteranceLexer:

    def __init__(self, raw):
        self.raw = raw
        self.__iter = iter(raw.strip())
        self.__forms = []
        self.parse()

    @property
    def forms(self):
        return self.__forms

    def __get_until(self, end_tokens=[' ']):
        has_read_nonempty = False
        tokens = []
        while True:
            tok = next(self.__iter, None)
            if tok in end_tokens and has_read_nonempty == False:
                pass
            elif tok != None and tok not in end_tokens:
                tokens.append(tok)
                has_read_nonempty = True
            else:
                break


        return "".join(tokens), len(tokens), tok

    def __pull(self):
        form, num, delim = self.__get_until()

        if num == 0:
            return False
        if form[:2] == "&-":
            self.__forms.append((annotation_clean(form), ULTokenType.FP))
        elif form[:1] == "&":
            self.__forms.append((form, ULTokenType.ANNOT))
        elif form[0] == "<":
            self.handle_group(form)
        elif form[0] == "[":
            self.handle_group(form, ending="]")
        elif form.strip() in MOR_PUNCT:
            self.__forms.append((form.strip(), ULTokenType.MORPUNCT))
        elif form.strip() in REPEAT_GROUP_MARKS:
            self.__forms.append((self.__forms.pop(-1)[0], ULTokenType.RETRACE))
            self.__forms.append((form.strip(), ULTokenType.FEAT))
        elif annotation_clean(form).strip() == "":
            self.__forms.append((form, ULTokenType.FEAT))
        else:
            self.__forms.append((annotation_clean(form).strip(), ULTokenType.REGULAR))

        return form

    def handle_group(self, form, ending=">"):
        forms = [annotation_clean(form)]

        # pull the form
        try:
            while form[-1] != ending:
                form, num, delim = self.__get_until()
                if form == None:
                    raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(forms)}")
                forms.append(annotation_clean(form))
        except IndexError:
            raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(forms)}")

        # pull the type
        if ending == ">":
            form, num, delim = self.__get_until()
            if form.strip() in REPEAT_GROUP_MARKS:
                for i in forms:
                    self.__forms.append((i, ULTokenType.RETRACE))
                self.__forms.append((form.strip(), ULTokenType.FEAT))
            else:
                raise CHATValidationException(f"Lexer failed! Unexpected group type mark. On line: '{self.raw}', parsed: {form.strip()}")
        elif ending == "]":
            for i in forms:
                self.__forms.append((i, ULTokenType.CORRECTION))

    def parse(self):

        while True:
            res = self.__pull()
            if res == False or res[-1] in ENDING_PUNCT:
                break

def lex(utterance):
    ut = UtteranceLexer(utterance)
    return ut.forms
