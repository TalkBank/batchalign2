import re
from enum import Enum

from ...utils import *
from ...constants import *

class UPTokenType(Enum):
    REGULAR = 0 # hello
    RETRACE = 1 # <I am I am> [/] 
    FEAT = 2 # (.)
    FP = 3 # &-uh
    ANNOT = 4 # &~ject &~head

class UtteranceParser:

    def __init__(self, raw):
        self.raw = raw
        self.__iter = iter(raw)
        self.__forms = []
        self.parse()

    @property
    def forms(self):
        return self.__forms

    def __get_until(self, end_tokens=[' ']):
        tokens = []
        while True:
            tok = next(self.__iter, None)
            if tok != None and tok not in end_tokens:
                tokens.append(tok)
            else:
                break


        return "".join(tokens), len(tokens), tok

    def __pull(self):
        form, num, delim = self.__get_until()
        if num == 0:
            return False
        if form[:2] == "&-":
            self.__forms.append((annotation_clean(form), UPTokenType.FP))
        elif form[:1] == "&":
            self.__forms.append((form, UPTokenType.ANNOT))
        elif form[0] == "<":
            self.handle_group(form)
        elif form.strip() == "[/]":
            self.__forms.append((self.__forms.pop(-1)[0], UPTokenType.RETRACE))
            self.__forms.append(("[/]", UPTokenType.FEAT))
        elif annotation_clean(form).strip() == "":
            self.__forms.append((form, UPTokenType.FEAT))
        else:
            self.__forms.append((annotation_clean(form).strip(), UPTokenType.REGULAR))

        return form

    def handle_group(self, form):
        forms = [annotation_clean(form)]

        # pull the form
        while form[-1] != ">":
            form, num, delim = self.__get_until()
            if form == None:
                raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(forms)}")
            forms.append(annotation_clean(form))

        # pull the type
        form, num, delim = self.__get_until()

        if form.strip() == "[/]":
            for i in forms:
                self.__forms.append((i, UPTokenType.RETRACE))
            self.__forms.append(("[/]", UPTokenType.FEAT))
        else:
            raise CHATValidationException(f"Lexer failed! Unexpected group type mark. On line: '{self.raw}', parsed: {form.strip()}")

    def parse(self):

        while True:
            res = self.__pull()
            if res == False or res[-1] in ENDING_PUNCT:
                break

def parse(utterance):
    ut = UtteranceParser(utterance)
    return ut.forms
