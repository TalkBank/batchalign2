import re
from enum import Enum

from batchalign.document import *
from batchalign.formats.chat.utils import *
from batchalign.constants import *

import warnings


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
            self.__forms.append((annotation_clean(form), TokenType.FP))
        elif form in ENDING_PUNCT:
            self.__forms.append((form, TokenType.PUNCT))
        elif form[:1] == "&":
            self.__forms.append((form, TokenType.ANNOT))
        elif form[0] == "<":
            self.handle_group(form, ending=">")
        elif form.strip() in REPEAT_GROUP_MARKS:
            self.__forms.append((self.__forms.pop(-1)[0], TokenType.RETRACE))
            self.__forms.append((form.strip(), TokenType.FEAT))
        elif form[0] == "[":
            self.handle_group(form, ending="]")
        elif form.strip() in MOR_PUNCT:
            self.__forms.append((form.strip(), TokenType.PUNCT))
        elif annotation_clean(form).strip() == "":
            self.__forms.append((form, TokenType.FEAT))
        elif annotation_clean(form).strip() in CHAT_IGNORE:
            self.__forms.append((annotation_clean(form).strip(), TokenType.ANNOT))
        else:
            self.__forms.append((annotation_clean(form).strip(), TokenType.REGULAR))

        return form

    def handle_group(self, form, ending=">"):
        initial_form = form
        forms = []
        if annotation_clean(form) != "": 
            forms.append(annotation_clean(form))

        # pull the form
        try:
            while ending not in form:
                form, num, delim = self.__get_until()
                if form == None or num == 0:
                    raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(forms)}")
                if annotation_clean(form).strip() == "":
                    continue
                forms.append(annotation_clean(form))
        except IndexError:
            raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(forms)}")

        # pull the type
        if ending == ">":
            form, num, delim = self.__get_until()
            if form.strip() in REPEAT_GROUP_MARKS:
                for i in forms:
                    self.__forms.append((i, TokenType.RETRACE))
                self.__forms.append((form.strip(), TokenType.FEAT))
            elif form.strip() in NORMAL_GROUP_MARKS:
                for i in forms:
                    self.__forms.append((i, TokenType.REGULAR))
                self.__forms.append((form.strip(), TokenType.FEAT))
            elif len(form) > 0 and form.strip()[0] == "[":
                for i in forms:
                    self.__forms.append((i, TokenType.REGULAR))
                self.handle_group(form, ending="]")
            else:
                raise CHATValidationException(f"Lexer falied! Unexpected group type mark. On line: '{self.raw}', parsed: {form.strip()}")
        elif ending == "]":
            for i in forms:
                self.__forms.append((i, TokenType.CORRECTION))

    def parse(self):

        while True:
            res = self.__pull()
            if res == False or res in ENDING_PUNCT or (res[-1] in ENDING_PUNCT
                                                       and re.findall("\w", res)):
                break

def lex(utterance):
    ut = UtteranceLexer(utterance)
    return ut.forms
