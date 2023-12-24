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
        self.__clauses = []
        self.parse()

    @property
    def forms(self):
        return self.decode(self.__clauses)

    @staticmethod
    def decode(arr):
        # basically, a "special" type overrides a "regular" type.
        # what that means is that, for instance, if we have a parse
        # that asks a bunch of tokens to be special, then they are all
        # special in the way those tokens are asked to be.
        # however, if we have a group that asks the tokens to be regular,
        # then the tokens are free to be whatever they want to be

        decoded = []

        for content, type in arr:
            if isinstance(content, str):
                decoded.append((content, type))
            # if the type is regular, we follow whatever type
            # the sub-clause wants us to be
            elif type == TokenType.REGULAR:
                decoded += UtteranceLexer.decode(content)
            # if the type is irregular, the sub-clause will
            # be whatever the sub-clause's new target is
            else:
                res = UtteranceLexer.decode(content)
                for i,_ in res:
                    decoded.append((i, type))
        return decoded

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


    def __handle(self, form, num, delim): 
        if num == 0:
            return False
        if form[:2] == "&-":
            self.__clauses.append((annotation_clean(form), TokenType.FP))
        elif form in ENDING_PUNCT:
            self.__clauses.append((form, TokenType.PUNCT))
        elif form[:1] == "&":
            self.__clauses.append((form, TokenType.ANNOT))
        elif form[0] == "<":
            self.__clauses.append((self.handle_group(form, type=">"), TokenType.REGULAR))
        elif form.strip() in REPEAT_GROUP_MARKS:
            self.__clauses.append((self.__clauses.pop(-1)[0], TokenType.RETRACE))
            self.__clauses.append((form.strip(), TokenType.FEAT))
        elif form.strip() in NORMAL_GROUP_MARKS:
            self.__clauses.append((form.strip(), TokenType.FEAT))
        elif form[0] == "[" and form[:2] != "[:":
            # we ignore all other things which are simple annotations
            self.handle_group(form, type="]")
        elif form[:2] == "[:":
            self.__clauses.pop(-1)
            self.__clauses.append((self.handle_group(form, type="]"), TokenType.REGULAR))
        elif form.strip() in MOR_PUNCT:
            self.__clauses.append((form.strip(), TokenType.PUNCT))
        elif annotation_clean(form).strip() == "":
            self.__clauses.append((form, TokenType.FEAT))
        elif annotation_clean(form).strip() in CHAT_IGNORE:
            self.__clauses.append((annotation_clean(form).strip(), TokenType.ANNOT))
        else:
            self.__clauses.append((annotation_clean(form).strip(), TokenType.REGULAR))

    def __pull(self):
        form, num, delim = self.__get_until()

        self.__handle(form, num, delim)

        return form

    def __get_group(self, form, type):
        text = ""
        group = [form]

        # scan forward until we have the first actual form, if
        # its a selection group
        if type == ">" and annotation_clean(form) == "":
            form, num, delim = self.__get_until()
            group = [group.pop(0).strip()+annotation_clean(form)]

        # grab forward the entire group 
        while type not in form:
            form, num, delim = self.__get_until()
            if form == None or num == 0:
                raise CHATValidationException(f"Lexer failed! Unexpected end to utterance within form group. On line: '{self.raw}', parsed group: {str(group)}")
            # if form[0] == "<" or form[0] == "[":
            #     group.append(form)
            #     # recursively handle a recursive group
            #     if form[0] == "<":
            #         recurse, form = self.handle_group(form, ">")
            #     elif form[0] == "[":
            #         recurse, form = self.handle_group(form, "]")
            #     for i,_,_ in recurse:
            #         group.append(i)
            #     continue
            group.append(form)
            text += (" "+form)

        # get rid 
        special = [re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']").sub("", i).strip() for i in group]
        words = [re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ']").sub("", i).strip() for i in group
                 if re.compile(r"[^A-Za-zÀ-ÖØ-öø-ÿ']").sub("", i).strip()!= ""]

        return words, special[0], text

    def handle_group(self, form, type=">"):
        orig_form = form

        # scan the group
        words, special, text = self.__get_group(form, type)
        text = form + text

        if len(text.strip()) == 0:
            raise CHATValidationException(f"There is a group with nothing within. Batchalign is really confused. Stopping all parses of this file. line='{self.raw}', group='{text}'")

        # parse the internal text of the group
        # without the outside group boundary
        # if the group is a "selection" i.e. < and >
        if text[0] == "<" and text[-1] == ">":
            lexer = UtteranceLexer(text.strip()[1:-1])
            forms = lexer.forms
        elif text[0] == "[" and text[-1] == "]":
            text = text.strip(special).strip("]").strip()
            lexer = UtteranceLexer(text)
            forms = lexer.forms


        return forms

        # for other groups, we simply need to parse the inside
        # by scanning what it is
        

        

        # if len(forms) == 0:
            # breakpoint()

        # for each type of notation, we parse it individually
        # this is a replacement; it should replace whatever we
        # had parsed before
        # if forms[0][0] == ":" and :

        # breakpoint()

        # # get the next token
        # form, num, delim = self.__get_until()

        # # collect forms to handle
        # to_handle = []

        # # continue handling until we get the final form
        # while len(form) > 0 and form[0] == "[" and form.strip() not in NORMAL_GROUP_MARKS + REPEAT_GROUP_MARKS:
        #     # read any replacements that exist
        #     if form[:2] == "[:":
        #         # this means that "words" need to be replaced
        #         # with whatever is in the *NEXT* group
        #         words, special, _ = self.__get_group(form, type="]")
        #     # in all other cases, we get the next group and ignore it
        #     else:
        #         grp = self.__get_group(form, type="]")
        #     # fetch the next form
        #     form, num, delim = self.__get_until()

        
        # # we just go through all the words and add the type that is appropriate
        # if form in NORMAL_GROUP_MARKS:
        #     for f in words:
        #         to_handle.append((f, num, delim))
        #         # self.__forms.append((annotation_clean(f).strip(), TokenType.REGULAR))
        #     self.__forms.append((form.strip(), TokenType.FEAT))
        # elif form in REPEAT_GROUP_MARKS:
        #     for f in words:
        #         self.__forms.append((annotation_clean(f).strip(), TokenType.RETRACE))
        #     self.__forms.append((form.strip(), TokenType.FEAT))
        # else:
        #     if special[0] != "[" or special[:2] == "[:":
        #         for f in words:
        #             to_handle.append((f, num, delim))
        #     if len(form) == 0:
        #         if special.strip() != "[+":
        #             warnings.warn(f"Lexer failed! Nothing was annotated after a group; we don't know what the group does. We are ignoring this group! On line: '{self.raw}'; parsed: '{words}'")
        #         # otherwise its an utterance end delimiter and we truly don't care
        #         return
        #     if form[0] != "<":
        #         to_handle.append((form, num, delim))
        #     else:
        #         to_handle += self.handle_group(form, type=">")

        # print("HANDLING", to_handle)

        # return to_handle, form
        

    def parse(self):

        while True:
            res = self.__pull()
            try:
                if res == "" or res == False or res in ENDING_PUNCT or (res[-1] in ENDING_PUNCT and re.findall("\w", res)):
                    break
            except IndexError:
                raise CHATValidationException(f"Lexer failed! Utterance ended without ending punct. Utterance: {self.raw}")

def lex(utterance):
    ut = UtteranceLexer(utterance)
    return ut.forms
