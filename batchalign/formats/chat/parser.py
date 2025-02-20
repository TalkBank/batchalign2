from batchalign.document import *
from batchalign.utils import *
from batchalign.errors import *
from batchalign.constants import *

from batchalign.formats.base import BaseFormat
from batchalign.formats.chat.utils import *
from batchalign.formats.chat.lexer import lex, TokenType

import re

def chat_parse_utterance(text, mor, gra, wor, additional):
    """Encode a CHAT utterance into a Batchalign utterance.

    Parameters
    ----------
    text : str
        String text of the main tier.
    mor : Optional[str]
        mor line, or None.
    gra : Optional[str]
        gra line, or None.
    wor : Optional[str]
        wor line, or None.
    additional : List[CustomLine]
        Any custom lines.

    Returns
    -------
    Utterance, str
        Parsed utterance and the utterance delimiter.

    Raises
    ------
    CHATValidationException
        If the utterance didn't parse corrected, we raise an
        exception describing the issue.
    """

    # scan the timing
    # lex the utterance
    to_lex = re.compile(r"\x15\d+_\d+\x15").sub("", text).strip()

    # if the first form has a < in it and has no words,
    # its probably a beginning delimiter which we do not lex
    if (len(to_lex) > 0 and
        ("<" in to_lex.split(" ")[0]  or "+" in to_lex.split(" ")[0] )
        and not re.findall(r"\w", to_lex.split(" ")[0])):
        beg = to_lex.split(" ")[0]
        to_lex = to_lex.replace(beg, "", 1)

    # fix all spacing issues
    to_lex = to_lex.replace("< ", "<")
    to_lex = to_lex.replace(" ]", "]")
    to_lex = to_lex.replace("[ ", "[")
    to_lex = to_lex.replace(" >", ">")
    to_lex = to_lex.replace("  ", " ") # we do this twice
    to_lex = to_lex.replace("  ", " ")


    # get rid of CA delimiters
    to_lex = re.compile("⌊[&-]=[ A-Za-zÀ-ÖØ-öø-ÿ'-]+⌋").sub("", to_lex).strip()

    # fix commas for people that don't annotate commas with a space
    to_lex = to_lex.replace(",", " ,")

    to_lex = re.sub(r"\([\d.:]+\)(?!$)", "", to_lex)
    to_lex = re.sub(r"↫.*?↫", "", to_lex)

    to_lex = re.sub(r"\(.\)$", r"$END_SPC$", to_lex)

    # if there is a punct, move it
    for end in sorted(ENDING_PUNCT, key=len, reverse=True):
        if end in to_lex:
            to_lex = to_lex.replace(end, f" {end}")
            break

    to_lex = to_lex.replace("  ", " ")

    tokens = lex(to_lex)
    if tokens[-1][0] == "END_SPC":
        tokens = tokens[:-1] + [("(.)", TokenType.PUNCT)]

    # correct 0 forms
    res = []
    for i,j in tokens:
        if i.strip() != "" and i[0] == '0':
            res.append((i[1:], TokenType.CORRECTION))
        else:
            res.append((i,j))

    tokens = res

    # seperate out main words by whether it should have phonation/morphology and add ending punct
    words = list(enumerate(tokens))
    lexed_words = [tok for tok in words if tok[1][1] in [TokenType.REGULAR,
                                                            TokenType.PUNCT]]
    phonated_words = [tok for tok in words if tok[1][1] in [TokenType.REGULAR,
                                                            TokenType.RETRACE,
                                                            TokenType.PUNCT,
                                                            TokenType.FP]]
    # create base forms
    parsed_forms = {indx:Form(text=text, type=type) for indx, (text, type) in phonated_words}

    # stamp an eding punct if its not there
    delim = "."
    if words[-1][1][0] in ENDING_PUNCT:
        if words[-1][1][1] == TokenType.REGULAR:
            lexed_words.append(words[-1])
        parsed_forms[words[-1][0]] = Form(text=words[-1][1][0], type=TokenType.PUNCT)
        delim = words[-1][1][0]

    # parse mor gra wor lines
    if mor == None:
        mor = [None for i in range(len(lexed_words))]
    else:
        mor = mor.split(" ")
    if gra != None:
        gra = gra.split(" ")
    if wor == None:
        wor = [None for i in range(len(phonated_words))]
    else:
        words = re.findall(rf"[^{''.join([i for i in ENDING_PUNCT if len(i) == 1])} ]+ ?(\x15\d+_\d+\x15)?", wor)
        wor = []
        for i in words:
            if i.strip() == "":
                wor.append(None)
                continue
            x, y = re.findall(r"\d+", i)
            wor.append([int(x),int(y)])

    # check lengths
    if len(lexed_words) != len(mor):
        raise CHATValidationException(f"Lengths of main and mor tiers are unaligned: lens main (filtered for morphology)={len(lexed_words)} mor={len(mor)}; line: '{text}'")
    if (len(phonated_words) > 0 and
        phonated_words[-1][1][1] == TokenType.PUNCT and # because we don't track last ending PUNCT
        (len(phonated_words)-1 != len(wor))) and (len(phonated_words) != len(wor)):
        raise CHATValidationException(f"Lengths of main and wor tiers are unaligned: lens main (filtered for phonation)={len(phonated_words)} wor={len(wor)}; line: '{text}'")

    # insert morphology into the parsed forms
    for (indx, _), m in zip(lexed_words, mor):
        if m:
            mor = chat_parse_mor(m)
            parsed_forms[indx].morphology = mor

            # grab the right lines in dependency
            if gra != None:
                deps = []
                for _ in range(len(mor)):
                    try:
                        deps.append(gra.pop(0))
                    except IndexError:
                        raise CHATValidationException(f"Lengths of mor and gra tiers are misaligned --- gra line too short on line: '{text}'")
                if all(i != None for i in deps):
                    deps = [chat_parse_gra(i) for i in deps]
                    parsed_forms[indx].dependency = deps

    if all([i != None for i in mor]) and gra != None and len(gra) != 0:
        raise CHATValidationException(f"Lengths of mor and gra tiers are misaligned --- gra line too long on line: '{text}'")

    # insert phonation into the parsed forms
    for (indx, _), w in zip(phonated_words, wor):
        if w:
            parsed_forms[indx].time = tuple(w)

    # return final parsed forms
    forms = [i[1] for i in sorted(parsed_forms.items(), key=lambda x:x[0])]

    return forms, delim

def chat_parse_doc(lines, special_mor=False):
    """Encode a CHAT files' lines into a Batchalign Document.

    Parameters
    ----------
    lines : List[str]
        The contents of the CHAT file
    special_mor : False
        Use umor/ugra tiers 

    Returns
    -------
    Document
        Parsed Batchalign document.

    Raises
    ------
    CHATValidationException
        If the utterance didn't parse correctly, we raise an
        exception describing the issue.
    """

    pid = None

    raw = lines
    # pop off anything that's before @Begin
    while raw[0].strip() != "@Begin" and raw[0].strip() != "\ufeff@Begin" :
        ut = raw.pop(0)
        if "@PID" in ut:
            try:
                head, end = ut.split("\t")
            except ValueError:
                raise CHATValidationException(f"Encountered unexpected PID line: {ut}")
            pid = end.strip()
    raw.pop(0)

    results = {
        "content": [], 
        "langs": [], 
        "media": None,
        "pid": pid
    }

    tiers = {}

    use_special_mor = False

    # read data
    while raw[0].strip() != "@End":
        try:
            line = raw.pop(0)

            # we throw away participants because there are duplicate
            # info of the same thing in @ID
            if "@Participants" in line or "@Options" in line:
                continue
            # we split because there are multiple languages possible 
            elif "@Languages" in line.strip():
                results["langs"] = [i.strip() for i in line.strip("@Languages:").strip().replace(" ", ",").strip().split(",") if i.strip() != ""]
                if len(results["langs"]) > 0 and results["langs"][0] == "eng" and special_mor:
                    use_special_mor = True
            # parse participants; the number of | delinates the metedata field
            elif "@ID" in line.strip():
                participant = line.strip("@ID:").strip().split("|")

                tier = Tier(lang=participant[0], corpus=participant[1], 
                            id=participant[2], name=participant[7],
                            birthday=participant[3], additional=[participant[i]
                                                                 for i in [4,5,6,8,9]])
                tiers[participant[2]] = tier
            # parse media type
            elif "@Media" in line.strip():
                type = MediaType.UNLINKED_AUDIO

                if "unlinked" in line and "audio" in line:
                    type = MediaType.UNLINKED_AUDIO
                elif "unlinked" in line and "video" in line:
                    type = MediaType.UNLINKED_VIDEO
                elif "missing" in line and "audio" in line:
                    type = MediaType.MISSING_AUDIO
                elif "missing" in line and "video" in line:
                    type = MediaType.MISSING_VIDEO
                elif "audio" in line:
                    type = MediaType.AUDIO
                elif "video" in line:
                    type = MediaType.VIDEO

                media = line.strip("@Media:").split(",")
                results["media"] = Media(type=type, name=media[0].strip(),
                                        url=None)
            # depenent tiers with @ are counted as "other" and are inserted as-is
            elif line.strip()[0] == "@":
                try:
                    beg,end = line.strip()[1:].split(":\t")
                except ValueError:
                    # we only have one
                    beg = line.strip()[1:].strip()
                    end = None
                line = CustomLine(id=beg.strip(),
                                    type=CustomLineType.INDEPENDENT,
                                    content=end.strip() if end != None else None)
                results["content"].append(line)
            # we now parse main tiers
            elif line.strip()[0] == "*":
                utterance = {}
                tier,text = line.strip()[1:].split(":\t")
                utterance["text"] = text

                # parse mor and gra lines, and append all other tiers
                mor = None
                gra = None
                wor = None
                translation = None
                additional = []

                while raw[0][0] == "%":
                    line = raw.pop(0)
                    beg,line = line.strip()[1:].split(":\t")
                    if beg.strip() == f"{'u' if use_special_mor else ''}mor":
                        mor = line
                    elif beg.strip() == f"{'u' if use_special_mor else ''}gra":
                        gra = line
                    elif beg.strip() == "wor" or beg.strip() == "xwor":
                        wor = line
                    elif beg.strip() == "xtra":
                        translation = line
                    else:
                        additional.append(CustomLine(id=beg.strip(),
                                                        type=CustomLineType.DEPENDENT,
                                                        content=line.strip()))

                # parse the actual utterance
                parsed, delim = chat_parse_utterance(text, mor, gra, wor, additional)

                # get the timing of the utterance
                try:
                    t = tiers[tier]
                except KeyError:
                    raise CHATValidationException(f"Encountered undeclared tier: tier='{tier}', line='{text}'")
                ut = Utterance.model_validate({
                    "tier": t,
                    "content": parsed,
                    "text": text,
                    "delim": delim,
                    "custom_dependencies": additional,
                    "translation": translation
                })

                timing = re.findall(rf"\x15(\d+)_(\d+)\x15", text)
                if len(timing) != 0:
                    x,y = timing[0]
                    ut.time = (int(x), int(y))
                results["content"].append(ut)

            # throw error for everything else
            else:
                raise CHATValidationException(f"Unknown line in input CHAT: '{line}'")
        except Exception as e:
            if isinstance(e, CHATValidationException):
                raise e
            if len(raw) > 0:
                raise ValueError(f"Unexpected Batchalign error when parsing CHAT: line='{raw[0].strip()}', error='{str(e)}'")
            else:
                raise ValueError(f"Unexpected Batchalign error when parsing CHAT: file is exausted, error='{str(e)}'")

    doc = Document.model_validate(results)
    return doc

