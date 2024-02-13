# system utils
import glob, os, re
from itertools import groupby

# pathing tools
from pathlib import Path

# UD tools
import stanza

from stanza.utils.conll import CoNLL
from stanza import Document, DownloadMethod
from stanza.models.common.doc import Token
from stanza.pipeline.core import CONSTITUENCY
from stanza import DownloadMethod
from torch import heaviside

from stanza.pipeline.processor import ProcessorVariant, register_processor_variant

# the loading bar
from tqdm import tqdm

from bdb import BdbQuit

from nltk import word_tokenize
from collections import defaultdict

import warnings

from stanza.utils.conll import CoNLL

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob.glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, pathlib.Path(file_path).name)


from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.formats.chat.parser import chat_parse_utterance

from batchalign.utils.dp import *

import logging
L = logging.getLogger("batchalign")

import pycountry

# one liner to parse features
def parse_feats(word):
    try:
        return {i.split("=")[0]: i.split("=")[1] for i in word.feats.split("|")}
    except AttributeError:
        return {}
# one liner to join feature string
def stringify_feats(*feats):
    template= ("-"+"-".join(filter(lambda x: x!= "", feats))).strip()

    if template == "-": return ""
    else: return template.replace(",", "")

# the following is a list of feature-extracting handlers
# it is used to extract features from specific parts of
# speech.

def handler(word, lang=None):
    """The generic handler"""

    # if the lemma is ", return the word
    # not sure what errors are coming along?
    target = word.lemma

    if target == '"':
        target = word.text
    if not target:
        target = word.text

    # unknown flag
    unknown = False

    # if there is a 0 in front, the word is unkown
    # so we mark it as such
    if target[0] == '0':
        target = word.text[1:]
        unknown = True

    # if there is..... dear god, a sequence start <SOS>
    # token in the model output, return the text instead
    # of teh predicted lemma as something has gone very wrong
    if "<SOS>" in target:
        target = word.text

    target = target.replace("$", "")
    target = target.replace(".", "")

    # if we have a clitic that's broken off, we remove the extra dash
    if target != "" and target[0] == "-":
        target = target[1:]

    # if we have a dash marker in the end, we remove the extra dash
    if target != "" and target[-1] == "-":
        target = target[:-1]

    # replace double dashes
    target = target.replace("--", "-")
    target = target.replace("--", "-")
    target = target.replace("<unk>", "")
    target = target.replace("<SOS>", "")

    target = target.replace(',', '')
    target = target.replace('\'', '')
    target = target.replace('~', '')
    target = target.replace('/100', '')
    target = target.replace('/r', '')

    # remove attachments
    if "|" in target:
        target = target.split("|")[0].strip()

    # clean out alternate spellings
    target = target.replace("_", "")
    target = target.replace("+", "")

    # door zogen fix
    if target == "door zogen":
        target = word.text

    # fix dash
    target = target.replace("-", "–")

    return f"{'' if not unknown else '0'}{word.upos.lower()}|{target}"

# POS specific handler
def handler__PRON(word, lang=None):
    # get the features
    feats = parse_feats(word)
    person = str(feats.get("Person", 1))

    if person == "0":
        person = '4'

    case = feats.get("Case","")
    if lang == "fr":
        from batchalign.pipelines.morphosyntax.fr.case import case as caser
        case = caser(word.text)

    # parse
    return (handler(word, lang)+
            stringify_feats(feats.get("PronType", "Int"),
                            case.replace(",", ""),
                            feats.get("Number", "S")[:1]+person))

def handler__DET(word, lang=None):
    # get the features
    try:
        feats = parse_feats(word)
    except AttributeError:
        return handler(word)

    # get gender and numer
    gender_str = "-"+feats.get("Gender", "").replace(",", "")

    # clear defaults
    if gender_str == "-Com,Neut" or gender_str == "-Com" or gender_str=="-": gender_str=""

    # parse
    return (handler(word, lang)+gender_str+"-"+
            feats.get("Definite", "Def") + stringify_feats(feats.get("PronType", "")))

def handler__ADJ(word, lang=None):
    # get the features
    feats = parse_feats(word)
    # if there is a non-degree
    deg = feats.get("Degree", "Pos")
    case = feats.get("Case", "").replace(",", "")
    number = feats.get("Number", "S")[0]
    person = str(feats.get("Person", 1))
    if person == "0":
        person = '4'

    return handler(word, lang)+stringify_feats(deg, case, number[:1]+person)

def handler__NOUN(word, lang=None):
    # get the features
    feats = parse_feats(word)

    # get gender and numer
    gender_str = "-"+feats.get("Gender", "ComNeut").replace(",", "")
    number_str = "-"+feats.get("Number", "Sing")
    case  = feats.get("Case", "").replace(",", "")
    type  = feats.get("PronType", "")


    # clear defaults
    if gender_str == "-Com,Neut" or gender_str == "-Com" or gender_str == "-ComNeut": gender_str=""
    if number_str == "-Sing": number_str=""

    return handler(word, lang)+gender_str+number_str+stringify_feats(case, type)

def handler__PROPN(word, lang=None):
    # code as noun
    parsed = handler__NOUN(word)
    return parsed.replace("propn", "noun")

def handler__VERB(word, lang=None):
    # get the features
    feats = parse_feats(word)
    # seed flag
    flag = ""
    # append number and form if needed
    flag += "-"+feats.get("VerbForm", "Inf").replace(",", "")
    # append tense
    aspect = feats.get("Aspect", "")
    mood = feats.get("Mood", "")
    person = str(feats.get("Person", ""))

    if person == "0":
        person = '4'
    number = feats.get("Number", "Sing")

    tense = feats.get("Tense", "")
    polarity = feats.get("Polarity", "")
    polite = feats.get("Polite", "")
    return handler(word, lang)+flag+stringify_feats(aspect, mood,
                                              tense, polarity, polite,
                                              number[:1]+person)

def handler__actual_PUNCT(word, lang=None):
    # actual punctuation handler
    if word.lemma=="," or word.lemma=="$,":
        return "cm|cm"
    elif word.lemma in ['.', '!', '?']:
        return word.lemma
    elif word.text in '‡':
        return "end|end"
    elif word.text in '„':
        return "end|end"

def handler__PUNCT(word, lang=None):
    # no idea why SYM and PUNCT returns punctuation
    # or sometimes straight up words, but  so it goes
    # either punctuation or inflection words
    if word.lemma in ['.', '!', '?', ',', '$,']:
        return handler__actual_PUNCT(word, lang)
    elif word.text in ['„', '‡']:
        return handler__actual_PUNCT(word, lang)
    # otherwise, if its a word, return the word
    elif word.text == "da":
        return "noun|da"
    elif re.match(r"^['\w-]+$", word.text): # we match text here because .text is the ultumate content
                                        # instead of the lemma, which maybe entirely weird
        return f"x|{word.text}"

# Register handlers
HANDLERS = {
    "PRON": handler__PRON,
    "DET": handler__DET,
    "ADJ": handler__ADJ,
    "NOUN": handler__NOUN,
    "PROPN": handler__PROPN,
    "AUX": handler__VERB, # reuse aux handler for verb
    "VERB": handler__VERB,
    "PUNCT": handler__PUNCT,
    "SYM": handler__PUNCT # symbols are handled like punctuation
}

def handle(word, lang):
    if word.lemma in ['.', '!', '?', ',', '$,']:
        return handler__actual_PUNCT(word, lang)

    return HANDLERS.get(word.upos, handler)(word, lang)

# the follow
def parse_sentence(sentence, delimiter=".", special_forms=[], lang="$nospecial$"):
    """Parses Stanza sentence into %mor and %gra strings

    Arguments:
        sentence: the stanza sentence object
        [delimiter]: the default delimiter to use to end utterances
        [special_forms]: a list of special forms to replace back
        [lang]: language we are working with

    Returns:
        (str, str): (mor_string, gra_string)---strings matching
                    the output to be returned to %mor and %gra
        [delimiter]: how to end the utterance
    """

    # parse analysis results
    mor = []
    gra = []

    # root indx to point the ending delimiter to
    root = 0

    # counter for number of words skipped
    actual_indicies = []
    num_skipped = 0

    # generating temp "gra" data (array numerical, before shift
    # correction)
    gra_tmp = []

    # keep track of mwts
    mwts = []
    clitics = []
    # locations of elements with -ce, -être, -là
    # needs to be joined
    auxiliaries = []

    # TODO jank 2O(n) parse!
    # get mwts
    for indx, token in enumerate(sentence.tokens):
        if token.text[0]=="-":

            # we have to subtract 1 becasue $ goes to the
            # NEXT element
            auxiliaries.append(token.id[0]-1)

        if len(token.id) > 1:
            mwts.append(token.id)

        # if token.text.strip() == "l'":
            # clitics.append(token.id[0])
        if token.text.strip()[0] == "_":
            auxiliaries.append(token.id[0]-1)
            # if its a _l', we have one more thing to check
            if lang=="fr" and token.text.strip() == "_l'":
                auxiliaries.append(token.id[0])
        # elif lang=="fr" and (token.text.strip() == "c’" or token.text.strip() == "c'"):
            # auxiliaries.append(token.id[-1])
        elif token.text.strip()[0] == "~":
            auxiliaries.append(token.id[0]-1)
        elif lang=="it" and token.text.strip()[-3:] == "ll'":
            auxiliaries.append(token.id[-1])
        elif lang=="it" and token.text.strip() == "gliel'":
            auxiliaries.append(token.id[-1])
        elif lang=="it" and token.text.strip() == "d'":
            auxiliaries.append(token.id[-1])
        elif lang=="it" and (token.text.strip() == "c’" or token.text.strip() == "c'"):
            auxiliaries.append(token.id[-1])
        elif lang=="it" and token.text.strip() == "qual'":
            auxiliaries.append(token.id[-1])
        # elif lang=="fr" and token.text.strip() == "qu'":
            # auxiliaries.append(token.id[-1])
        elif lang=="fr" and token.text.strip() == "jusqu'":
            auxiliaries.append(token.id[-1])
        elif lang=="fr" and token.text.strip() == "puisqu'":
            auxiliaries.append(token.id[-1])
        elif lang=="fr" and token.text.strip() == "quelqu'":
            auxiliaries.append(token.id[-1])
        elif lang=="fr" and token.text.strip() == "aujourd":
            auxiliaries.append(token.id[0]+1)
        elif lang=="fr" and token.text.strip() == "aujourd'":
            auxiliaries.append(token.id[-1])
        # elif lang=="fr" and token.text.strip() == "aux":
            # auxiliaries.append(token.id[0])
        elif (lang=="fr" and token.text.strip() == "au" and
              type(token.id) == tuple and indx != 0
              and sentence.tokens[indx-1].text != "jusqu'"):
            auxiliaries.append(token.id[0])
        elif lang=="fr" and len(token.text.strip()) == 2 and token.text.strip()[-1] == "'":
            auxiliaries.append(token.id[-1])

    # because we pop from it
    special_forms = special_forms.copy()
    special_form_ids = []
    # get words
    for indx, word in enumerate(sentence.words):
        # append the appropriate mor line
        # by trying all handlers, and defaulting
        # to the default handler
        mor_word = handle(word, lang)
        # exception: if the word is 0, it is probably 0word
        # occationally Stanza screws up and makes forms like 0thing as 2 tokens:
        # 0 and thing
        if word.text.strip() == "0":
            mor.append("$ZERO$")
            num_skipped+=1 # mark skipped if skipped
            actual_indicies.append(root) # TODO janky but if anybody refers to a skipped
                                         # word they are root now.
        # normal parsing
        elif mor_word or word.text.strip() in ["xbxxx", '‡', '„']:
            if word.text.strip() == '‡':
                mor_word = "cm|begin"
            elif word.text.strip() == '„':
                mor_word = "cm|end"


            # specivl forms: recall the special form marker is xbxxx
            if "xbxxx" in word.text.strip():
                form = special_forms.pop(0)
                mor.append(f"x|{form.strip()}")
                special_form_ids.append(word.id)
            else:
                mor.append(mor_word)

            # +1 because we are 1-indexed
            # and .head is also 1-indexed already
            deprel = word.deprel.upper()
            deprel = deprel.replace(":", "-")
            gra_tmp.append(((indx+1)-num_skipped, word.head, deprel))
            actual_indicies.append((indx+1)-num_skipped) # so we can check later
            # if depedence relation is root, mark the current
            # ID as root
            if word.deprel.upper() == "ROOT":
                root = ((indx+1)-num_skipped)
        # some handlers may return None to skip the word
        else:
            mor.append(None)
            num_skipped+=1 # mark skipped if skipped
            actual_indicies.append(root) # TODO janky but if anybody refers to a skipped
                                         # word they are root now.

    # and now for each element, we shift and generate
    # recall that indicies are one indexed
    for i, elem in enumerate(gra_tmp):
        # if we are at a special form ID, append a special form mark instead
        if elem[0] in special_form_ids:
            elem = (elem[0], elem[1], "FLAT")
        # the third element is responsible for looking up the correctly
        # shifted index for the item in question
        gra.append(f"{elem[0]}|{actual_indicies[elem[1]-1]}|{elem[2]}")

    # append ending delimiter to GRA
    gra.append(f"{len(sentence.words)+1-num_skipped}|{root}|PUNCT")

    mor_clone = mor.copy()

    # we will join all the segments with ' in the end with
    # a dollar sign because those are considered
    # one word
    # recall again one indexing
    while len(clitics) > 0:
        clitic = clitics.pop()
        try:
            mor_clone[clitic-1] = mor_clone[clitic-1]+"$"+mor_clone[clitic]
        except IndexError:
            breakpoint()
        mor_clone[clitic] = None

    # connect auxiliaries with a "~"
    # recall 1 indexing
    for aux in auxiliaries:
        # if the previous one was joined,
        # we keep searching backwards

        orig_aux = aux
        while not mor_clone[aux-1]:
            aux -= 1

        if mor_clone[orig_aux]:
            mor_clone[aux-1] = mor_clone[aux-1]+"~"+mor_clone[orig_aux]
            mor_clone[orig_aux] = None

    while len(mwts) > 0:
        # handle MWTs
        # TODO assumption MWTs are continuous
        mwt = mwts.pop(0)
        mwt_start = mwt[0]
        mwt_end = mwt[-1]

        # why the copious -1s? One indexing

        # combine results
        mwt_str = "~".join([i for i in mor_clone[mwt_start-1:mwt_end] if i])

        # delete old
        for j in range(mwt_start, mwt_end+1):
            mor_clone[j-1] = None

        # replace in new dict
        mor_clone[mwt_start-1] = mwt_str

    mor_str = (" ".join(filter(lambda x:x, mor_clone))).strip().replace(",", "")
    gra_str = (" ".join(gra)).strip()

    # handle special zeros, see $ZERO$ above
    mor_str = mor_str.replace("$ZERO$ ","0")

    # add the endning delimiter
    if len(mor_str) != 1: # if we actually have content (not just . or ?)
                          # add a deliminator
        mor_str = mor_str + " " + delimiter


    mor_str = mor_str.replace("<UNK>", "")
    gra_str = gra_str.replace("<UNK>", "")

    # empty utterances fix
    if mor_str.strip() in ["+//.", "+//?", "+//!"]:
        mor_str=None

    if mor_str == None:
        mor_str = ""
    if gra_str == None:
        gra_str = ""

    if mor_str.strip() == "" or gra_str.strip() == "" or mor_str.strip()==".":
        mor_str = ""
        gra_str = ""

    return (mor_str, gra_str)

def clean_sentence(sent):
    """clean a sentence

    Arguments:
        sent (string):
    """

    remove = ["+,", "++", "+\""]

    sent = sent

    # remove each element
    for i in remove:
        sent = sent.replace(i, "")

    return sent

def matches(i, word):
    return (type(i) == tuple and i[0] == word) or (i == word)

def matches_in(i, fragment):
    return (type(i) == tuple and fragment in i[0]) or (fragment in i)

def front_matches(i, word):
    return (type(i) == tuple and i[:len(word)] == word) or (i[:len(word)] == word)

def conform(i):
    return i[0] if type(i) == tuple else i

def tokenizer_processor(tokenized, lang, sent):
    # split tokenized. in case stuff got combined
    tokenized = [j for i in tokenized for j in conform(i).split(" ")]
    # tabulate results
    res = []
    # align the input sentence and the tokenization results
    payloads = []
    split_passage = sent.split(" ")
    # create alignment backplates, where the split_passage
    # is reference and the tokenized is ptarget
    targets = []
    refs = []
    for indx, i in enumerate(tokenized):
        for char in conform(i):
            if char.strip() != "":
                targets.append(PayloadTarget(char.strip(), indx))
    for indx, i in enumerate(split_passage):
        for char in i:
            if char.strip() != "":
                refs.append(ReferenceTarget(char.strip(), indx))

    # create groups such that if multiple of the tokenized result
    # belongs to the same group (i.e. orthographically
    # the same unit), we combine it into one
    groups = []
    alignment = align(targets, refs, tqdm=False)
    if any([isinstance(i, Extra) for i in alignment]):
        breakpoint()
    alignments = groupby(alignment, lambda x:x.reference_payload)
    for key, grp in alignments:
        group = []
        for elem in grp:
            group.append(elem.payload)
        groups.append(list(sorted(set(group))))

    # create new tokenizations, marking MWTs
    seen = []
    new_toks = []
    for i in groups:
        i = list(filter(lambda x:x not in seen, i))
        if len(i) == 1:
            new_toks.append(tokenized[i[0]])
        elif len(i) == 0:
            continue
        else:
            # we combine all the tokens and mark as MWT
            new_toks.append(("".join([conform(tokenized[j]) for j in i]), False))
        seen += i

    tokenized = new_toks

    indx = 0
    while indx < len(tokenized):
        i = tokenized[indx]
        # italian taggs l' as MWT, we patch that
        if ("it" in lang) and type(i) == tuple and i[0]=="l'" and i[1] == True:
            res.append("l'")
        # italian breaks up lei into le, i, we patch that
        elif ("it" in lang) and matches(i, "i") and len(res) != 0 and matches(res[-1], "le"):
            res.pop(-1)
            res.append("lei")
        elif ("pt" in lang) and matches(i, "d'água"):
            res.append(("d'água", True))
        elif ("fr" in lang) and matches(i, "aujourd'hui"):
            res.append("aujourd'hui")
        elif ("fr" in lang) and matches(i, "aujourd'"):
            res.append("aujourd'hui")
            indx += 1
        elif ("fr" in lang) and matches(i, "au"):
            res.append((conform(i), True))
        # we need to triple clitics...
        # "pour essayer d'l'attraper"
        elif ("fr" in lang) and re.match(r"(\w')+\w+", conform(i)):
            parts = conform(i).split("'")
            with_clitic, without = parts[:-1], parts[-1]
            for elem in with_clitic:
                res.append((f'{elem}\'', False))
            res.append((f'{without}', False))
        elif ("fr" in lang) and conform(i).split("'")[0] in ["jusqu", "puisqu", "quelqu", "aujourd"]:
            before,after = conform(i).split("'")
            res.append((f'{before}\'', False))
            res.append((after, False))
        elif ("en" in lang) and matches_in(i, "'"):
            res.append((conform(i), True))
        elif ("nl" in lang) and conform(i).endswith("'s"):
            res.append((conform(i), False))
        else:
            res.append(i)
        indx += 1

    return res

######
def morphoanalyze(doc: Document, retokenize:bool, status_hook:callable = None):
    L.debug("Starting Stanza...")
    inputs = []

    lang = []
    for i in doc.langs:
        try:
            lang.append(pycountry.languages.get(alpha_3=i).alpha_2)
        except:
            # some languages don't have alpha 2
            pass


# pycountry.languages.get(alpha_3=i).alpha_2 for i in lang

    config = {"processors": {"tokenize": "default",
                             "pos": "default",
                             # "mwt": "gum" if ("en" in lang) else "default",
                             "lemma": "default",
                             "depparse": "default"},
              "tokenize_no_ssplit": True}

    if not retokenize:
        config["tokenize_postprocessor"] = lambda x:[tokenizer_processor(i, lang, inputs[-1]) for i in x]

    if "zh" in lang:
        lang.pop(lang.index("zh"))
        lang.append("zh-hans")

    elif not any([i in ["hr", "zh", "zh-hans", "ja", "ko",
                        "sl", "sr", "bg", "ru", "et", "hu",
                        "eu", "el", "he", "af"] for i in lang]):
        if "en" in lang:
            config["processors"]["mwt"] = "gum"
        else:
            config["processors"]["mwt"] = "default"

    configs = {}
    for l in lang:
        configs[l] = config.copy()

    nlp = stanza.MultilingualPipeline(
        lang_configs = configs,
        lang_id_config = {"langid_lang_subset": lang},
        download_method=DownloadMethod.REUSE_RESOURCES
    )

    for indx, i in enumerate(doc.content):
        L.info(f"Stanza processing utterance {indx+1}/{len(doc.content)}")
        if not isinstance(i, Utterance):
            continue

        # generate simplified version of the line
        line = str(i)

        # if we have nothing except a punctuuation, we give up
        if line.strip() in ENDING_PUNCT:
            continue

        # every legal utterance will have an ending delimiter
        # so we split it out
        try:
            ending = i.strip(join_with_spaces=True).split(" ")[-1]
        except AttributeError:
            breakpoint()

        if re.findall("\w", ending):
            ending = "."
            line_cut = i.strip(join_with_spaces=True)
        else:
            line_cut = i.strip(join_with_spaces=True)[:-len(ending)].strip()
            # ending = ending.replace("+//", "")

        # if we don't have anything in line cut, just take the original
        # this is compensating for things that are missing ending decimeters
        if line_cut == '':
            line_cut = i.strip(join_with_spaces=True)
            ending = '.'

        # clean the sentence
        line_cut = clean_sentence(line_cut)

        # if at this point we still have nothing, just
        # assume its an end punctuation (i.e. the whole
        # utterance was probably just ++ or something
        # that clean_sentence cut out

        if line_cut == "":
            line_cut = ending

        # line_cut = line_cut.replace("_", "-")
        line_cut = line_cut.replace("+<", "")
        line_cut = line_cut.replace("+/", "")
        line_cut = line_cut.replace("(", "")
        line_cut = line_cut.replace(")", "")
        line_cut = line_cut.replace("+^", "")
        line_cut = line_cut.replace("+//", "")
        line_cut = line_cut.replace("+...", "")
        line_cut = line_cut.replace("_", "")

        # xbxxx is a sepecial xxx-class token to mark
        # special form markers, used for processing later
        # down the line
        special_forms = re.findall(r"\w+@[\w\:]+", line_cut)
        special_forms_cleaned = []
        for form in special_forms:
            line_cut = line_cut.replace(form, "xbxxx")
            special_forms_cleaned.append(re.sub(r"@[\w\:]+", "", form).strip())

        # if line cut is still nothing, we get very angry
        if line_cut == "":
            line_cut = '.'

        # Norwegian apostrophe fix
        if line_cut[-1] == "'":
            line_cut = line_cut[:-1]

        line_cut = line_cut.replace(",", " ,")
        line_cut = line_cut.replace("+ ,", "+,")
        line_cut = line_cut.replace("  ", " ")
        # line_cut = line_cut.replace("c'est", "c' est")


        # try:
        inputs.append(line_cut)

        try:
            sents = nlp(line_cut.strip()).sentences

            if len(sents) == 0:
                continue

            if status_hook:
                status_hook(indx+1, len(doc.content))


            # parse the stanza output
            mor, gra = parse_sentence(sents[0], ending, special_forms_cleaned, lang[0])
            # breakpoint()

            if mor.strip() == "" or mor.strip() in ENDING_PUNCT:
                L.debug(f"Encountered an utterance that's likely devoid of morphological information; skipping... utterance='{doc.content[indx]}'")
                continue

            if retokenize:
                # rewrite the sentence with our desired tokenizations
                ut, end = chat_parse_utterance(" ".join([i.text for i in sents[0].words])+" "+ending,
                                          mor, gra,
                                          None, None)
                doc.content[indx] = Utterance(content=ut,
                                              tier=doc.content[indx].tier,
                                              time=doc.content[indx].time,
                                              custom_dependencies=doc.content[indx].custom_dependencies)
                                          
            else:
                # insert morphology into the parsed forms
                forms, _ = chat_parse_utterance(line, mor, gra, None, None)

                if len(doc.content[indx].content) != len(forms):
                    warnings.warn(f"Generated UD output has length mismatch with the main tier! line='{line}'. Skipping...")
                    continue

                # stitch the morphology back
                for content, form in zip(doc.content[indx].content, forms):
                    content.morphology = form.morphology
                    content.dependency = form.dependency

        except Exception as e:
            warnings.warn(f"Utterance failed parsing, skipping ud tagging... line='{line}', error='{e}'.\n")

    L.debug("Stanza done.")
    return doc

class StanzaEngine(BatchalignEngine):
    tasks = [ Task.MORPHOSYNTAX ]
    status_hook = None

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def process(self, doc, **kwargs):
        return morphoanalyze(doc, retokenize=kwargs.get("retokenize", False), status_hook=self.status_hook)
