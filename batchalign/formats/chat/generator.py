from batchalign.document import *
from batchalign.constants import *
import numbers

import warnings

# c = CHATFile("./extern/test.cha")
# document = c.doc

# document[0].model_dump()
# document[3].text = None
# document[3].model_dump()

def generate_chat_utterance(utterance: Utterance, special_mor=False, write_wor=True):
    """Converts at Utterance to a CHAT string.

    Parameters
    ----------
    utterance : Utterance
        The utterance to be written to string.
    special_mor : False
        Use umor/ugra tiers

    Returns
    -------
    str
        The generated string.
    """
    
    main_line = str(utterance)
    tier = utterance.tier

    mors = []
    gras = []
    has_wor = False
    wor_elems = []
    has_coref = False
    coref_elems = []

    for i in utterance.content:
        mors.append(i.morphology)
        gras.append(i.dependency)
        if i.time:
            has_wor = True
            wor_elems.append(re.sub(r"@(\w)(\w\w\w)", r"@\1:\2", f"{i.text} \x15{str(i.time[0])}_{str(i.time[1])}\x15"))
        else:
            wor_elems.append(re.sub(r"@(\w)(\w\w\w)", r"@\1:\2", i.text))

        if i.coreference:
            has_coref = True
            coref_str_form = []
            for j in i.coreference:
                coref_str = ""
                if j.start:
                    coref_str += "("
                coref_str += str(j.chain)
                if j.end:
                    coref_str += ")"
                coref_str_form.append(coref_str)
            coref_elems.append(" ".join(coref_str_form))
        else:
            coref_elems.append("-")

        if bool(mors[-1]) != bool(gras[-1]):
            warnings.warn(f"Batchalign has detected a mismatch between lengths of mor and gra tiers for utterance; output will not pass CHATTER; line='{main_line}'")


    # assemble final output
    result = [f"*{tier.id}:\t"+main_line]

    #### MOR LINE GENERATION ####
    # we need to first join every MWT with ~
    mor_elems = []
    for mor in mors:
        if mor != None:
            # I'm sorry. This is just mor line generation; there is actually not that much complexity here
            mor_elems.append("~".join(f"{m.pos}|{m.lemma}{'-' if any([m.feats.startswith(i) for i in UD__GENDERS]) else ('-' if m.feats else '')}{m.feats}"
                                    for m in mor))
    if len(mor_elems) > 0:
        # if the end is punct, drop the tag
        if mor_elems[-1].startswith("PUNCT"):
            mor_elems[-1] = mor_elems[-1].split("|")[1]
        result.append(f"%{'u' if special_mor else ''}mor:\t"+" ".join(mor_elems))

    #### GRA LINE GENERATION ####
    # gra list is not different for MWT tokens so we flatten it
    gras = [i for j in gras if j for i in j]
    # assemble gra line
    gra_line = None
    if len(gras) > 0:
        result.append(f"%{'u' if special_mor else ''}gra:\t"+" ".join([f"{i.id}|{i.dep_id}|{i.dep_type}" for i in gras]))

    #### WOR LINE GENERATION ####
    if has_wor and write_wor:
        result.append("%wor:\t"+" ".join(wor_elems))
    if has_coref:
        result.append("%coref:\t"+(", ".join(coref_elems)))
    if utterance.translation != None:
        result.append("%xtra:\t"+utterance.translation)


    #### EXTRA LINE GENERATION ####
    for special in utterance.custom_dependencies:
        if special.content:
            result.append(f"%{special.id}:\t"+special.content)

    return "\n".join(result)

def check_utterances_ordered(doc):
    """check if the utterances are ordered such that one is aligned after another

    Parameters
    ----------
        doc : Document
                The CHAT document to check.

    Returns
    -------
        bool
                Whether the utterances timings are ordered or not.
    """

    n = -1
    for i in doc.content:
        if isinstance(i, Utterance) and i.alignment:
            (start, end) = i.alignment
            if isinstance(start, numbers.Number) != None and isinstance(end, numbers.Number) != None:
                if end < start:
                    return False
                if start < n:
                    return False
                n = end
    return True

def generate_chat_preamble(doc, birthdays=[]):
    """Generate header for a Batchalign document.

    Parameters
    ----------
    doc : Document
        The document to generate a CHAT header.
    birthdays : List[CustomLine]
        A list of custom lines for which the id mentions "Birthday"
        (It's apparently a CHAT requirement to put them right after @ID)

    Returns
    -------
    str
        The generated CHAT preamble.
    """
    
    header = []
    header.append("@Languages:\t"+", ".join(doc.langs))
    header.append("@Participants:\t"+", ".join([f"{i.id} {i.name}" for i in doc.tiers]))
    if not check_utterances_ordered(doc):
        header.append("@Options:\tbullets")
    header.append("\n".join([f"@ID:\t{i.lang}|{i.corpus}|{i.id}|{i.birthday}|{i.additional[0]}|{i.additional[1]}|{i.additional[2]}|{i.name}|{i.additional[3]}|{i.additional[4]}|" for i in doc.tiers]))
    for i in birthdays:
        header.append(f"@{i.id}:\t{i.content}")
    if doc.media:
        header.append(f"@Media:\t{doc.media.name}, {doc.media.type.value}")

    return "\n".join(header)

