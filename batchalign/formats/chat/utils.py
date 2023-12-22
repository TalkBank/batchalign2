import re 
from ...document import *
from ...constants import *
from ...errors import *

def chat_parse_gra(mor_str):
    """Parses a gra string into id, dest, type

    Parameters
    ----------
    gra_str : str
        The raw mor string.

    Returns
    -------
    List[Dependency]
        Parsed dependenices.
    """

    src, dest, type = mor_str.split("|")
    return Dependency.model_validate({
        "id": src,
        "dep_id": dest,
        "dep_type": type,
    })



def chat_parse_mor(mor_str):
    """Parses a mor string into pos, lemmas, feats.

    Parameters
    ----------
    mor_str : str
        The raw mor string.

    Returns
    -------
    List[Morphology]
        Parsed morphologies.
    """

    if mor_str in ENDING_PUNCT:
        return [Morphology(lemma=mor_str, pos="PUNCT", feats="")]

    try:
        mors = [i.split("|") for i in re.split("[~$]", mor_str)]
        feats = [re.split("[-&]", i[1]) for i in mors]
        lemmas, feats = zip(*[(i[0], "&".join(i[1:])) for i in feats])
        pos = [i[0] for i in mors]
    except:
        raise CHATValidationException(f"mor parser recieved invalid mor string: '{mor_str}'")

    mors = [Morphology.model_validate({
        "lemma": l,
        "pos": p,
        "feats": f,
    }) for p,l,f in zip(pos, lemmas, feats)]

    return mors
    
def annotation_clean(content):
    """Clean anotation marks from string.

    Parameters
    ----------
    content : str
        The string from which annotation marks should be cleaned.

    Returns
    -------
    str
        The resulting string without annotation marks.
    """
 
    word = content
    cleaned_word = word.replace("[/]","") # because of weird spacing inclusions
    cleaned_word = re.sub(r"\x15\d+_\d+\x15", '', cleaned_word)
    cleaned_word = re.sub(r"&~\w+", '', cleaned_word)
    cleaned_word = cleaned_word.replace("(","").replace(")","")
    cleaned_word = cleaned_word.replace("[","").replace("]","")
    cleaned_word = cleaned_word.replace("<","").replace(">","")
    cleaned_word = cleaned_word.replace("“","").replace("”","")
    cleaned_word = cleaned_word.replace(",","").replace("!","")
    cleaned_word = cleaned_word.replace("?","").replace(".","")
    cleaned_word = cleaned_word.replace("&=","").replace("&-","")
    cleaned_word = cleaned_word.replace("+","").replace("&","")
    cleaned_word = cleaned_word.replace(":","").replace("^","")
    cleaned_word = cleaned_word.replace("$","").replace("\"","")
    cleaned_word = cleaned_word.replace("&*","").replace("∬","")
    cleaned_word = cleaned_word.replace("-","").replace("≠","")
    cleaned_word = cleaned_word.replace(":","").replace("↑","")
    cleaned_word = cleaned_word.replace("↓","").replace("↑","")
    cleaned_word = cleaned_word.replace("⇗","").replace("↗","")
    cleaned_word = cleaned_word.replace("→","").replace("↘","")
    cleaned_word = cleaned_word.replace("⇘","").replace("∞","")
    cleaned_word = cleaned_word.replace("≋","").replace("≡","")
    cleaned_word = cleaned_word.replace("∙","").replace("⌈","")
    cleaned_word = cleaned_word.replace("⌉","").replace("⌊","")
    cleaned_word = cleaned_word.replace("⌋","").replace("∆","")
    cleaned_word = cleaned_word.replace("∇","").replace("*","")
    cleaned_word = cleaned_word.replace("??","").replace("°","")
    cleaned_word = cleaned_word.replace("◉","").replace("▁","")
    cleaned_word = cleaned_word.replace("▔","").replace("☺","")
    cleaned_word = cleaned_word.replace("♋","").replace("Ϋ","")
    cleaned_word = cleaned_word.replace("∲","").replace("§","")
    cleaned_word = cleaned_word.replace("∾","").replace("↻","")
    cleaned_word = cleaned_word.replace("Ἡ","").replace("„","")
    cleaned_word = cleaned_word.replace("‡","").replace("ạ","")
    cleaned_word = cleaned_word.replace("ʰ","").replace("ā","")
    cleaned_word = cleaned_word.replace("ʔ","").replace("ʕ","")
    cleaned_word = cleaned_word.replace("š","").replace("ˈ","")
    cleaned_word = cleaned_word.replace("ˌ","").replace("‹","")
    cleaned_word = cleaned_word.replace("›","").replace("〔","")
    cleaned_word = cleaned_word.replace("〕","").replace("//","").replace(";","")
    cleaned_word = re.sub(r"@.", '', cleaned_word)
    cleaned_word = re.sub(r"&.", '', cleaned_word)

    return cleaned_word



