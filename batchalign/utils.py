import re
import nltk
from nltk import word_tokenize as WT
from nltk import sent_tokenize as ST
from nltk.tokenize.treebank import TreebankWordDetokenizer

def word_tokenize(str):
    """Tokenize a string by word

    Parameters
    ----------
    str : str
        input string.

    Returns
    -------
    List[str]
        Word tokens.
    """
    
    try:
        return WT(str)
    except LookupError:
        nltk.download("punkt")
        return WT(str)

def sent_tokenize(str):
    """Tokenize a string by sentence

    Parameters
    ----------
    str : str
        input string.

    Returns
    -------
    List[str]
        Sentence tokens.
    """
 
    try:
        return ST(str)
    except LookupError:
        nltk.download("punkt")
        return ST(str)

def detokenize(tokens):
    """Merge tokenized words.

    Parameters
    ----------
    tokens : List[str]
        input tokens.

    Returns
    -------
    str
        Result strings.
    """
 
    try:
        return TreebankWordDetokenizer().detokenize(tokens)
    except LookupError:
        nltk.download("punkt")
        return TreebankWordDetokenizer().detokenize(tokens)

def mor_clean(content):
    """Cleans input tier like mor would.

    Parameters
    ----------
    content : str
        The string for which the input should be cleaned.

    Returns
    -------
    str
        The cleaned input, like mor would.
    """
    

    pass

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
    cleaned_word = cleaned_word.replace("〕","").replace("//","")
    cleaned_word = re.sub(r"@.", '', cleaned_word)
    cleaned_word = re.sub(r"&.", '', cleaned_word)

    return cleaned_word



