"""
verbforms.py
Fix Japanese verb forms.
"""

def verbform(upos, target, text):
    if text == "な" and upos == "part":
        return "aux", "うな"
    if text == "呼ん":
        return upos, "呼ん"
    if text == "たり":
        return "aux", "たり"
    if text == "たら":
        return "sconj", "たら"
    if text == "たっ":
        return "sconj", "たって"
    if text == "て" and upos == "sconj":
        return "aux", "て"
    if text == "なさい" and target == "為さる":
        return "aux", "為さい"
    if text == "な" and upos == "part":
        return "aux", "な"
    if text == "脱" and upos == "noun":
        return "verb", "脱"
    if text == "よう" and upos == "aux":
        return "aux", "よう"
    if text == "ろ" and upos == "aux" and target == "為る":
        return "aux", "ろ"
    if upos == "verb" and "る" in target:
        return "verb", target.replace("る","").strip()

    return upos,target


