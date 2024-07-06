"""
verbforms.py
Fix Japanese verb forms.
"""

def verbform(upos, target, text):
    if "遣" in text and upos == "noun":
        return "verb", "遣る"
    if "死" in text:
        return "verb", "死ぬ"
    if "立" in text:
        return "verb", "立つ"
    if "引" in text:
        return "verb", "引く"
    if "出" in text:
        return "verb", "出す"
    if "引" in text:
        return "verb", "引く"
    if "飲" in text:
        return "verb", "飲む"
    if "呼" in text:
        return "verb", "呼ぶ"
    if "脱" in text:
        return "verb", "脱ぐ"
    if text == "な" and upos == "part":
        return "aux", "な"
    if text == "呼ん":
        return "verb", "呼ぶ"
    if text == "な" and upos == "aux":
        return "aux", "な"
    if text == "だり":
        return "aux", "たり"
    if text == "たり":
        return "aux", "たり"
    if text == "たら":
        return "sconj", "たら"
    if text == "たっ":
        return "sconj", "たって"
    # if text == "て" and upos == "sconj":
    #     return "aux", "て"
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
    # if upos == "verb" and "る" in target:
    #     return "verb", target.replace("る","").strip()

    return upos,target


