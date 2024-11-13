"""
verbforms.py
Fix Japanese verb forms.
"""

def verbform(upos, target, text):
    if "ちゃ" in text:
        return "sconj", "ば"
    if "なきゃ" in text:
        return "sconj", "なきゃ"
    if "じゃ" in text:
        return "sconj", "ちゃ"
    if "れる" in text:
        return "aux", "られる"
    if "じゃう" in text:
        return "aux", "ちゃう"
    if "よう" in text:
        return "aux", "おう"
    if "だら" in text:
        return "aux", "たら"
    if "だ" in target:
        return "aux", "た"
    if "為る" in target and 'さ' == text:
        return "part", "為る"
    if "無い" in target:
        return "aux", "ない"
    if "せる" in target:
        return "aux", "させる"
    if "撮る" in text:
        return "verb", "撮る"
    if "貼る" in text:
        return "verb", "貼る"
    if "混ぜ" in text:
        return "verb", "混ぜる"
    if "釣る" in text:
        return "verb", "釣る"
    if "速い" in text and upos == "adj":
        return "adj", "速い"
    if "治ま" in text:
        return "verb", "治まる"
    if "刺す" in text:
        return "verb", "刺す"
    if "降り" in text:
        return "verb", "降りる"
    if "降" in text:
        return "verb", "降る"
    if "載せ" in text:
        return "verb", "載せる"
    if "帰" in text:
        return "verb", "帰る"
    if "はい" in text:
        return "intj", "はい"
    if "うん" in text:
        return "intj", "うん"
    if "おっ" in text:
        return "intj", "おっ"
    if "ほら" in text:
        return "intj", "ほら"
    if "ヤッホー" in text:
        return "intj", "ヤッホー"
    if "ただいま" in text:
        return "intj", "ただいま"
    if "あたし" in text:
        return "pron", "あたし"
    if "舐め" in text:
        return "verb", "舐める"
    if "バツ" in text:
        return "noun", "バツ"
    if "ブラシ" in text:
        return "noun", "ブラシ"
    if "引き出し" in text:
        return "noun", "引き出し"
    if "下さい" in text:
        return "noun", "下さい"
    if target in ["シャャミー", "物コャミ"]:
        return "noun", "クシャミ"
    if "マヨネーズ" in text:
        return "noun", "マヨネーズ"
    if "マヨ" in text:
        return "noun", "マヨ"
    if "チップス" in text:
        return "noun", "チップス"
    if "ゴロンっ" in text:
        return "noun", "ゴロンっ"
    if "モチーンっ" in text:
        return "noun", "モチーンっ"
    if "人っ" == text:
        return "noun", "人"
    if text == "掻く":
        return "part", "かい"
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
        return "sconj", "たっ"
    # if text == "て" and upos == "sconj":
    #     return "aux", "て"
    if text == "なさい" and target == "為さる":
        return "aux", "為さい"
    if target == "ちゃ":
        return "sconj", "ちゃ"
    if target == "ない":
        return "aux", "ない"
    if text == "な" and upos == "part":
        return "aux", "な"
    if text == "脱" and upos == "noun":
        return "verb", "脱"
    if text == "よう" and upos == "aux":
        return "aux", "よう"
    if text == "ろ" and upos == "aux" and target == "為る":
        return "aux", "ろ"
    if text == "で":
        return "sconj", "で"

    # if upos == "verb" and "る" in target:
    #     return "verb", target.replace("る","").strip()

    return upos,target


