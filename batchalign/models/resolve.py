"""
resolve.py

Resolve model shortcodes + language to huggingface ID
"""

from typing import Any

resolver: dict[str, dict[str, str | tuple[str, str]]] = {
    "utterance": {
        'eng': "talkbank/CHATUtterance-en",
        "zho": "talkbank/CHATUtterance-zh_CN",
        "yue": "PolyU-AngelChanLab/Cantonese-Utterance-Segmentation",
    },
    "whisper": {
        'eng': ("talkbank/CHATWhisper-en", "openai/whisper-large-v2"),
        'yue': ("alvanlii/whisper-small-cantonese", "alvanlii/whisper-small-cantonese"),
        "heb": ("ivrit-ai/whisper-large-v3", "ivrit-ai/whisper-large-v3")
    }
}

def resolve(model_class: str, lang_code: str) -> str | tuple[str, str] | None:
    class_resolver = resolver.get(model_class)
    if class_resolver is None:
        return None
    return class_resolver.get(lang_code)



