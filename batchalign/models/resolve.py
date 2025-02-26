"""
resolve.py

Resolve model shortcodes + language to huggingface ID
"""

resolver = {
    "utterance": {
        'eng': "talkbank/CHATUtterance-en",
        "zho": "talkbank/CHATUtterance-zh_CN",
        "yue": "PolyU-AngelChanLab/Cantonese-Utterance-Segmentation",
    },
    "whisper": {
        'eng': ("talkbank/CHATWhisper-en-large-v1", "openai/whisper-large-v2"),
        'yue': ("alvanlii/whisper-small-cantonese", "alvanlii/whisper-small-cantonese"),
    }
}

def resolve(model_class, lang_code):
    return resolver.get(model_class, {}).get(lang_code)



