"""
resolve.py

Resolve model shortcodes + language to huggingface ID
"""

resolver = {
    "utterance": {
        'eng': "talkbank/CHATUtterance-en",
        "zho": "talkbank/CHATUtterance-zh_CN",
        "yue": "talkbank/CHATUtterance-zh_CN",
    },
    "whisper": {
        'eng': ("talkbank/CHATWhisper-en-large-v1", "openai/whisper-large-v2"),
    }
}

def resolve(model_class, lang_code):
    return resolver.get(model_class, {}).get(lang_code)



