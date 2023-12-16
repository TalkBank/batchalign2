"""
dispatch.py
Default packages, and package string dispatch.
"""

# default for all languages
DEFAULT = {
    "asr": "whisper",
    "fa": "whisper",
}

OVERRIDES = {
    "eng": {
        "asr": "rev"
    }
}

