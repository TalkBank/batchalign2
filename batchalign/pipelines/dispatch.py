"""
dispatch.py
Tabulate default packages and options.
"""

from batchalign import (WhisperEngine, WhisperFAEngine, StanzaEngine, RevEngine,
                        NgramRetraceEngine, DisfluencyReplacementEngine, WhisperUTREngine,
                        RevUTREngine, EvaluationEngine, WhisperXEngine, NemoSpeakerEngine,
                        StanzaUtteranceEngine, CorefEngine, Wave2VecFAEngine, SeamlessTranslationModel, TencentEngine)
from batchalign import BatchalignPipeline
from batchalign.models import resolve

from batchalign.utils.config import config_read
from batchalign.errors import *

import logging
L = logging.getLogger("batchalign")

# default for all languages
DEFAULT_PACKAGES = {
    "asr": "whisper",
    "utr": "whisper_utr",
    "fa": "whisper_fa",
    "speaker": "nemo_speaker",
    "morphosyntax": "stanza",
    "disfluency": "replacement",
    "retracing": "ngram",
    "eval": "evaluation",
    "utterance": "stanza_utt",
    "coref": "stanza_coref",
    "translate": "seamless_translate",
}

LANGUAGE_OVERRIDE_PACKAGES = {
    "eng": {
    }
}

def dispatch_pipeline(pkg_str, lang, num_speakers=None, **arg_overrides):
    """Dispatch pipeline with sane defaults.

    Parameters
    ----------
    pkg_str : str
        The user requested pipeline string description.
    lang : str
        Lang code, 3 letters.
    num_speakers : Optional[int]
        Number of speakers

    Returns
    -------
    BatchalignPipeline
        The requested pipeline.
    """
    
    packages = [i.strip() for i in pkg_str.split(",")]

    try:
        config = dict(config_read())
    except ConfigNotFoundError:
        config = {}

    L.debug(f"Initializing packages, got: packages='{packages}' and config='{config}'")


    # create all the engines
    engines = []
    overrides = LANGUAGE_OVERRIDE_PACKAGES.get(lang, {})

    # if asr is in engines but disfluency or retracing is not
    # add them
    if "asr" in packages:
        if "disfluency" not in packages:
            packages.append("disfluency")
        if "retracing" not in packages:
            packages.append("retracing")
        if "utterance" not in packages and resolve("utterance", lang) == None and lang in ["cho", "eng", "ind", "ita",
                                                                                           "jpn", "por", "spa", "tur", "vie"]:
            packages.append("utterance")
    if "fa" in packages:
        if "utr" not in packages:
            packages.append("utr")

    L.info(f"Initializing engines...")
    L.info(f"-------------------------------")
    for key in packages:
        # default is the default
        engine = DEFAULT_PACKAGES.get(key)
        # apply language override
        engine = overrides.get(key, engine)
        # apply user preference
        engine = dict(config.get(key, {})).get("engine", engine)
        # apply argument-level overrides
        engine = arg_overrides.get(key, engine)

        if engine == None:
            raise ValueError(f"Unknown task short name; we can't get a package automatically for that. Provided task: '{key}'.")

        L.info(f"| {key: <12} | {engine:>12} |")
        L.info(f"-------------------------------")

       
        # decode and initialize
        if engine == "whisper":
            engines.append(WhisperEngine(lang=lang))
        elif engine == "whisperx":
            engines.append(WhisperXEngine(lang=lang))
        elif engine == "rev":
            engines.append(RevEngine(lang=lang, num_speakers=num_speakers))
        elif engine == "stanza":
            engines.append(StanzaEngine())
        elif engine == "replacement":
            engines.append(DisfluencyReplacementEngine())
        elif engine == "ngram":
            engines.append(NgramRetraceEngine())
        elif engine == "whisper_fa":
            engines.append(WhisperFAEngine())
        elif engine == "whisper_utr":
            engines.append(WhisperUTREngine(lang=lang))
        elif engine == "rev_utr":
            engines.append(RevUTREngine(lang=lang))
        elif engine == "evaluation":
            engines.append(EvaluationEngine())
        elif engine == "nemo_speaker":
            engines.append(NemoSpeakerEngine(num_speakers=num_speakers))
        elif engine == "stanza_utt":
            engines.append(StanzaUtteranceEngine())
        elif engine == "stanza_coref":
            engines.append(CorefEngine())
        elif engine == "wav2vec_fa":
            engines.append(Wave2VecFAEngine())
        elif engine == "seamless_translate":
            engines.append(SeamlessTranslationModel())
        elif engine == "tencent":
            engines.append(TencentEngine(lang=lang))


    L.debug(f"Done initalizing packages.")
    return BatchalignPipeline(*engines)
    
