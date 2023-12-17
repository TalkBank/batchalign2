"""
dispatch.py
Tabulate default packages and options.
"""

from batchalign import (WhisperEngine, WhisperFAEngine, StanzaEngine, RevEngine,
                        NgramRetraceEngine, DisfluencyReplacementEngine)
from batchalign import BatchalignPipeline

from batchalign.utils.config import config_read
from batchalign.errors import *

import logging
L = logging.getLogger("batchalign")



# default for all languages
DEFAULT_PACKAGES = {
    "asr": "whisper",
    "fa": "whisper_fa",
    "morphosyntax": "stanza",
    "disfluency": "replacement",
    "retracing": "ngram"
}

LANGUAGE_OVERRIDE_PACKAGES = {
    "eng": {
    }
}

def dispatch_pipeline(pkg_str, lang, n_speakers=None, **arg_overrides):
    """Dispatch pipeline with sane defaults.

    Parameters
    ----------
    pkg_str : str
        The user requested pipeline string description.
    lang : str
        Lang code, 3 letters.
    n_speakers : Optional[int]
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

        L.info(f"| {key: <12} | {engine:>12} |")
        L.info(f"-------------------------------")

       
        # decode and initialize
        if engine == "whisper":
            engines.append(WhisperEngine(lang_code=lang))
        elif engine == "rev":
            engines.append(RevEngine(lang_code=lang, num_speakers=n_speakers))
        elif engine == "stanza":
            engines.append(StanzaEngine())
        elif engine == "replacement":
            engines.append(DisfluencyReplacementEngine())
        elif engine == "ngram":
            engines.append(NgramRetraceEngine())
        elif engine == "whisper_fa":
            engines.append(WhisperFAEngine())

    L.debug(f"Done initalizing packages.")
    return BatchalignPipeline(*engines)
    