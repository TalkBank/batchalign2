"""
dispatch.py
Tabulate default packages and options.
"""

from batchalign.models import resolve

from batchalign.utils.config import config_read
from batchalign.errors import *

import logging
L = logging.getLogger("batchalign")

# default for all languages
DEFAULT_PACKAGES = {
    "asr": "whisper_oai",
    "utr": "whisper_utr",
    "fa": "whisper_fa",
    "speaker": "pyannote",
    "morphosyntax": "stanza",
    "disfluency": "replacement",
    "retracing": "ngram",
    "eval": "evaluation",
    "utterance": "stanza_utt",
    "coref": "stanza_coref",
    "translate": "gtrans",
    "opensmile": "opensmile_egemaps",
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
            from batchalign.pipelines.asr import WhisperEngine
            engines.append(WhisperEngine(lang=lang))
        elif engine == "whisperx":
            from batchalign.pipelines.asr import WhisperXEngine
            engines.append(WhisperXEngine(lang=lang))
        elif engine == "rev":
            from batchalign.pipelines.asr import RevEngine
            engines.append(RevEngine(lang=lang, num_speakers=num_speakers))
        elif engine == "stanza":
            from batchalign.pipelines.morphosyntax import StanzaEngine
            engines.append(StanzaEngine())
        elif engine == "replacement":
            from batchalign.pipelines.cleanup import DisfluencyReplacementEngine
            engines.append(DisfluencyReplacementEngine())
        elif engine == "ngram":
            from batchalign.pipelines.cleanup import NgramRetraceEngine
            engines.append(NgramRetraceEngine())
        elif engine == "whisper_fa":
            from batchalign.pipelines.fa import WhisperFAEngine
            engines.append(WhisperFAEngine())
        elif engine == "whisper_utr":
            from batchalign.pipelines.utr import WhisperUTREngine
            engines.append(WhisperUTREngine(lang=lang))
        elif engine == "rev_utr":
            from batchalign.pipelines.utr import RevUTREngine
            engines.append(RevUTREngine(lang=lang))
        elif engine == "evaluation":
            from batchalign.pipelines.analysis import EvaluationEngine
            engines.append(EvaluationEngine())
        elif engine == "nemo_speaker":
            from batchalign.pipelines.speaker import NemoSpeakerEngine
            engines.append(NemoSpeakerEngine(num_speakers=num_speakers))
        elif engine == "stanza_utt":
            from batchalign.pipelines.utterance import StanzaUtteranceEngine
            engines.append(StanzaUtteranceEngine())
        elif engine == "stanza_coref":
            from batchalign.pipelines.morphosyntax import CorefEngine
            engines.append(CorefEngine())
        elif engine == "wav2vec_fa":
            from batchalign.pipelines.fa import Wave2VecFAEngine
            engines.append(Wave2VecFAEngine())
        elif engine == "seamless_translate":
            from batchalign.pipelines.translate import SeamlessTranslationModel
            engines.append(SeamlessTranslationModel())
        elif engine == "gtrans":
            from batchalign.pipelines.translate import GoogleTranslateEngine
            engines.append(GoogleTranslateEngine())
        elif engine == "whisper_oai":
            from batchalign.pipelines.asr import OAIWhisperEngine
            engines.append(OAIWhisperEngine(lang=lang))
        elif engine == "pyannote":
            from batchalign.pipelines.diarization import PyannoteEngine
            engines.append(PyannoteEngine())
        elif engine == "opensmile_egemaps":
            from batchalign.pipelines.opensmile import OpenSMILEEngine
            engines.append(OpenSMILEEngine(feature_set='eGeMAPSv02'))
        elif engine == "opensmile_gemaps":
            from batchalign.pipelines.opensmile import OpenSMILEEngine
            engines.append(OpenSMILEEngine(feature_set='GeMAPSv01b'))
        elif engine == "opensmile_compare":
            from batchalign.pipelines.opensmile import OpenSMILEEngine
            engines.append(OpenSMILEEngine(feature_set='ComParE_2016'))
        elif engine == "opensmile_eGeMAPSv01b":
            from batchalign.pipelines.opensmile import OpenSMILEEngine
            engines.append(OpenSMILEEngine(feature_set='eGeMAPSv01b'))


    L.debug(f"Done initalizing packages.")
    from batchalign.pipelines.pipeline import BatchalignPipeline
    return BatchalignPipeline(*engines)
