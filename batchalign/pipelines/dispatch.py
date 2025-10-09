"""
dispatch.py
Resolves requested packages for automatic model selection, downloading, loading, and caching
"""

from batchalign.pipelines import (WhisperEngine, RevEngine, NgramRetraceEngine, 
                        NgramRetraceEngine, DisfluencyReplacementEngine, WhisperUTREngine,
                        RevUTREngine, EvaluationEngine, WhisperXEngine, NemoSpeakerEngine,
                        StanzaUtteranceEngine, CorefEngine, Wave2VecFAEngine, SeamlessTranslationModel,
                        GoogleTranslateEngine, OAIWhisperEngine, PyannoteEngine, OpenSMILEEngine)
from batchalign import BatchalignPipeline
from batchalign.models import resolve

import logging as L 
baL = L.getLogger('batchalign')

# default packages to use for each of the different operations
# to modify a package, pass in its name as a kwarg to dispatch
DEFAULT_PACKAGES = {
    "asr": "whisperx",
    "fa": "whisper_fa",
    "retrace": "ngram_retrace",
    "ng": "ngram_retrace",
    "disfluency": "disfluency",
    "cleanup": "disfluency",
    "utr": "whisper_utr",
    "eval": "eval",
    "speaker": "nemo_speaker",
    "utterance": "stanza_utt",
    "coref": "stanza_coref",
    "translate": "gtrans",
    "opensmile": "opensmile_egemaps",
}

LANGUAGE_OVERRIDE_PACKAGES = {
    "yue": {
        "asr": "whisper",
        "fa": "whisper_fa"
    }
}

def dispatch_pipeline(pkg_str, lang, num_speakers=None, **arg_overrides):
    """Resolve packages from given information.
    
    Parameters
    ----------
    pkg_str : str
        Comma seperated string of packages to use.
        If not overrided, each package will resolve to a
        default model for that operation. 
    lang : str
        ISO-639-3 language code (eng, spa, etc.)
    num_speakers : int
        Number of speakers; if there is one speaker, speaker
        diarization will not be performed.
    arg_overrides: dict
        Dictionary of arguments to override. Each argument is
        a package name, and the value is the name of a model to
        use instead. For instance, `asr='rev'` will use Rev.ai
        instead of the default ASR model.

    Returns
    -------
    BatchalignPipeline
        A fully configured BatchalignPipeline with all utilities
        requested.

    """

    # resolve language overrides
    overrides = LANGUAGE_OVERRIDE_PACKAGES.get(lang, {})

    L.debug(f"Got packages string {pkg_str} with language {lang}; attempting to resolve.")

    engines = []
    for pkg in pkg_str.split(","):
        # see if user requested an override for this package name
        if pkg in arg_overrides:
            engine = arg_overrides[pkg]
        # see if the language requested any overrides
        elif pkg in overrides:
            engine = overrides[pkg]
        # default to the defaults
        else:
            engine = DEFAULT_PACKAGES[pkg]

        L.debug(f"Attempting to resolve package {pkg} to model {engine}...")

        # having figured out the model name, initialize the
        # actual python class that does the processing

        # each engine which needs a .pt or model loading requires
        # resolve to be called. having called that, you initialize it
        if engine == "wave2vec_fa":
            model = resolve(engine, lang=lang)
            from batchalign.models.wave2vec import infer_fa
            engines.append(Wave2VecFAEngine(model, infer_fa))
        elif engine == "whisper_fa":
            model = resolve(engine, lang=lang)
            from batchalign.models.whisper import infer_fa
            engines.append(Wave2VecFAEngine(model, infer_fa))
        # in some special cases, the engine IS a model which we downloaded
        # so in those cases, instead of calling the model loader we just
        # call the init of the engine
        elif engine == "ngram_retrace":
            model = resolve(engine, lang=lang)
            engines.append(NgramRetraceEngine(model, lang))
        elif engine == "disfluency":
            engines.append(DisfluencyReplacementEngine(lang))
        elif engine == "stanza_utt":
            engines.append(StanzaUtteranceEngine(lang))
        elif engine == "stanza_coref":
            engines.append(CorefEngine(lang))
        elif engine == "eval":
            engines.append(EvaluationEngine())
        elif engine == "whisper_utr":
            engines.append(WhisperUTREngine(lang=lang))
        elif engine == "rev_utr":
            engines.append(RevUTREngine())
        elif engine == "nemo_speaker":
            engines.append(NemoSpeakerEngine(num_speakers))
        elif engine == "gtrans":
            engines.append(GoogleTranslateEngine())
        elif engine == "seamless":
            engines.append(SeamlessTranslationModel())
        elif engine == "rev":
            engines.append(RevEngine(lang))
        elif engine == "whisper":
            engines.append(WhisperEngine(lang))
        elif engine == "whisperx":
            engines.append(WhisperXEngine(lang))
        elif engine == "whisper_oai":
            engines.append(OAIWhisperEngine())
        elif engine == "pyannote":
            engines.append(PyannoteEngine())
        elif engine == "opensmile_egemaps":
            engines.append(OpenSMILEEngine(feature_set='eGeMAPSv02'))
        elif engine == "opensmile_gemaps":
            engines.append(OpenSMILEEngine(feature_set='GeMAPSv01b'))
        elif engine == "opensmile_compare":
            engines.append(OpenSMILEEngine(feature_set='ComParE_2016'))
        elif engine == "opensmile_eGeMAPSv01b":
            engines.append(OpenSMILEEngine(feature_set='eGeMAPSv01b'))

    L.debug(f"Done initalizing packages.")
    return BatchalignPipeline(*engines)