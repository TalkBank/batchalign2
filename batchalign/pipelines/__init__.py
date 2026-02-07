"""
Pipelines Module

Provides the BatchalignPipeline orchestrator and all processing engines.
Uses lazy loading to avoid importing heavy dependencies until needed.

Engines are typically imported from their subpackages directly, e.g.:
    from batchalign.pipelines.asr import WhisperEngine
    from batchalign.pipelines.fa import Wave2VecFAEngine
"""

def __getattr__(name):
    if name == 'BatchalignPipeline':
        from .pipeline import BatchalignPipeline
        return BatchalignPipeline
    if name == 'BatchalignEngine':
        from .base import BatchalignEngine
        return BatchalignEngine
    if name == 'StanzaEngine':
        from .morphosyntax import StanzaEngine
        return StanzaEngine
    if name == 'StanzaUtteranceEngine':
        from .utterance import StanzaUtteranceEngine
        return StanzaUtteranceEngine

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
