"""
Analysis Pipeline Module

Provides engines for analyzing and evaluating transcripts.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for analysis engines to avoid circular imports and improve startup time"""
    if name == 'EvaluationEngine':
        from .eval import EvaluationEngine
        return EvaluationEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['EvaluationEngine']
