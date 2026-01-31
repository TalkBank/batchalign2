"""
Translate Pipeline Module

Provides engines for translating transcripts between languages.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for translation engines to avoid circular imports and improve startup time"""
    if name == 'SeamlessTranslationModel':
        from .seamless import SeamlessTranslationModel
        return SeamlessTranslationModel
    elif name == 'GoogleTranslateEngine':
        from .gtrans import GoogleTranslateEngine
        return GoogleTranslateEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['SeamlessTranslationModel', 'GoogleTranslateEngine']
