"""
AVQI Pipeline Module

Acoustic Voice Quality Index calculation.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for AVQI engines to avoid circular imports and improve startup time"""
    if name == 'AVQIEngine':
        from .engine import AVQIEngine
        return AVQIEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['AVQIEngine']
