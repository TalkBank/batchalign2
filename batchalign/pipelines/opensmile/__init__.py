"""
OpenSMILE Pipeline Module

Feature extraction using OpenSMILE toolkit.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for OpenSMILE engines to avoid circular imports and improve startup time"""
    if name == 'OpenSMILEEngine':
        from .engine import OpenSMILEEngine
        return OpenSMILEEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['OpenSMILEEngine']
