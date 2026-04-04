"""
Segmentation Pipeline Module

Provides engines for word segmentation of unsegmented text.
Uses lazy loading to avoid importing heavy dependencies until needed.
"""

def __getattr__(name):
    """Lazy loading for segmentation engines."""
    if name == 'CantoneseSegmentationEngine':
        from .cantonese_seg import CantoneseSegmentationEngine
        return CantoneseSegmentationEngine
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['CantoneseSegmentationEngine']
