"""
OpenSMILE Engine for Batchalign2 - M1 Mac Compatible Version
Audio feature extraction using the openSMILE toolkit
"""

import opensmile
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import platform

from batchalign.pipelines.base import BatchalignEngine
from batchalign.document import Task, TaskType, Document

L = logging.getLogger('batchalign')

class OpenSMILEEngine(BatchalignEngine):
    """Engine for extracting openSMILE audio features."""

    def __init__(self, feature_set: str = 'eGeMAPSv02', 
                 feature_level: str = 'functionals'):
        super().__init__()
        self._tasks = [Task.FEATURE_EXTRACT]

        self.feature_set = feature_set
        self.feature_level = feature_level

        self.is_m1_mac = (platform.system() == 'Darwin' and 
                         platform.processor() == 'arm')

        try:
            if self.is_m1_mac:
                L.info("M1 Mac detected - using default openSMILE configuration")
                self.smile = opensmile.Smile()
                self._requested_feature_set = feature_set
            else:
                self.smile = opensmile.Smile(
                    feature_set=feature_set,
                    feature_level=feature_level,
                )
            L.debug(f"OpenSMILE initialized (M1 compatibility mode: {self.is_m1_mac})")
        except Exception as e:
            L.error(f"Failed to initialize openSMILE: {e}")
            raise

    @property
    def tasks(self):
        return self._tasks

    def analyze(self, doc: Document, feature_set: str = None, **kwargs) -> Dict:
        """
        Extract openSMILE features from Document.
        
        Args:
            doc: Document with media attached
            feature_set: Feature set to use (ignored on M1 Mac)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with extraction results and metadata
        """

        if not doc.media or not doc.media.url:
            return {
                'error': 'Document has no media attached',
                'success': False
            }

        actual_audio_path = doc.media.url

        if feature_set and feature_set != self.feature_set:
            if self.is_m1_mac:
                L.warning(f"Feature set switching not supported on M1 Mac - using default features instead of {feature_set}")
            else:
                L.info(f"Switching feature set from {self.feature_set} to {feature_set}")
                try:
                    self.feature_set = feature_set
                    self.smile = opensmile.Smile(
                        feature_set=feature_set,
                        feature_level=self.feature_level,
                    )
                except Exception as e:
                    L.error(f"Failed to switch to feature set {feature_set}: {e}")
                    return {
                        'feature_set': self.feature_set,
                        'num_features': 0,
                        'error': f"Feature set switch failed: {str(e)}",
                        'success': False
                    }

        try:
            L.info(f"Extracting features from: {Path(actual_audio_path).name}")
            if self.is_m1_mac:
                L.info("Using M1-compatible default feature set (eGeMAPSv02 equivalent)")
            else:
                L.info(f"Using {self.feature_set} feature set")

            features_df = self.smile.process_file(actual_audio_path)

            if features_df is None or features_df.empty:
                raise ValueError("Feature extraction returned empty results")

            results_df = features_df.T

            num_features = len(features_df.columns)
            duration_segments = len(features_df)

            first_row_features = {}
            if duration_segments > 0:
                first_row_features = features_df.iloc[0].to_dict()

            actual_feature_set = self.feature_set
            if self.is_m1_mac:
                actual_feature_set = "M1-default (eGeMAPSv02-like)"

            results = {
                'feature_set': actual_feature_set,
                'feature_level': self.feature_level,
                'num_features': num_features,
                'duration_segments': duration_segments,
                'audio_file': str(actual_audio_path),
                'features_sample': first_row_features,
                'success': True,
                'm1_compatibility_mode': self.is_m1_mac,
                'features_df': results_df,
            }

            if self.is_m1_mac and hasattr(self, '_requested_feature_set'):
                results['requested_feature_set'] = self._requested_feature_set
                results['warning'] = f"M1 Mac compatibility: used default features instead of {self._requested_feature_set}"

            L.info(f"Successfully extracted {num_features} features from {duration_segments} segments")
            return results

        except Exception as e:
            L.error(f"Error extracting openSMILE features from {actual_audio_path}: {e}")
            return {
                'feature_set': self.feature_set,
                'feature_level': self.feature_level,
                'num_features': 0,
                'duration_segments': 0,
                'audio_file': str(actual_audio_path),
                'error': str(e),
                'success': False,
                'm1_compatibility_mode': self.is_m1_mac
            }

    def get_available_feature_sets(self) -> list:
        """Return list of available feature sets (limited on M1 Mac)."""
        if self.is_m1_mac:
            return ['M1-default (eGeMAPSv02-like)']
        return [
            'eGeMAPSv02',
            'eGeMAPSv01b', 
            'GeMAPSv01b',
            'ComParE_2016'
        ]

    def get_feature_set_info(self, feature_set: str) -> dict:
        """Get information about a specific feature set."""
        if self.is_m1_mac:
            return {
                'description': 'M1 Mac compatible default feature set (similar to eGeMAPSv02)',
                'num_features': 'Variable',
                'recommended_for': 'General audio analysis on Apple Silicon'
            }

        info = {
            'eGeMAPSv02': {
                'description': 'Extended Geneva Minimalistic Acoustic Parameter Set v02',
                'num_features': 88,
                'recommended_for': 'General emotion and paralinguistic analysis'
            },
            'eGeMAPSv01b': {
                'description': 'Extended Geneva Minimalistic Acoustic Parameter Set v01b', 
                'num_features': 88,
                'recommended_for': 'Emotion recognition, clinical assessment'
            },
            'GeMAPSv01b': {
                'description': 'Geneva Minimalistic Acoustic Parameter Set v01b',
                'num_features': 62,
                'recommended_for': 'Basic paralinguistic analysis'
            },
            'ComParE_2016': {
                'description': 'Computational Paralinguistics Challenge 2016 feature set',
                'num_features': 6373,
                'recommended_for': 'Comprehensive analysis (large feature space)'
            }
        }
        return info.get(feature_set, {'description': 'Unknown feature set', 'num_features': 'Unknown'})