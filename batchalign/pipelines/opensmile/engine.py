"""
OpenSMILE Engine for Batchalign2
Audio feature extraction using the openSMILE toolkit
"""

import opensmile
from opensmile import FeatureSet, FeatureLevel
import pandas as pd
from pathlib import Path
import logging
from typing import Dict

from batchalign.pipelines.base import BatchalignEngine
from batchalign.document import Task, TaskType, Document

L = logging.getLogger('batchalign')

class OpenSMILEEngine(BatchalignEngine):
    """Engine for extracting openSMILE audio features."""

    # Map string names to FeatureSet enums
    FEATURE_SET_MAP = {
        'eGeMAPSv02': FeatureSet.eGeMAPSv02,
        'eGeMAPSv01b': FeatureSet.eGeMAPSv01b,
        'GeMAPSv01b': FeatureSet.GeMAPSv01b,
        'ComParE_2016': FeatureSet.ComParE_2016,
    }

    def __init__(self, feature_set: str = 'eGeMAPSv02', 
                 feature_level: str = 'functionals'):
        super().__init__()
        self._tasks = [Task.FEATURE_EXTRACT]

        self.feature_set = feature_set
        self.feature_level = feature_level

        try:
            feature_set_enum = self.FEATURE_SET_MAP.get(feature_set, FeatureSet.eGeMAPSv02)
            feature_level_enum = FeatureLevel.Functionals if feature_level == 'functionals' else FeatureLevel.LowLevelDescriptors
            
            self.smile = opensmile.Smile(
                feature_set=feature_set_enum,
                feature_level=feature_level_enum,
            )
            L.debug(f"OpenSMILE initialized with {feature_set}")
        except Exception as e:
            L.error(f"Failed to initialize openSMILE: {e}")
            raise

    @property
    def tasks(self):
        return self._tasks

    def analyze(self, doc: Document, **kwargs) -> Dict:
        """
        Extract openSMILE features from Document.
        
        Args:
            doc: Document with media attached
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

        try:
            L.info(f"Extracting features from: {Path(actual_audio_path).name}")
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

            results = {
                'feature_set': self.feature_set,
                'feature_level': self.feature_level,
                'num_features': num_features,
                'duration_segments': duration_segments,
                'audio_file': str(actual_audio_path),
                'features_sample': first_row_features,
                'success': True,
                'features_df': results_df,
            }

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
                'success': False
            }

    def get_available_feature_sets(self) -> list:
        """Return list of available feature sets."""
        return list(self.FEATURE_SET_MAP.keys())

    def get_feature_set_info(self, feature_set: str) -> dict:
        """Get information about a specific feature set."""
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