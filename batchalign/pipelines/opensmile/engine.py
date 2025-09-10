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
from batchalign.document import Task, TaskType

L = logging.getLogger('batchalign')

class OpenSMILEEngine(BatchalignEngine):
    """Engine for extracting openSMILE audio features."""

    def __init__(self, feature_set: str = 'eGeMAPSv02', 
                 feature_level: str = 'functionals'):
        super().__init__()
        # Use existing feature extraction task type
        self._tasks = [Task.FEATURE_EXTRACT]

        self.feature_set = feature_set
        self.feature_level = feature_level

        # Check if we're on M1 Mac and handle accordingly
        self.is_m1_mac = (platform.system() == 'Darwin' and 
                         platform.processor() == 'arm')

        # Initialize openSMILE with M1 compatibility handling
        try:
            if self.is_m1_mac:
                # On M1 Mac, use default initialization to avoid compatibility issues
                L.warning("M1 Mac detected - using default openSMILE configuration due to known compatibility issues")
                self.smile = opensmile.Smile()
                # Store requested feature set for error reporting
                self._requested_feature_set = feature_set
            else:
                # Standard initialization for other platforms
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

    def analyze(self, audio_file: str, output_file: str = None, 
               feature_set: str = None, output_format: str = 'csv', **kwargs) -> Dict:
        """
        Extract openSMILE features from audio file for dispatch system.
        
        Args:
            audio_file: Path to input audio file (can be a Document with media or string path)
            output_file: Path to output feature file  
            feature_set: Feature set to use (ignored on M1 Mac)
            output_format: Output format ('csv' or 'tsv')
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with extraction results and metadata
        """

        # Handle Document input from dispatch system
        if hasattr(audio_file, 'media') and audio_file.media:
            actual_audio_path = audio_file.media.url
        elif isinstance(audio_file, str):
            actual_audio_path = audio_file
        else:
            return {
                'error': 'Invalid audio input - expected file path or Document with media',
                'success': False
            }

        # Handle feature set switching (not supported on M1 Mac)
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

            # Extract features using openSMILE
            features_df = self.smile.process_file(actual_audio_path)

            if features_df is None or features_df.empty:
                raise ValueError("Feature extraction returned empty results")

            # Handle output file creation for dispatch system
            if output_file is None:
                audio_path = Path(actual_audio_path)
                ext = '.csv' if output_format == 'csv' else '.tsv'
                output_file = str(audio_path.with_suffix(f'.opensmile{ext}'))

            # Determine output format and save
            output_path = Path(output_file)
            if output_format == 'csv' or output_path.suffix.lower() == '.csv':
                # Transpose for better CSV format (features as rows)
                features_df.T.to_csv(output_file, header=['value'], index_label='feature')
            elif output_format == 'tsv' or output_path.suffix.lower() == '.tsv':
                features_df.to_csv(output_file, sep='\t', index=True)
            else:
                # Default to CSV
                features_df.T.to_csv(output_file, header=['value'], index_label='feature')

            # Prepare summary results
            num_features = len(features_df.columns)
            duration_segments = len(features_df)

            # Get first row of features for summary (if available)
            first_row_features = {}
            if duration_segments > 0:
                first_row_features = features_df.iloc[0].to_dict()

            # Determine actual feature set used
            actual_feature_set = self.feature_set
            if self.is_m1_mac:
                actual_feature_set = "M1-default (eGeMAPSv02-like)"

            results = {
                'feature_set': actual_feature_set,
                'feature_level': self.feature_level,
                'num_features': num_features,
                'duration_segments': duration_segments,
                'output_file': str(output_file),
                'audio_file': str(actual_audio_path),
                'features_sample': first_row_features,
                'success': True,
                'm1_compatibility_mode': self.is_m1_mac
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
