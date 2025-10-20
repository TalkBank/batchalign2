"""
AVQI Engine for Batchalign2
Acoustic Voice Quality Index calculation for voice quality assessment
"""

import parselmouth
import numpy as np
from parselmouth.praat import call
import re
from typing import Tuple, Dict, Optional
import os
from pathlib import Path
import logging
import torchaudio

from batchalign.pipelines.base import BatchalignEngine
from batchalign.document import Task, Document


L = logging.getLogger('batchalign')


class AVQIEngine(BatchalignEngine):
    """Engine for calculating Acoustic Voice Quality Index (AVQI)."""

    def __init__(self):
        super().__init__()
        self._tasks = [Task.FEATURE_EXTRACT]

    @property
    def tasks(self):
        return self._tasks

    def extract_voiced_segments(self, sound):
        """Extract voiced segments from audio."""
        original = call(sound, "Copy", "original")
        sampling_rate = call(original, "Get sampling frequency")
        onlyVoice = call("Create Sound", "onlyVoice", 0, 0.001, sampling_rate, "0")
        textgrid = call(
            original,
            "To TextGrid (silences)",
            50,
            0.003,
            -25,
            0.1,
            0.1,
            "silence",
            "sounding",
        )
        intervals = call(
            [original, textgrid],
            "Extract intervals where",
            1,
            False,
            "does not contain",
            "silence",
        )
        onlyLoud = call(intervals, "Concatenate")
        globalPower = call(onlyLoud, "Get power in air")
        voicelessThreshold = globalPower * 0.3
        signalEnd = call(onlyLoud, "Get end time")
        windowBorderLeft = call(onlyLoud, "Get start time")
        windowWidth = 0.03
        while windowBorderLeft + windowWidth <= signalEnd:
            part = call(
                onlyLoud,
                "Extract part",
                windowBorderLeft,
                windowBorderLeft + windowWidth,
                "Rectangular",
                1.0,
                False,
            )
            partialPower = call(part, "Get power in air")
            if partialPower > voicelessThreshold:
                try:
                    start = 0.0025
                    startZero = call(part, "Get nearest zero crossing", start)
                    if startZero is not None and not np.isinf(startZero):
                        onlyVoice = call([onlyVoice, part], "Concatenate")
                except:
                    pass
            windowBorderLeft += 0.03
        return onlyVoice

    def calculate_avqi_features(self, cs_file, sv_file):
        """Calculate AVQI score and features from continuous speech and sustained vowel files."""
        cs_sound = parselmouth.Sound(cs_file)
        sv_sound = parselmouth.Sound(sv_file)
        cs_filtered = call(cs_sound, "Filter (stop Hann band)", 0, 34, 0.1)
        sv_filtered = call(sv_sound, "Filter (stop Hann band)", 0, 34, 0.1)
        voiced_cs = self.extract_voiced_segments(cs_filtered)
        sv_duration = call(sv_filtered, "Get total duration")
        if sv_duration > 3:
            sv_start = sv_duration - 3
            sv_part = call(
                sv_filtered, "Extract part", sv_start, sv_duration, "rectangular", 1, False
            )
        else:
            sv_part = call(sv_filtered, "Copy", "sv_part")
        concatenated = call([voiced_cs, sv_part], "Concatenate")
        powercepstrogram = call(concatenated, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        cpps = call(
            powercepstrogram,
            "Get CPPS",
            False,
            0.01,
            0.001,
            60,
            330,
            0.05,
            "Parabolic",
            0.001,
            0,
            "Straight",
            "Robust",
        )
        ltas = call(concatenated, "To Ltas", 1)
        slope = call(ltas, "Get slope", 0, 1000, 1000, 10000, "energy")
        ltas_copy = call(ltas, "Copy", "ltas_for_tilt")
        try:
            call(ltas_copy, "Compute trend line", 1, 10000)
            tilt = call(ltas_copy, "Get slope", 0, 1000, 1000, 10000, "energy")
            if abs(tilt - slope) < 0.01:
                ltas_copy2 = call(ltas, "Copy", "ltas_for_tilt2")
                call(ltas_copy2, "Compute trend line", 100, 8000)
                tilt = call(ltas_copy2, "Get slope", 0, 1000, 1000, 10000, "energy")
            if abs(tilt - slope) < 0.01:
                tilt = slope + 5.5
        except:
            tilt = slope + 5.5
        pointprocess = call(concatenated, "To PointProcess (periodic, cc)", 50, 400)
        shim_percent = call(
            [concatenated, pointprocess],
            "Get shimmer (local)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
        shim = shim_percent * 100
        shdb = call(
            [concatenated, pointprocess],
            "Get shimmer (local_dB)",
            0,
            0,
            0.0001,
            0.02,
            1.3,
            1.6,
        )
        pitch = call(
            concatenated,
            "To Pitch (cc)",
            0,
            75,
            15,
            False,
            0.03,
            0.45,
            0.01,
            0.35,
            0.14,
            600,
        )
        pointprocess2 = call([concatenated, pitch], "To PointProcess (cc)")
        voice_report = call(
            [concatenated, pitch, pointprocess2],
            "Voice report",
            0,
            0,
            75,
            600,
            1.3,
            1.6,
            0.03,
            0.45,
        )
        hnr_match = re.search(
            r"Mean harmonics-to-noise ratio:\s*([-+]?\d*\.?\d+)", voice_report
        )
        hnr = float(hnr_match.group(1)) if hnr_match else 0.0
        avqi = (
            4.152
            - (0.177 * cpps)
            - (0.006 * hnr)
            - (0.037 * shim)
            + (0.941 * shdb)
            + (0.01 * slope)
            + (0.093 * tilt)
        ) * 2.8902
        return avqi, {
            "cpps": cpps,
            "hnr": hnr,
            "shimmer_local": shim,
            "shimmer_local_db": shdb,
            "slope": slope,
            "tilt": tilt,
        }

    def analyze(self, doc: Document, **kwargs) -> Dict:
        """
        Analyze audio files and calculate AVQI.
        
        Expects the document to have a .cs audio file, and looks for matching .sv file.
        
        Parameters
        ----------
        doc : Document
            Document with .cs media attached
        **kwargs : dict
            Additional arguments
            
        Returns
        -------
        Dict
            Dictionary containing AVQI score and features
        """
        
        if not doc.media or not doc.media.url:
            return {
                'error': 'Document has no media attached',
                'success': False
            }
        
        cs_file = doc.media.url
        cs_path = Path(cs_file)
        
        # Check if this is a .cs file
        if not '.cs.' in cs_path.name:
            return {
                'error': f'Expected .cs audio file, got {cs_path.name}',
                'success': False
            }
        
        # Find matching .sv file by replacing .cs. with .sv.
        sv_name = cs_path.name.replace('.cs.', '.sv.')
        sv_file = cs_path.parent / sv_name
        
        if not sv_file.exists():
            return {
                'error': f'Matching .sv file not found: {sv_name}',
                'success': False
            }
        
        L.info(f"Calculating AVQI for CS: {cs_path.name}, SV: {sv_name}")

        # Create mono versions of both files
        cs_mono_path = None
        sv_mono_path = None

        try:
            # Convert cs_file to mono
            L.info(f"Converting {cs_path.name} to mono")
            cs_waveform, cs_sample_rate = torchaudio.load(str(cs_file))
            if cs_waveform.shape[0] > 1:
                cs_mono = cs_waveform.mean(dim=0, keepdim=True)
            else:
                cs_mono = cs_waveform

            # Create mono filename: file_name.cs.[extension].mono
            cs_mono_path = cs_path.parent / f"{cs_path.name}.mono.wav"
            torchaudio.save(str(cs_mono_path), cs_mono, cs_sample_rate)

            # Convert sv_file to mono
            L.info(f"Converting {sv_name} to mono")
            sv_waveform, sv_sample_rate = torchaudio.load(str(sv_file))
            if sv_waveform.shape[0] > 1:
                sv_mono = sv_waveform.mean(dim=0, keepdim=True)
            else:
                sv_mono = sv_waveform

            # Create mono filename: file_name.sv.[extension].mono
            sv_mono_path = sv_file.parent / f"{sv_file.name}.mono.wav"
            torchaudio.save(str(sv_mono_path), sv_mono, sv_sample_rate)

            # Calculate AVQI using mono versions
            avqi_score, features = self.calculate_avqi_features(str(cs_mono_path), str(sv_mono_path))

            results = {
                'avqi': avqi_score,
                'cpps': features['cpps'],
                'hnr': features['hnr'],
                'shimmer_local': features['shimmer_local'],
                'shimmer_local_db': features['shimmer_local_db'],
                'slope': features['slope'],
                'tilt': features['tilt'],
                'cs_file': cs_path.name,
                'sv_file': sv_name,
                'success': True
            }

            L.info(f"AVQI Score: {avqi_score:.3f}")
            return results

        except Exception as e:
            L.error(f"Error calculating AVQI: {e}")
            return {
                'avqi': 0.0,
                'cpps': 0.0,
                'hnr': 0.0,
                'shimmer_local': 0.0,
                'shimmer_local_db': 0.0,
                'slope': 0.0,
                'tilt': 0.0,
                'error': str(e),
                'success': False
            }
        finally:
            # Clean up temporary mono files
            if cs_mono_path and cs_mono_path.exists():
                L.info(f"Cleaning up temporary file: {cs_mono_path.name}")
                cs_mono_path.unlink()
            if sv_mono_path and sv_mono_path.exists():
                L.info(f"Cleaning up temporary file: {sv_mono_path.name}")
                sv_mono_path.unlink()
