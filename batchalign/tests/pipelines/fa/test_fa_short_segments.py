"""
Regression test for GitHub issue: FA crash with "targets length too long for CTC"

Root cause: When UTR (bulletize_doc) produces word-level timings where only a few 
words get timestamps, Utterance.alignment can compute an impossibly short span
(e.g., 200ms for 17 words). FA then tries to align this and crashes because
CTC can't align 52 characters to 9 audio frames (~0.18 seconds).

The fix catches alignment exceptions and skips failed segments with a warning,
rather than crashing. This restores the original behavior (pre-7ece607) but with
proper logging instead of silent failure.

History:
- Pre-7ece607: bare `except:` silently swallowed all errors including CTC failures
- 7ece607: changed to `except (IndexError, ValueError):` which let RuntimeError through
- Fix: changed to `except Exception as e:` with logging to catch CTC errors gracefully
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from batchalign.document import Document, Utterance, Form, Media, MediaType, TokenType


def make_utterance_with_short_alignment(words: list[str], start_ms: int, end_ms: int) -> Utterance:
    """Create an utterance where only first and last words have timing.
    
    This simulates what happens when bulletize_doc's DP alignment only matches
    a few words, resulting in Utterance.alignment returning a short span
    despite having many words.
    """
    content = []
    for i, word in enumerate(words):
        form = Form(text=word, type=TokenType.REGULAR)
        # Only first and last words get timing (simulating sparse DP matches)
        if i == 0:
            form.time = (start_ms, start_ms + 50)
        elif i == len(words) - 1:
            form.time = (end_ms - 50, end_ms)
        # Middle words have no timing (common with poor ASR matches)
        content.append(form)
    
    return Utterance(content=content)


def test_utterance_alignment_from_sparse_word_timings():
    """Verify Utterance.alignment computes span from first/last timed words."""
    # 17 words but only 200ms span (the problematic case from the bug report)
    words = ["word"] * 17
    utt = make_utterance_with_short_alignment(words, start_ms=913560, end_ms=913760)
    
    # alignment should be computed from first and last timed words
    assert utt.alignment is not None
    assert utt.alignment == (913560, 913760)
    
    # This is the problematic case: 200ms for 17 words is impossible
    duration_ms = utt.alignment[1] - utt.alignment[0]
    assert duration_ms == 200
    assert len(words) == 17


def test_fa_catches_runtime_error():
    """
    REGRESSION TEST: FA must catch exceptions from alignment failures and continue.
    
    This test verifies that ANY exception during alignment is caught and the
    engine continues processing rather than crashing. This is the resilient
    behavior we want - keep plowing through even if individual segments fail.
    
    This test will FAIL if someone narrows the exception handling (e.g., to
    only catch specific exception types like IndexError, ValueError).
    """
    from batchalign.pipelines.fa.wave2vec_fa import Wave2VecFAEngine
    
    # Test with multiple exception types to ensure broad catching
    exception_types = [
        RuntimeError("targets length is too long for CTC"),
        ValueError("invalid audio data"),
        IndexError("index out of range"),
        TypeError("unexpected type"),
        Exception("generic failure"),
    ]
    
    for exc in exception_types:
        mock_wav2vec = MagicMock()
        mock_wav2vec.side_effect = exc
        
        mock_audio = MagicMock()
        mock_audio.chunk = MagicMock(return_value=MagicMock())
        mock_audio.hash_chunk = MagicMock(return_value="fake_hash")
        
        doc = Document(
            content=[
                Utterance(content=[
                    Form(text="test", type=TokenType.REGULAR, time=(0, 1000)),
                    Form(text="words", type=TokenType.REGULAR, time=(1000, 2000)),
                ])
            ],
            langs=["eng"],
            media=Media(type=MediaType.AUDIO, name="test", url="/fake/path.mp3")
        )
        
        with patch.object(Wave2VecFAEngine, '__init__', lambda self, lang="eng": None):
            engine = Wave2VecFAEngine()
            engine._Wave2VecFAEngine__wav2vec = mock_wav2vec
            engine._Wave2VecFAEngine__wav2vec.load = MagicMock(return_value=mock_audio)
            engine.status_hook = None
            
            # This MUST NOT raise - exceptions should be caught and logged
            try:
                engine.process(doc)
            except Exception as e:
                pytest.fail(
                    f"REGRESSION: {type(exc).__name__} was not caught! "
                    f"FA should catch all exceptions and continue processing. "
                    f"Error: {e}"
                )


def test_fa_logs_warning_on_skip(caplog):
    """
    REGRESSION TEST: FA must log a WARNING when skipping a failed segment.
    
    This ensures users are notified when segments are skipped, rather than
    silent failure (the pre-7ece607 behavior with bare `except:`).
    """
    from batchalign.pipelines.fa.wave2vec_fa import Wave2VecFAEngine
    
    mock_wav2vec = MagicMock()
    mock_wav2vec.side_effect = RuntimeError("targets length is too long for CTC")
    
    mock_audio = MagicMock()
    mock_audio.chunk = MagicMock(return_value=MagicMock())
    mock_audio.hash_chunk = MagicMock(return_value="fake_hash")
    
    doc = Document(
        content=[
            Utterance(content=[
                Form(text="test", type=TokenType.REGULAR, time=(0, 1000)),
                Form(text="words", type=TokenType.REGULAR, time=(1000, 2000)),
            ])
        ],
        langs=["eng"],
        media=Media(type=MediaType.AUDIO, name="test", url="/fake/path.mp3")
    )
    
    with patch.object(Wave2VecFAEngine, '__init__', lambda self, lang="eng": None):
        engine = Wave2VecFAEngine()
        engine._Wave2VecFAEngine__wav2vec = mock_wav2vec
        engine._Wave2VecFAEngine__wav2vec.load = MagicMock(return_value=mock_audio)
        engine.status_hook = None
        
        with caplog.at_level(logging.WARNING, logger="batchalign"):
            engine.process(doc)
        
        # Check that a warning was logged about skipping
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        skip_warnings = [m for m in warning_messages if "Skipping segment" in m]
        
        assert len(skip_warnings) > 0, (
            "REGRESSION: No warning logged when skipping failed segment! "
            "The fix should log: L.warning(f'Skipping segment ...: {e}')"
        )


def test_ctc_constraint_explanation():
    """Document the CTC constraint that causes alignment to fail on short segments.
    
    CTC (Connectionist Temporal Classification) requires:
        num_output_frames >= num_target_characters
    
    Wave2Vec outputs ~50 frames/sec, so:
    - 200ms audio → ~10 frames
    - 17 words (~52 characters) need 52+ frames
    - 10 < 52 → CTC fails
    
    The fix catches this failure and skips the segment with a warning.
    """
    # Wave2Vec frame rate
    frames_per_second = 50
    ms_per_frame = 1000 / frames_per_second
    assert ms_per_frame == 20  # 20ms per frame
    
    # The bug case
    audio_duration_ms = 200
    num_frames = audio_duration_ms / ms_per_frame
    assert num_frames == 10
    
    # Average ~3 chars per word, 17 words
    num_words = 17
    avg_chars_per_word = 3
    num_chars = num_words * avg_chars_per_word
    assert num_chars == 51
    
    # CTC constraint violated
    assert num_frames < num_chars  # This is why it fails
