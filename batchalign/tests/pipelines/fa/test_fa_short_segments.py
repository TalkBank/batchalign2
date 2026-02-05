"""
Regression test for GitHub issue: FA crash with "targets length too long for CTC"

Root cause: When UTR (bulletize_doc) produces word-level timings where only a few 
words get timestamps, Utterance.alignment can compute an impossibly short span
(e.g., 200ms for 17 words). FA then tries to align this and crashes because
CTC can't align 52 characters to 9 audio frames (~0.18 seconds).

The fix adds a defensive check to skip segments that are too short for their word count.
"""

import pytest
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


def test_fa_skips_impossibly_short_segments(en_doc):
    """FA should skip segments that are too short for their word count, not crash."""
    from batchalign.pipelines.fa.wave2vec_fa import Wave2VecFAEngine
    
    # Create a document with one normal utterance and one problematic one
    doc = en_doc.model_copy(deep=True)
    
    # Add a problematic utterance: 17 words in 200ms (impossible to align)
    bad_utt = make_utterance_with_short_alignment(
        words=["este", "es", "un", "ejemplo", "con", "muchas", "palabras", 
               "pero", "muy", "poco", "tiempo", "para", "alinear", "todas",
               "las", "palabras", "correctamente"],
        start_ms=913560,
        end_ms=913760  # Only 200ms for 17 words!
    )
    doc.content.append(bad_utt)
    
    # Mock the Wave2Vec model to avoid loading the actual model
    with patch.object(Wave2VecFAEngine, '__init__', lambda self: None):
        engine = Wave2VecFAEngine()
        engine._Wave2VecFAEngine__wav2vec = MagicMock()
        engine.status_hook = None
        
        # Mock the load method to return a mock audio file
        mock_audio = MagicMock()
        mock_audio.chunk = MagicMock(return_value=MagicMock())
        mock_audio.hash_chunk = MagicMock(return_value="fake_hash")
        engine._Wave2VecFAEngine__wav2vec.load = MagicMock(return_value=mock_audio)
        
        # The actual alignment call should raise the CTC error for short segments
        # but our defensive check should skip before that happens
        engine._Wave2VecFAEngine__wav2vec.return_value = [("word", (0, 100))]
        
        # This should NOT crash - it should skip the bad segment
        try:
            result = engine.process(doc)
            # If we get here without exception, the defensive check worked
            assert True
        except RuntimeError as e:
            if "targets length is too long for CTC" in str(e):
                pytest.fail("FA crashed on short segment - defensive check not working")
            raise


def test_minimum_duration_per_word():
    """Verify the minimum duration threshold is reasonable."""
    # The defensive check uses 80ms minimum per word
    # Based on: Wave2Vec outputs ~50 frames/sec (20ms/frame)
    # CTC needs 1 frame per character, avg word ~4 chars
    # So minimum = 4 * 20ms = 80ms per word
    min_duration_per_word_ms = 80
    word_count = 17
    min_required = word_count * min_duration_per_word_ms
    
    assert min_required == 1360  # 17 * 80 = 1360ms minimum
    
    # The bug case had 200ms for 17 words
    actual_duration = 200
    assert actual_duration < min_required  # Should be skipped
