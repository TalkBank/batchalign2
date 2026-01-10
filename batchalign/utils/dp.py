"""
dp.py
Dynamic Programming Utilities

Generally used for minimum-edit sequence alignment across the program.
This module now uses a Hirschberg-style divide-and-conquer aligner to
produce the same outputs while using linear space and less Python-level
overhead than the previous full-matrix implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional, Sequence

# the target to align against the reference
# carries a payload which will be stitched to maching
# values in the reference
@dataclass
class PayloadTarget:
    key: str # the key to align against the reference
    payload: Any

# the refrenece to which the target will be aligned
@dataclass
class ReferenceTarget:
    key: str # the key to align against the reference
    payload: Any = None

# for extra outputs, which one has the extra
# i.e. "which one is contributing to this Extra" class
class ExtraType(Enum):
    PAYLOAD=0
    REFERENCE=1

class OutputType(Enum):
    MATCH=0
    EXTRA_PAYLOAD=1
    EXTRA_REFERENCE=2
    EXTRA_BOTH=3

# and two classes for the output

# if there is a match between payload-type target and reference
# we return a Match class
@dataclass
class Match:
    key: str
    payload: Any
    reference_payload: Any

# if there is _not_ a match, we return the Extra type, and
# further describe whether or not its an extra sequence in PAYLOAD
# or REFERENCE
@dataclass
class Extra:
    key: str
    extra_type: ExtraType
    payload: Optional[Any] = None


def __serialize_arr(src: Iterable[Any], tgt: Iterable[Any]):
    """Serialize arbitrary sequences into PayloadTarget / ReferenceTarget lists."""

    src_serialized = [PayloadTarget(i, None) for i in src]
    tgt_serialized = [ReferenceTarget(i) for i in tgt]

    return src_serialized, tgt_serialized


def _cost(is_match: bool) -> int:
    return 0 if is_match else 2


def _row_costs(reference: Sequence[ReferenceTarget],
               payload: Sequence[PayloadTarget],
               match_fn: Callable[[Any, Any], bool],
               progress) -> List[int]:
    """Compute only the last DP row for a reference slice vs payload slice."""

    prev = list(range(len(payload)+1))
    for ref_idx, ref_item in enumerate(reference):
        cur = [ref_idx+1]
        for pay_idx, pay_item in enumerate(payload):
            is_match = match_fn(ref_item.key, pay_item.key)
            sub_cost = prev[pay_idx] + _cost(is_match)
            del_cost = prev[pay_idx+1] + 1
            ins_cost = cur[pay_idx] + 1
            cur.append(min(sub_cost, del_cost, ins_cost))
        prev = cur
        if progress:
            progress.update(len(payload))
    return prev


def _align_small(reference: Sequence[ReferenceTarget],
                 payload: Sequence[PayloadTarget],
                 match_fn: Callable[[Any, Any], bool],
                 progress) -> List[Any]:
    """Full-table alignment for small problems to keep recursion simple."""

    rows = len(reference) + 1
    cols = len(payload) + 1
    dp: List[List[tuple]] = [[(0, None, None) for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = (i, OutputType.EXTRA_REFERENCE, (i-1, 0))
    for j in range(1, cols):
        dp[0][j] = (j, OutputType.EXTRA_PAYLOAD, (0, j-1))

    for i in range(1, rows):
        for j in range(1, cols):
            is_match = match_fn(reference[i-1].key, payload[j-1].key)
            dist_sub = dp[i-1][j-1][0] + _cost(is_match)
            dist_del = dp[i-1][j][0] + 1
            dist_ins = dp[i][j-1][0] + 1

            if dist_sub <= dist_del and dist_sub <= dist_ins:
                action = OutputType.MATCH if is_match else OutputType.EXTRA_BOTH
                dp[i][j] = (dist_sub, action, (i-1, j-1))
            elif dist_del <= dist_sub and dist_del <= dist_ins:
                dp[i][j] = (dist_del, OutputType.EXTRA_REFERENCE, (i-1, j))
            else:
                dp[i][j] = (dist_ins, OutputType.EXTRA_PAYLOAD, (i, j-1))

    if progress:
        progress.update((rows-1) * (cols-1))

    output: List[Any] = []
    i, j = rows-1, cols-1
    _, action, prev = dp[i][j]
    while prev:
        ref_index, payload_index = prev
        if action == OutputType.MATCH:
            output.append(Match(reference[ref_index].key,
                                payload[payload_index].payload,
                                reference[ref_index].payload))
        if action in (OutputType.EXTRA_BOTH, OutputType.EXTRA_PAYLOAD):
            output.append(Extra(payload[payload_index].key,
                                ExtraType.PAYLOAD,
                                payload[payload_index].payload))
        if action in (OutputType.EXTRA_BOTH, OutputType.EXTRA_REFERENCE):
            output.append(Extra(reference[ref_index].key,
                                ExtraType.REFERENCE,
                                reference[ref_index].payload))
        i, j = ref_index, payload_index
        _, action, prev = dp[i][j]

    return list(reversed(output))


def _hirschberg(reference: Sequence[ReferenceTarget],
                payload: Sequence[PayloadTarget],
                match_fn: Callable[[Any, Any], bool],
                progress,
                small_cutoff: int = 2048) -> List[Any]:
    """Space-efficient alignment using Hirschberg's divide-and-conquer."""

    if not reference:
        return [Extra(p.key, ExtraType.PAYLOAD, p.payload) for p in payload]
    if not payload:
        return [Extra(r.key, ExtraType.REFERENCE, r.payload) for r in reference]

    if len(reference) * len(payload) <= small_cutoff:
        return _align_small(reference, payload, match_fn, progress)

    mid = len(reference) // 2
    left_ref = reference[:mid]
    right_ref = reference[mid:]

    score_left = _row_costs(left_ref, payload, match_fn, progress)
    score_right = _row_costs(list(reversed(right_ref)),
                             list(reversed(payload)),
                             match_fn,
                             progress)

    split = 0
    best = None
    payload_len = len(payload)
    for k in range(payload_len + 1):
        cost = score_left[k] + score_right[payload_len - k]
        if best is None or cost < best:
            best = cost
            split = k

    left_alignment = _hirschberg(left_ref, payload[:split], match_fn, progress, small_cutoff)
    right_alignment = _hirschberg(right_ref, payload[split:], match_fn, progress, small_cutoff)

    return left_alignment + right_alignment


def align(source_payload_sequence,
          target_reference_sequence,
          tqdm=True,
          match_fn=lambda x,y: x==y):
    """Align two sequences with a Hirschberg-based edit-distance aligner."""

    use_tqdm = bool(tqdm)
    progress = None

    payload_seq: Sequence[PayloadTarget]
    reference_seq: Sequence[ReferenceTarget]

    if len(source_payload_sequence) > 0 and isinstance(source_payload_sequence[0], PayloadTarget):
        payload_seq = source_payload_sequence
        reference_seq = target_reference_sequence
    else:
        payload_seq, reference_seq = __serialize_arr(source_payload_sequence,
                                                    target_reference_sequence)

    if use_tqdm and len(payload_seq) and len(reference_seq):
        try:
            from tqdm import tqdm as _tqdm
            progress = _tqdm(unit="cell")
        except Exception:
            progress = None

    try:
        return _hirschberg(reference_seq, payload_seq, match_fn, progress)
    finally:
        if progress:
            progress.close()

# align([1,2,3,4,4,5,5,5], [1,1,3,4,4,12,5,5,18])


