"""
dp.py
Dynamic Programing Utilities

Generally used for minimum-edit seq alignment across the program
"""

from dataclasses import dataclass
from typing import Optional, Any
from tqdm import tqdm
from enum import Enum

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

def __serialize_arr(src, tgt):
    """utility function to serialize to array with no payloads

    Parameters
    ----------
    src: List[any]
        The source sequence to align, "payload"
    tgt: List[any]
        The target sequence to align, "reference"

    Returns
    -------
    Tuple[List[PayloadTarget], List[ReferenceTarget]]
         The serialized output
    """

    src_serialized = [PayloadTarget(i, None) for i in src]
    tgt_serialized = [ReferenceTarget(i) for i in tgt]

    return src_serialized, tgt_serialized

def __dp(payload, reference, t):
    """Performs bottom-up dynamic programming alignment

    Parameters
    ----------
    payload: List[PayloadTarget]
        The payload-type source sequence to align.
    reference: List[ReferenceTarget]
        The reference-type target sequence to align.
    t : Bool
        Whether to use TQDM.

    Returns
    -------
    Tuple[Int, List[Extra | Match]]
        The Levistein edit distance, as well as the actual
        edited sequence expressed by "Extra" vs. "Match"
    """

    from tqdm import tqdm
    tqdm = tqdm if t else (lambda total="":None)
    
    # bottom-up DP table:
    # every ROW represents a new element in REFERENCE
    # every COLUMN represents a new element in PAYLOAD
    # indexing is ROW MAJOR; so dp[5][1] represents 
    # trying to align the fist 5 elements of REFERENCE
    # against the first element of PAYLOAD
    #
    # **AND YES. THIS MEANS THAT WE ARE 1 INDEXING!**
    # sentry nodes are excellent.
    #
    # each element contains a tuple of 2 elements:
    # (dist, List[]), where the first element is the
    # edit distance, and the second element is the
    # stiching so far
    dp = [[(None, None, None) for _ in range(len(payload)+1)]
        for _ in range(len(reference)+1)]

    # the centry node has edit distance 0 (aligning
    # two empty sequences)
    dp[0][0] = (0, None, None)
    # and seed the edges: which are literally just straight appending
    # this is also solable by better, more sinsible defaults of seeding
    # the table but eh
    for i in range(1, len(reference)+1):
        prev_dist, _, _ = dp[i-1][0]
        dp[i][0] = (prev_dist+1, OutputType.EXTRA_REFERENCE, (i-1, 0))
    for j in range(1, len(payload)+1):
        prev_dist, _, _ = dp[0][j-1]
        dp[0][j] = (prev_dist+1, OutputType.EXTRA_PAYLOAD, (0, j-1))

    # now, its dp time.
    bar = tqdm(total=(len(reference)+1)*(len(payload)+1))
    for i in range(1, len(reference)+1):
        for j in range(1, len(payload)+1):
            # get the three possible base solutions
            dist1, _, _ = dp[i-1][j-1]
            dist2, _, _ = dp[i-1][j]
            dist3, _, _ = dp[i][j-1]
            # Get the most optimal solution to solve for.
            # Solutions 2 and 3 are both just extra in one direction
            # so dist+1.
            # As we just need to solve the subproblem of
            # matching a 0-length sequence against a 1-length
            # sequence (i.e. elements j or i respectively).
            # 
            # Solution 1 is either dist+2 (levistein style) if
            # i != j, as the subproblem of aligning i,j element
            # would be a substitution. If they equal then we
            # get a match.

            # recall 1 indexing
            is_match = (reference[i-1].key == payload[j-1].key)

            # calculate new distances
            new_dist1 = dist1+(0 if is_match else 2)
            new_dist2 = dist2+1 # Extra in one direction
            new_dist3 = dist3+1 # Extra in one direction

            # if solution 1 is the best:
            # by levistein style, solution 1 either has an added
            # weight of 0 (match!) or 2 where both are extras.
            if new_dist1 <= new_dist2 and new_dist1 <= new_dist3:
                if is_match:
                    dp[i][j] = (new_dist1, OutputType.MATCH, (i-1, j-1))
                else:
                    dp[i][j] = (new_dist1, OutputType.EXTRA_BOTH, (i-1, j-1))
            elif new_dist2 <= new_dist1 and new_dist2 <= new_dist3:
                dp[i][j] = (new_dist2, OutputType.EXTRA_REFERENCE, (i-1, j))
            elif new_dist3 <= new_dist1 and new_dist3 <= new_dist2:
                dp[i][j] = (new_dist3, OutputType.EXTRA_PAYLOAD, (i, j-1))

            if bar:
                bar.update(1)

    # get the final solution by backtracking through the solutions
    dist, action, prev = dp[len(reference)][len(payload)]

    output = []

    # as long as there is a node to backtrace to
    while prev:
        ref_index, payload_index = prev
        # apply action
        # note that we index reference and payload by the PREVIOUS
        # indicies; this is a trick: because we are 1-indexing, the access
        # value for elements in the list would be 1- the index value.
        #
        # therefore, previous value, indexed correclty, is actually exactly
        # the index needed
        if action == OutputType.MATCH:
            output.append(Match(reference[ref_index].key,
                                payload[payload_index].payload,
                                reference[ref_index].payload))
        if action == OutputType.EXTRA_BOTH or action == OutputType.EXTRA_PAYLOAD:
            output.append(Extra(payload[payload_index].key,
                                ExtraType.PAYLOAD,
                                payload[payload_index].payload))
        if action == OutputType.EXTRA_BOTH or action == OutputType.EXTRA_REFERENCE:
            output.append(Extra(reference[ref_index].key,
                                ExtraType.REFERENCE,
                                reference[ref_index].payload))

        _, action, prev = dp[ref_index][payload_index]
        

    # given we backtraced, we need to reverse this list to output the right direction
    return list(reversed(output))


def align(source_payload_sequence,
          target_reference_sequence,
          tqdm=True):
    """Align two sequences"""

    if (len(source_payload_sequence) > 0 and
        type(source_payload_sequence[0]) == PayloadTarget):
        return __dp(source_payload_sequence, target_reference_sequence, tqdm)
    else:
        return __dp(*__serialize_arr(source_payload_sequence,
                                     target_reference_sequence), tqdm)

# align([1,2,3,4,4,5,5,5], [1,1,3,4,4,12,5,5,18])


