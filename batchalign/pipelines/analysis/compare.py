"""
compare.py
Engines for transcript comparison against gold-standard references.

CompareEngine (PROCESSING): Aligns main vs gold transcripts word-by-word
using the same conform/match_fn logic as WER evaluation, then annotates
each gold utterance with comparison tokens (%xsrep / %xsmor).

CompareAnalysisEngine (ANALYSIS): Reads the comparison annotations and
computes error-rate metrics for CSV output.
"""

import re
import logging
from collections import Counter
from batchalign.document import *
from batchalign.pipelines.base import *
from batchalign.utils.dp import align, ExtraType, Extra, Match
from batchalign.utils.names import names
from batchalign.utils.compounds import compounds
from batchalign.utils.abbrev import abbrev

L = logging.getLogger("batchalign")

# --- Duplicated from eval.py to avoid heavy import chain (asr.utils -> num2words) ---

joined_compounds = ["".join(k) for k in compounds]
lowered_abbrev = [k for k in abbrev]

fillers = ["um", "uhm", "em", "mhm", "uhhm", "eh", "uh", "hm"]

def conform(x):
    result = []
    for i in x:
        if i.strip().lower() in joined_compounds:
            for k in compounds[joined_compounds.index(i.strip().lower())]:
                result.append(k)
        elif i.strip() in lowered_abbrev:
            for j in i.strip():
                result.append(j.strip())
        elif "'s" in i.strip().lower():
            result.append(i.split("\u2019")[0] if "\u2019" in i else i.split("'")[0])
            result.append("is")
        elif "\u2019ve" in i.strip().lower() or "'ve" in i.strip().lower():
            result.append(i.split("\u2019")[0] if "\u2019" in i else i.split("'")[0])
            result.append("have")
        elif "\u2019d" in i.strip().lower() or "'d" in i.strip().lower():
            result.append(i.split("\u2019")[0] if "\u2019" in i else i.split("'")[0])
            result.append("had")
        elif "\u2019m" in i.strip().lower() or "'m" in i.strip().lower():
            result.append(i.split("\u2019")[0] if "\u2019" in i else i.split("'")[0])
            result.append("am")
        elif i.strip().lower() in fillers:
            result.append("um")
        elif "-" in i.strip().lower():
            result += [k.strip() for k in i.lower().split("-")]
        elif "ok" == i.strip().lower():
            result.append("okay")
        elif "gimme" == i.strip().lower():
            result.append("give")
            result.append("me")
        elif "hafta" == i.strip().lower() or "havta" == i.strip().lower():
            result.append("have")
            result.append("to")
        elif i.strip().lower() in names:
            result.append("name")
        elif "dunno" == i.strip().lower():
            result.append("don't")
            result.append("know")
        elif "wanna" == i.strip().lower():
            result.append("want")
            result.append("to")
        elif "gonna" == i.strip().lower():
            result.append("going")
            result.append("to")
        elif "gotta" == i.strip().lower():
            result.append("got")
            result.append("to")
        elif "kinda" == i.strip().lower():
            result.append("kind")
            result.append("of")
        elif "sorta" == i.strip().lower():
            result.append("sort")
            result.append("of")
        elif "alright" == i.strip().lower() or "alrightie" == i.strip().lower():
            result.append("all")
            result.append("right")
        elif "shoulda" == i.strip().lower():
            result.append("should")
            result.append("have")
        elif "sposta" == i.strip().lower():
            result.append("supposed")
            result.append("to")
        elif "hadta" == i.strip().lower():
            result.append("had")
            result.append("to")
        elif "til" == i.strip().lower():
            result.append("until")
        elif "ed" == i.strip().lower():
            result.append("education")
        elif "mm" == i.strip().lower() or "hmm" == i.strip().lower():
            result.append("hm")
        elif "eh" == i.strip().lower():
            result.append("uh")
        elif "em" == i.strip().lower():
            result.append("them")
        elif "farmhouse" == i.strip().lower():
            result.append("farm")
            result.append("house")
        elif "this'll" == i.strip().lower():
            result.append("this")
            result.append("will")
        elif "i'd" == i.strip().lower():
            result.append("i")
            result.append("had")
        elif "mba" == i.strip().lower():
            result.append("m")
            result.append("b")
            result.append("a")
        elif "tli" == i.strip().lower():
            result.append("t")
            result.append("l")
            result.append("i")
        elif "bbc" == i.strip().lower():
            result.append("b")
            result.append("b")
            result.append("c")
        elif "ai" == i.strip().lower():
            result.append("a")
            result.append("i")
        elif "ii" == i.strip().lower():
            result.append("i")
            result.append("i")
        elif "aa" == i.strip().lower():
            result.append("a")
            result.append("a")
        elif "_" in i.strip().lower():
            for j in i.strip().split("_"):
                result.append(j)
        else:
            result.append(i.lower())

    return result

def match_fn(x, y):
    x = x.lower()
    y = y.lower()
    return (y == x or
            y.replace("(", "").replace(")", "") == x.replace("(", "").replace(")", "") or
            re.sub(r"\((.*)\)", r"", y) == x or re.sub(r"\((.*)\)", r"", x) == y)

# --- End of eval.py duplicates ---


def _find_best_segment(gold_tokens, main_tokens, mfn):
    """Find a rough window using bag-of-words overlap.

    The rough pass is order-invariant: it scores contiguous windows by token
    multiset overlap with the gold utterance, ignoring order. To keep common
    words from swallowing later transcript material, it only considers windows
    near the gold utterance length. Among equally good windows it prefers the
    one with the fewest non-matching (wasted) tokens, then the most
    Levenshtein alignment matches, then the latest one as a final tiebreaker
    (so CHI repetitions / self-corrections beat earlier INV tokens). The caller then
    runs the full Levenshtein aligner inside that window to produce token
    annotations.
    """
    if not gold_tokens or not main_tokens:
        return 0, 0

    gold_counts = Counter(gold_tokens)
    gold_len = len(gold_tokens)
    main_len = len(main_tokens)

    min_window = max(1, gold_len - 2)
    max_window = min(main_len, gold_len + 2)

    best = (0, min(main_len, gold_len))
    best_score = -1.0
    best_waste = None
    best_align_matches = -1

    for span in range(min_window, max_window + 1):
        window_counts = Counter(main_tokens[:span])
        overlap = sum(min(window_counts[token], gold_counts[token]) for token in window_counts)

        for start in range(0, main_len - span + 1):
            if start > 0:
                left = main_tokens[start - 1]
                right = main_tokens[start + span - 1]

                overlap -= min(window_counts[left], gold_counts[left])
                window_counts[left] -= 1
                overlap += min(window_counts[left], gold_counts[left])

                overlap -= min(window_counts[right], gold_counts[right])
                window_counts[right] += 1
                overlap += min(window_counts[right], gold_counts[right])

            score = overlap / gold_len
            waste = span - overlap  # non-matching tokens in the window
            end = start + span

            # Alignment-based match count (Levenshtein) as tiebreaker
            window = main_tokens[start:end]
            alignment = align(window, gold_tokens, False, mfn)
            align_matches = sum(1 for item in alignment if isinstance(item, Match))

            if score > best_score:
                best = (start, end)
                best_score = score
                best_waste = waste
                best_align_matches = align_matches
            elif score == best_score:
                if best_waste is None or waste < best_waste:
                    best = (start, end)
                    best_waste = waste
                    best_align_matches = align_matches
                elif waste == best_waste:
                    if align_matches > best_align_matches:
                        best = (start, end)
                        best_align_matches = align_matches
                    elif align_matches == best_align_matches and end > best[1]:
                        best = (start, end)

    # If no tokens overlap at all, return an empty window so the caller
    # doesn't consume main tokens that belong to a later gold utterance.
    if best_score <= 0:
        return 0, 0

    return best


def _best_rotation(window_tokens, gold_tokens, mfn):
    """Find the cyclic rotation of *window_tokens* that maximises matches.

    Returns the rotation offset *r* such that
    ``window_tokens[r:] + window_tokens[:r]`` best aligns to *gold_tokens*.
    """
    if len(window_tokens) <= 1:
        return 0

    best_r = 0
    best_matches = -1
    n = len(window_tokens)

    for r in range(n):
        rotated = window_tokens[r:] + window_tokens[:r]
        alignment = align(rotated, gold_tokens, False, mfn)
        matches = sum(1 for item in alignment if isinstance(item, Match))
        if matches > best_matches:
            best_matches = matches
            best_r = r

    return best_r


def _get_pos(form):
    """Extract uppercased POS from a Form's morphology, or '?' if absent."""
    if form is not None and form.morphology:
        return form.morphology[0].pos.upper()
    return "?"


def conform_with_mapping(words, conform_fn):
    """Apply conform() per word, returning expanded tokens and an index mapping.

    Parameters
    ----------
    words : list[str]
        Original word list.
    conform_fn : callable
        The conform function.

    Returns
    -------
    conformed : list[str]
        The conformed (expanded) token list.
    mapping : list[int]
        mapping[j] = index into the original `words` list that conformed[j]
        originated from.
    """
    conformed = []
    mapping = []
    for idx, word in enumerate(words):
        expanded = conform_fn([word])
        for token in expanded:
            conformed.append(token)
            mapping.append(idx)
    return conformed, mapping


class CompareEngine(BatchalignEngine):
    tasks = [Task.COMPARE]

    def __init__(self):
        self._stanza = None

    def _get_stanza(self):
        if self._stanza is None:
            from batchalign.pipelines.morphosyntax import StanzaEngine
            self._stanza = StanzaEngine()
        return self._stanza

    def process(self, doc, **kwargs):
        gold = kwargs.get("gold")
        if not gold or not isinstance(gold, Document):
            raise ValueError(
                f"CompareEngine requires a 'gold' Document kwarg, got '{type(gold)}'"
            )

        # --- 0. Run morphosyntax on both docs ---
        stanza = self._get_stanza()
        non_gold_kwargs = {k: v for k, v in kwargs.items() if k != "gold"}
        doc = stanza.process(doc, **non_gold_kwargs)
        gold = stanza.process(gold, **non_gold_kwargs)

        # --- 1. Extract words from main utterances ---
        main_utterances = [
            u for u in doc.content if isinstance(u, Utterance)
        ]
        main_info = []  # (utt_idx, form_idx, Form)
        main_words = []

        for utt_idx, utt in enumerate(main_utterances):
            for form_idx, form in enumerate(utt.content):
                if form.text.strip() in MOR_PUNCT + ENDING_PUNCT:
                    continue
                if _get_pos(form) == "PUNCT":
                    continue
                if form.text.strip().lower() in fillers:
                    continue
                main_info.append((utt_idx, form_idx, form))
                main_words.append(form.text)

        # --- 2. Extract words from gold utterances ---
        gold_utterances = [
            u for u in gold.content if isinstance(u, Utterance)
        ]
        gold_info = []  # (utt_idx, form_idx, Form)
        gold_words = []
        gold_punct = {}  # utt_idx -> list of (form_idx, Form)

        for utt_idx, utt in enumerate(gold_utterances):
            gold_punct[utt_idx] = []
            for form_idx, form in enumerate(utt.content):
                if form.text.strip() in MOR_PUNCT + ENDING_PUNCT or _get_pos(form) == "PUNCT":
                    gold_punct[utt_idx].append((form_idx, form))
                    continue
                if form.text.strip().lower() in fillers:
                    continue
                gold_info.append((utt_idx, form_idx, form))
                gold_words.append(form.text)

        # --- 3. Apply conform() with mapping ---
        conformed_main, main_map = conform_with_mapping(main_words, conform)
        conformed_gold, gold_map = conform_with_mapping(gold_words, conform)

        # --- 4. Partition conformed gold tokens by utterance ---
        gold_utt_tokens = {i: [] for i in range(len(gold_utterances))}
        gold_utt_maps = {i: [] for i in range(len(gold_utterances))}
        for j in range(len(conformed_gold)):
            orig_idx = gold_map[j]
            utt_idx = gold_info[orig_idx][0]
            gold_utt_tokens[utt_idx].append(conformed_gold[j])
            gold_utt_maps[utt_idx].append(orig_idx)

        # --- 5. Per-utterance alignment ---
        # For each gold utterance, find a rough last-possible bag-of-words
        # window in the remaining main tokens, then run Levenshtein inside
        # that window to produce the annotations.
        utt_positioned = {i: [] for i in range(len(gold_utterances))}
        # Track which main (child) forms land in each gold utterance's window
        utt_main_forms = {i: [] for i in range(len(gold_utterances))}
        # Track which main utterance(s) contribute to each gold utterance
        utt_main_speakers = {i: [] for i in range(len(gold_utterances))}
        search_start = 0

        for utt_idx in range(len(gold_utterances)):
            g_tokens = gold_utt_tokens[utt_idx]
            g_maps = gold_utt_maps[utt_idx]
            G = len(g_tokens)

            if G == 0:
                continue

            remaining_main = conformed_main[search_start:]
            win_start, win_end = _find_best_segment(g_tokens, remaining_main, match_fn)

            abs_start = search_start + win_start
            abs_end = search_start + win_end

            # Collect unique main forms in the window (deduplicate
            # across conformed expansions that map to the same form)
            seen_main = set()
            for j in range(abs_start, abs_end):
                orig_idx = main_map[j]
                if orig_idx not in seen_main:
                    seen_main.add(orig_idx)
                    m_utt_idx, m_form_idx, m_form = main_info[orig_idx]
                    utt_main_forms[utt_idx].append(m_form)
                    utt_main_speakers[utt_idx].append(m_utt_idx)

            # Align the chosen window against this gold utterance,
            # trying cyclic rotations to avoid spurious del/ins pairs.
            window_main = conformed_main[abs_start:abs_end]
            window_len = len(window_main)
            rotation = _best_rotation(window_main, g_tokens, match_fn)
            if rotation > 0:
                window_main = window_main[rotation:] + window_main[:rotation]
            utt_alignment = align(window_main, g_tokens, False, match_fn)

            local_main_cursor = 0
            local_gold_cursor = 0
            last_gold_form_idx = -1

            for item in utt_alignment:
                if isinstance(item, Match):
                    global_main_idx = abs_start + (local_main_cursor + rotation) % window_len
                    orig_main_idx = main_map[global_main_idx]
                    main_form = main_info[orig_main_idx][2]
                    orig_gold_idx = g_maps[local_gold_cursor]
                    gold_form_idx = gold_info[orig_gold_idx][1]
                    gold_form = gold_info[orig_gold_idx][2]
                    last_gold_form_idx = gold_form_idx

                    utt_positioned[utt_idx].append((gold_form_idx, CompareToken(
                        text=item.key,
                        pos=_get_pos(gold_form),
                        status="match"
                    )))
                    local_main_cursor += 1
                    local_gold_cursor += 1

                elif isinstance(item, Extra):
                    if item.extra_type == ExtraType.REFERENCE:
                        orig_gold_idx = g_maps[local_gold_cursor]
                        gold_form_idx = gold_info[orig_gold_idx][1]
                        gold_form = gold_info[orig_gold_idx][2]
                        last_gold_form_idx = gold_form_idx

                        utt_positioned[utt_idx].append((gold_form_idx, CompareToken(
                            text=item.key,
                            pos=_get_pos(gold_form),
                            status="extra_gold"
                        )))
                        local_gold_cursor += 1

                    else:
                        global_main_idx = abs_start + (local_main_cursor + rotation) % window_len
                        orig_main_idx = main_map[global_main_idx]
                        main_form = main_info[orig_main_idx][2]

                        pos = last_gold_form_idx + 0.5
                        utt_positioned[utt_idx].append((pos, CompareToken(
                            text=item.key,
                            pos=_get_pos(main_form),
                            status="extra_main"
                        )))
                        local_main_cursor += 1

            search_start = abs_end

        # --- 6. Merge punctuation from gold at original positions ---
        for utt_idx in range(len(gold_utterances)):
            for form_idx, form in gold_punct[utt_idx]:
                utt_positioned[utt_idx].append((form_idx, CompareToken(
                    text=form.text,
                    pos="PUNCT",
                    status="match"
                )))
            utt_positioned[utt_idx].sort(key=lambda x: x[0])

        # --- 7. Replace each gold utterance's main line with the child's
        #         (main/proposal) forms and set the child's speaker. ---
        for utt_idx, utt in enumerate(gold_utterances):
            tokens = [tok for _, tok in utt_positioned[utt_idx]]
            utt.comparison = tokens if tokens else None

            # Replace content with the child/main forms from the window
            child_forms = utt_main_forms[utt_idx]
            if child_forms:
                # Grab the ending punctuation from gold so the utterance
                # delimiter is preserved
                gold_ending = [
                    f for f in utt.content
                    if f.text.strip() in ENDING_PUNCT
                ]
                # Strip dependency from child forms (morphosyntax was run
                # on the main doc separately; gold structure differs)
                for f in child_forms:
                    f.dependency = None
                utt.content = child_forms + gold_ending

                # Set speaker from the child utterance that contributed
                # the most forms to this window
                speaker_indices = utt_main_speakers[utt_idx]
                if speaker_indices:
                    # Pick the most common main utterance index
                    majority_utt_idx = max(
                        set(speaker_indices), key=speaker_indices.count
                    )
                    utt.tier = main_utterances[majority_utt_idx].tier

            # Derive utterance timing from the (now-child) forms
            timed_forms = [f for f in utt.content if f.time is not None]
            if timed_forms:
                utt.time = (timed_forms[0].time[0], timed_forms[-1].time[1])
                utt.text = None

        # Copy @Media header from the main doc if it has one
        if doc.media is not None:
            gold.media = doc.media

        return gold


class CompareAnalysisEngine(BatchalignEngine):
    tasks = [Task.COMPARE_ANALYSIS]

    def analyze(self, doc, **kwargs):
        from collections import defaultdict

        matches = 0
        extra_main = 0
        extra_gold = 0

        # Per-POS counters: pos -> {matches, insertions, deletions}
        pos_counts = defaultdict(lambda: {"matches": 0, "insertions": 0, "deletions": 0})

        for utt in doc.content:
            if not isinstance(utt, Utterance) or utt.comparison is None:
                continue
            for tok in utt.comparison:
                if tok.pos == "PUNCT":
                    continue
                if tok.status == "match":
                    matches += 1
                    pos_counts[tok.pos]["matches"] += 1
                elif tok.status == "extra_main":
                    extra_main += 1
                    pos_counts[tok.pos]["insertions"] += 1
                elif tok.status == "extra_gold":
                    extra_gold += 1
                    pos_counts[tok.pos]["deletions"] += 1

        total_gold = matches + extra_gold
        total_main = matches + extra_main
        wer = (extra_main + extra_gold) / total_gold if total_gold > 0 else 0.0
        accuracy = 1.0 - wer

        metrics = {
            "wer": round(wer, 4),
            "accuracy": round(accuracy, 4),
            "matches": matches,
            "insertions": extra_main,
            "deletions": extra_gold,
            "total_gold_words": total_gold,
            "total_main_words": total_main,
        }

        # Add per-POS breakdown
        for pos in sorted(pos_counts.keys()):
            counts = pos_counts[pos]
            total = counts["matches"] + counts["deletions"]
            metrics[f"{pos}:matches"] = counts["matches"]
            metrics[f"{pos}:insertions"] = counts["insertions"]
            metrics[f"{pos}:deletions"] = counts["deletions"]
            metrics[f"{pos}:total"] = total

        return {
            "doc": doc,
            "metrics": metrics,
        }
