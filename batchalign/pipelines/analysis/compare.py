"""
compare.py
Engines for transcript comparison against gold-standard references.

CompareEngine (PROCESSING): Aligns main vs gold transcripts word-by-word
using the same conform/match_fn logic as WER evaluation, then annotates
each main utterance with comparison tokens (%xsrep / %xsmor).

CompareAnalysisEngine (ANALYSIS): Reads the comparison annotations and
computes error-rate metrics for CSV output.
"""

import re
import logging
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

    def process(self, doc, **kwargs):
        gold = kwargs.get("gold")
        if not gold or not isinstance(gold, Document):
            raise ValueError(
                f"CompareEngine requires a 'gold' Document kwarg, got '{type(gold)}'"
            )

        # --- 1. Extract words from main utterances ---
        main_utterances = [
            u for u in doc.content if isinstance(u, Utterance)
        ]
        main_info = []  # (utt_idx, form_idx, Form)
        main_words = []
        main_punct = {}  # utt_idx -> list of (form_idx, Form)

        for utt_idx, utt in enumerate(main_utterances):
            main_punct[utt_idx] = []
            for form_idx, form in enumerate(utt.content):
                if form.text.strip() in MOR_PUNCT + ENDING_PUNCT:
                    main_punct[utt_idx].append((form_idx, form))
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

        for utt_idx, utt in enumerate(gold_utterances):
            for form_idx, form in enumerate(utt.content):
                if form.text.strip() in MOR_PUNCT + ENDING_PUNCT:
                    continue
                if form.text.strip().lower() in fillers:
                    continue
                gold_info.append((utt_idx, form_idx, form))
                gold_words.append(form.text)

        # --- 3. Apply conform() with mapping ---
        conformed_main, main_map = conform_with_mapping(main_words, conform)
        conformed_gold, gold_map = conform_with_mapping(gold_words, conform)

        # --- 4. Align ---
        alignment = align(conformed_main, conformed_gold, False, match_fn)

        # --- 5. Redistribute alignment results per main utterance ---
        # Store (position, CompareToken) pairs so we can interleave punct
        utt_positioned = {i: [] for i in range(len(main_utterances))}
        current_main_utt = 0
        last_main_form_idx = -1
        main_cursor = 0
        gold_cursor = 0

        for item in alignment:
            if isinstance(item, Match):
                orig_main_idx = main_map[main_cursor]
                main_utt_idx = main_info[orig_main_idx][0]
                main_form_idx = main_info[orig_main_idx][1]
                main_form = main_info[orig_main_idx][2]
                current_main_utt = main_utt_idx
                last_main_form_idx = main_form_idx

                utt_positioned[main_utt_idx].append((main_form_idx, CompareToken(
                    text=item.key,
                    pos=_get_pos(main_form),
                    status="match"
                )))
                main_cursor += 1
                gold_cursor += 1

            elif isinstance(item, Extra):
                if item.extra_type == ExtraType.PAYLOAD:
                    # Word in main but not in gold -> extra_main (+)
                    orig_main_idx = main_map[main_cursor]
                    main_utt_idx = main_info[orig_main_idx][0]
                    main_form_idx = main_info[orig_main_idx][1]
                    main_form = main_info[orig_main_idx][2]
                    current_main_utt = main_utt_idx
                    last_main_form_idx = main_form_idx

                    utt_positioned[main_utt_idx].append((main_form_idx, CompareToken(
                        text=item.key,
                        pos=_get_pos(main_form),
                        status="extra_main"
                    )))
                    main_cursor += 1

                else:
                    # Word in gold but not in main -> extra_gold (-)
                    orig_gold_idx = gold_map[gold_cursor]
                    gold_form = gold_info[orig_gold_idx][2]

                    # Position just after last main form for correct ordering
                    pos = last_main_form_idx + 0.5
                    utt_positioned[current_main_utt].append((pos, CompareToken(
                        text=item.key,
                        pos=_get_pos(gold_form),
                        status="extra_gold"
                    )))
                    gold_cursor += 1

        # --- 6. Merge punctuation at original positions ---
        for utt_idx in range(len(main_utterances)):
            for form_idx, form in main_punct[utt_idx]:
                utt_positioned[utt_idx].append((form_idx, CompareToken(
                    text=form.text,
                    pos="PUNCT",
                    status="match"
                )))
            # Stable sort by position preserves order within same form_idx
            utt_positioned[utt_idx].sort(key=lambda x: x[0])

        # --- 7. Set comparison on each utterance ---
        for utt_idx, utt in enumerate(main_utterances):
            tokens = [tok for _, tok in utt_positioned[utt_idx]]
            utt.comparison = tokens if tokens else None

        return doc


class CompareAnalysisEngine(BatchalignEngine):
    tasks = [Task.COMPARE_ANALYSIS]

    def analyze(self, doc, **kwargs):
        matches = 0
        extra_main = 0
        extra_gold = 0

        for utt in doc.content:
            if not isinstance(utt, Utterance) or utt.comparison is None:
                continue
            for tok in utt.comparison:
                if tok.status == "match":
                    matches += 1
                elif tok.status == "extra_main":
                    extra_main += 1
                elif tok.status == "extra_gold":
                    extra_gold += 1

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

        return {
            "doc": doc,
            "metrics": metrics,
        }
