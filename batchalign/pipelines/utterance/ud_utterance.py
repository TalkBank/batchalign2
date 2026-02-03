# system utils
import glob, os, re
import pathlib
from itertools import groupby

# pathing tools
from pathlib import Path

import copy

# UD tools imports removed from top-level to speed up runs

# the loading bar
from tqdm import tqdm

from bdb import BdbQuit

from nltk import word_tokenize
from collections import defaultdict

import warnings

# Oneliner of directory-based glob and replace
globase = lambda path, statement: glob.glob(os.path.join(path, statement))
repath_file = lambda file_path, new_dir: os.path.join(new_dir, pathlib.Path(file_path).name)


from batchalign.document import (
    Document, Utterance, Form, TokenType, Task, ENDING_PUNCT
)
from batchalign.pipelines.base import BatchalignEngine
from batchalign.formats.chat.parser import chat_parse_utterance

from batchalign.utils.dp import (
    PayloadTarget, ReferenceTarget, Match, Extra, ExtraType, align
)

import logging
L = logging.getLogger("batchalign")

import pycountry

def rollout_to_leaf(tree):
    """Extract the leaf nodes from a subtree via dfs"""

    try:
        children = tree.children
    except AttributeError:
        return []
    leafs = []

    for c in children:
        if c.is_leaf():
            leafs.append(c.label)
        else:
            leafs += rollout_to_leaf(c)

    return leafs


def parse_tree(subtree):
    stack = []

    subtree_labels = [i.label.lower() if i.label else "" for i in subtree.children]
    possible_labels = ["cc", "conj"]
    # if we have a coordinating conjuction at this level
    # we will consider all full sentence phrases in this
    # lavel for parsing
    if len(list(filter(lambda x:x in possible_labels, subtree_labels))) > 0:
        stack += [i for i in subtree.children if i.label == "S"]
    # also, parse all other subtrees
    stack += [j for i in subtree.children for j in parse_tree(i)]

    # return stringified represnetation
    return [" ".join(rollout_to_leaf(i)).strip()
                     if type(i) != str else i
                     for i in stack]

def process_ut(ut, nlp):
    import stanza

    # remove punct
    if (ut.content[-1].type == TokenType.PUNCT or
        ut.content[-1].text in ENDING_PUNCT):
        ut.content = ut.content[:-1]
        
    raw = ut.strip()
    
    parse = nlp(raw).sentences

    # rollout_to_leaf(
    # parse the text!
    pt = parse[0].constituency
    # parse_tree
    # get the rollouts
    possible_forms = parse_tree(pt)
    possible_forms = (sorted(possible_forms, key=lambda x:len(x)))

    # get unique short forms
    unique_short_forms: list[str] = []
    for i in list(reversed(possible_forms))+[" ".join(rollout_to_leaf(pt))]:
        for j in [x for x in unique_short_forms if x in i]:
            i = i.replace(j, "")
        if i.strip() != "" and len([x for x in unique_short_forms if i in x]) == 0:
            unique_short_forms.append(i)
    unique_short_forms_reversed = list(reversed(unique_short_forms))
    # retokenize (notice we combined forms with " ", so even if the language doesn't delinate
    # by space this should work fine
    unique_split_forms = [[j for j in i.split(" ") if j != ""] for i in unique_short_forms_reversed]
    # drop all single word forms (we will reattach them later---they are usually CCs or SCs)
    unique_split_forms = [i for i in unique_split_forms if len(i) != 1]
    # reattach back to our original forms
    # first, assemble refrencees whose payload will be index into the utterance
    refs = [ReferenceTarget(key=i.text, payload=indx) for indx, i in enumerate(ut.content) if isinstance(i, Form)]
    # our alignments will be the Phrase ID of each unique short form
    # the number doesn't matter, it simply matters how different they are
    payloads = [PayloadTarget(key=j, payload=indx)
                for indx, i in enumerate(unique_split_forms)
                for j in i]

    # import random
    # tmp1 = payloads[:]
    # tmp2 = refs[:]
    # random.shuffle(tmp1)
    # random.shuffle(tmp2)
    # 
    # and now, a good time: we have to align our targets a group at a time because they maybe
    # out of order, meaning weird edit distances
    matches = []
    # breakpoint()

    alignment = align(payloads, refs, False)
    new_refs = []
    # we want to collect the Matches, and resealize any
    # reference extras (i.e. those we haven't aligned yet)
    for i in alignment:
        if isinstance(i, Match):
            matches.append(i)
        elif i.extra_type == ExtraType.REFERENCE:
            new_refs.append(ReferenceTarget(key=i.key, payload=i.payload if i.payload else -1))

    # we now sort the references based on their orignial utterance order
    matches_combined: list[Match | ReferenceTarget] = matches + new_refs
    matches_combined = sorted(matches_combined, key=lambda x:x.reference_payload if isinstance(x, Match) else x.payload)


    # for each group, we combine into utterances based on the following heuristics
    utterances = []
    current_ut = []
    current_group = -1 # this is the "utterance group" information 

    for indx, i in enumerate(matches_combined):
        # this is to cache cases where reference taget is used
        next_payload = -1
        # if something didn't align, we stick it to next
        # group of the next form that did; if there is no
        # next form, we will stick it with the previous form
        if isinstance(i, ReferenceTarget):
            tmp = indx + 1
            while tmp < len(matches_combined) and not isinstance(matches_combined[tmp], Match):
                tmp += 1
                # we found nothing or we found the same group so we just stick to the current one
            if tmp == len(matches_combined) or matches_combined[tmp].payload == current_group:
                current_ut.append(i.payload)
                continue
            else:
                next_payload = matches_combined[tmp].payload

        # in other cases, if our current group is not the previous one
        # (or we are in a new extra and we haven't dealth with that)
        # we will flush this utterance and make a new group
        if isinstance(i, ReferenceTarget) or i.payload != current_group:
            utterances.append(current_ut)
            current_ut = [(i.reference_payload
                            if not isinstance(i, ReferenceTarget)
                            else i.payload)]
            current_group = (i.payload
                            if not isinstance(i, ReferenceTarget)
                            else next_payload)
        else:
            # otherwise, we are in the middle of an utterance
            current_ut.append(i.reference_payload)

    utterances.append(current_ut)
    utterances = utterances[1:]


    # for every single word drop, we combine it with the next utterance
    # as in---for every single word utterance we make, we just stick it onto the
    # next utterance
    utterances_copy = utterances[:]
    utterances = []
    indx = 0
    comb: list[int] = []
    while indx < len(utterances_copy):
        if len(utterances_copy[indx]) < 3:
            comb += utterances_copy[indx]
        else:
            utterances.append(comb + utterances_copy[indx])
            comb = []
        indx += 1

    # create new utterance
    tier = ut.tier
    new_uts = []
    for st in utterances:
        new_ut = []
        for j in st:
            new_ut.append(ut.content[j])
        new_ut = Utterance(content=new_ut, tier=tier)
        # if we are missing an ending, fix that
        if new_ut.content[-1].text not in ENDING_PUNCT:
            new_ut.content.append(Form(text=".", type=TokenType.PUNCT))
        new_uts.append(new_ut)

    return new_uts
 
class StanzaUtteranceEngine(BatchalignEngine):
    tasks = [ Task.UTTERANCE_SEGMENTATION ]

    def __init__(self):
        self.status_hook = None

    def _hook_status(self, status_hook):
        self.status_hook = status_hook

    def _get_stanza_version(self):
        """Get the stanza version without a full import."""
        try:
            import importlib.metadata
            return importlib.metadata.version("stanza")
        except Exception:
            import stanza
            return stanza.__version__

    def process(self, doc, **kwargs):
        import stanza
        from stanza import DownloadMethod
        
        # Import caching components
        from batchalign.pipelines.cache import (
            CacheManager, UtteranceSegmentationCacheKey, _get_batchalign_version
        )

        # Initialize cache infrastructure
        cache = CacheManager()
        key_gen = UtteranceSegmentationCacheKey()
        engine_version = self._get_stanza_version()
        ba_version = _get_batchalign_version()
        override_cache = kwargs.get("override_cache", False)

        # Get primary language for cache key
        primary_lang = doc.langs[0] if doc.langs else "eng"

        L.debug("Starting Stanza...")
        lang_alpha2 = []
        for i in doc.langs:
            if i == "yue":
                lang_alpha2.append("zh-hant")
            else:
                try:
                    lang_alpha2.append(pycountry.languages.get(alpha_3=i).alpha_2)
                except:
                    # some languages don't have alpha 2
                    pass


        # pycountry.languages.get(alpha_3=i).alpha_2 for i in lang_alpha2

        config = {"processors": {"tokenize": "default",
                                    "pos": "default",
                                    # "mwt": "gum" if ("en" in lang_alpha2) else "default",
                                    "lemma": "default",
                                    "constituency": "default"}}


        if "zh" in lang_alpha2:
            lang_alpha2.pop(lang_alpha2.index("zh"))
            lang_alpha2.append("zh-hans")

        elif not any([i in ["hr", "zh", "zh-hans", "zh-hant", "ja", "ko",
                            "sl", "sr", "bg", "ru", "et", "hu",
                            "eu", "el", "he", "af", "ga", "da"] for i in lang_alpha2]):
            if "en" in lang_alpha2:
                config["processors"]["mwt"] = "gum"
            else:
                config["processors"]["mwt"] = "default"

        configs = {}
        for l in lang_alpha2:
            configs[l] = config.copy()

        L.debug("Stanza Loaded.")
        contents = []
        
        # Phase 1: Pre-generate keys and check cache
        idx_to_key = {}
        for idx, item in enumerate(doc.content):
            if not isinstance(item, Utterance) or len(item.content) == 0:
                continue
            
            try:
                key = key_gen.generate_key(item, lang=primary_lang)
                idx_to_key[idx] = key
            except Exception:
                pass

        cached_results = {}
        if not override_cache and idx_to_key:
            cached_results = cache.get_batch(list(idx_to_key.values()), "utterance_segmentation", engine_version)

        # Phase 2: Process utterances (using cache or Stanza)
        nlp_obj = None
        new_cached_entries = []

        for indx, i in enumerate(doc.content):
            if self.status_hook:
                self.status_hook(indx+1, len(doc.content))

            if not isinstance(i, Utterance):
                contents.append(i)
                continue
            
            if len(i.content) == 0:
                continue

            # Check cache
            cache_key: str | None = idx_to_key.get(indx)
            if cache_key and cache_key in cached_results:
                new_uts = key_gen.deserialize_output(cached_results[cache_key], i)
                contents += new_uts
                continue

            # Cache miss - need Stanza
            if nlp_obj is None:
                L.info(f"Stanza utseg cache miss at index {indx}, loading model...")
                if len(lang_alpha2) > 1:
                    nlp_obj = stanza.MultilingualPipeline(
                        lang_configs = configs,
                        lang_id_config = {"langid_lang_subset": lang_alpha2},
                        download_method=DownloadMethod.REUSE_RESOURCES
                    )
                else:
                    nlp_obj = stanza.Pipeline(
                        lang=lang_alpha2[0],
                        **configs[lang_alpha2[0]],
                        download_method=DownloadMethod.REUSE_RESOURCES
                    )

            L.info(f"Stanza utseg processing turn {indx+1}/{len(doc.content)}")
            try:
                new_uts = process_ut(i, nlp_obj)
                # Store for batch caching
                if cache_key:
                    data = key_gen.serialize_output(new_uts)
                    new_cached_entries.append((cache_key, data))
            except IndexError:
                new_uts = [i]
            contents += new_uts

        # Phase 3: Store newly processed results in cache
        if new_cached_entries:
            cache.put_batch(new_cached_entries, "utterance_segmentation", engine_version, ba_version)

        doc.content = contents
        return doc



#         # # TODO sometimes tokenization causes problems at this stage, however, in all the cases
#         # # I can think of sticking the form to the top of the next utterance will do just fine



        # # parse[0].constituency

        # # # this is a really long utterance that i would really like to cut into multiple pieces but i'm not sure if its possible because there is not basic sentence segmentation
        # # # "this is a really long utterance
        # # #  that i would really like to cut into multiple pieces
        # # #  but i'm not sure if its possible
        # # #  because there is not basic sentence segmentation"

        # # rollout_form(nlp("Barry and the boys went shopping, and Ronny and Roys went bopping.").sentences[0].constituency)
        # # for ut in doc:
        # ut = doc[0]

