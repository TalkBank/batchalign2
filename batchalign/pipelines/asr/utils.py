import re
from batchalign.document import *
from batchalign.utils import *

from batchalign.constants import ENDING_PUNCT

import logging
L = logging.getLogger("batchalign")



# Define a regular expression to match various punctuation marks
def retokenize(intermediate_output):
    """
    Retokenize the output of the ASR system to split by sentences whenever
    ending punctuation is detected, and ensure the segmentation is correct.

    Parameters
    ----------
    intermediate_output : List of tuples
        [(speaker, [(word, [start_time, end_time])])]

    Returns
    -------
    final_outputs : List of tuples
        [(speaker, [(word, [start_time, end_time])])]
    """
    
    final_outputs = []  # Store the final output clauses

    for speaker, utterance in intermediate_output:
        tmp = []  # Used to store the list of words for the current clause

        for word, bullet in utterance:
            # Preprocess the word, replacing some special characters
            word = word.replace("。", ".")
            word = word.replace("¿", " ").replace("¡", " ")

            # If the current word contains punctuation, separate and handle it
            split_indices = [index for index, char in enumerate(word) if char in ENDING_PUNCT]
            if split_indices:
                prev_index = 0
                for index in split_indices:
                    # Add the first part of the word (if it exists)
                    if prev_index < index:
                        tmp.append((word[prev_index:index], bullet))

                    # Add the punctuation mark as a separate clause
                    tmp.append((word[index], [None, None]))
                    if tmp:  # Ensure tmp is not empty before adding
                        final_outputs.append((speaker, tmp))
                    tmp = []  # Clear tmp to start processing a new clause

                    prev_index = index + 1

                # Handle any remaining word after the punctuation
                if prev_index < len(word):
                    tmp.append((word[prev_index:], bullet))
            else:
                # If the current word doesn't contain punctuation, add it to tmp
                tmp.append((word, bullet))

            # Check if the current word is an ending punctuation mark, if so, handle it separately
            if word in ENDING_PUNCT or word[-1] in ENDING_PUNCT:
                if word in ENDING_PUNCT:
                    # The entire word is a punctuation mark, add tmp as a complete clause
                    if tmp:  # Ensure tmp is not empty
                        final_outputs.append((speaker, tmp))
                else:
                    # The word ends with a punctuation mark, separate it from the word
                    final_word, time = tmp.pop(-1)
                    tmp.append((final_word[:-1], time))  # Remove the punctuation from the word
                    tmp.append((final_word[-1], [None, None]))  # Add the punctuation as a separate clause
                    final_outputs.append((speaker, tmp))

                tmp = []  # Reset tmp for processing the next clause

        # If there are any unprocessed sentences left at the end (no ending punctuation)
        if tmp:
            # Add a sentence ending mark to avoid passing empty sentences to the next function
            tmp.append((".", [None, None]))
            final_outputs.append((speaker, tmp))

    # Ensure the final results don't contain empty sentences or clause lists
    final_outputs = [entry for entry in final_outputs if entry[1]]

    return final_outputs


def retokenize_with_engine(intermediate_output, engine):
    """Retokenize `intermediate_input` given utterance engine `engine`"""

    final_outputs = []

    for speaker, utterance in intermediate_output:
        # Since we are using an utterance engine, remove all existing punctuation marks
        for i in utterance:
            for j in MOR_PUNCT + ENDING_PUNCT:
                i[0] = i[0].strip(j).lower()

        # Remove all empty elements
        utterance = [i for i in utterance if i[0].strip() != ""]

        # Join all the text into one large string, replacing some punctuation marks
        joined = " ".join([i[0] for i in utterance])
        joined = joined.replace("。", ".")
        
        # L.Debug the joined string before passing it to engine.split()
        # L.debug(f"Debug - Input to engine.split: {joined}")
        
        # Use the engine to split the text
        split = engine(joined)
        
        # L.Debug the split results
        # L.debug(f"Debug - Engine Split Result: {split}")

        # Process the returned split
        for i in split:
            if i[-1] in ENDING_PUNCT:
                new_ut, delim = (i[:-1].split(" "), i[-1])
            else:
                new_ut, delim = (i.split(" "), ".")

            tmp = []

            for s in new_ut:
                tmp.append((s, utterance.pop(0)[1]))

            final_outputs.append((speaker, tmp + [[delim, [None, None]]]))

    # L.debug(f"Debug - Final Outputs: {final_outputs}")
    return final_outputs


def process_generation(output, lang="eng", utterance_engine=None):
    """Process Rev.AI style ASR generation

    Parameters
    ----------
    output : dict
        The raw Rev.AI style output from your ASR engine.
    lang : str
        The language ID.
    utterance_engine : optional 
        Utterance segmentation engine to use.

    Returns
    -------
    Document
        The reassembled document.
    """
    doc = Document()
    utterance_col = []

    # L.debug("Debug - Initial ASR Output:")
    # L.debug(f"Number of Monologues: {len(output['monologues'])}")
    for idx, monologue in enumerate(output["monologues"]):
        # L.debug(f"Monologue {idx}: {monologue}")

    # Process each monologue to build the utterance_col
    for utterance in output["monologues"]:
        # L.debug(f"Debug - Initial Elements for Speaker {utterance['speaker']}: {utterance['elements']}")

        # Get a list of words with their timestamps
        words = utterance["elements"]
        words = [[i["value"], [round(i["ts"] * 1000) if i.get("ts") is not None else None,
                               round(i["end_ts"] * 1000) if i.get("end_ts") is not None else None]]
                 for i in words if i["value"].strip() != "" and not re.match(r'<.*>', i["value"])]
        # L.debug("Debug - Words after processing (without filtering '[None, None]'):")
        # L.debug(words)

        final_words = []
        for word, (i, o) in words:
            # L.debug(f"Debug - Processing Word: '{word}', Start: {i}, End: {o}")
            if word.strip()[0] == "-" and len(final_words) > 0:
                last = final_words.pop(-1)
                word = last[0].strip() + word.strip()
                i = last[1][0]
                o = o
            word_parts = word.split(" ")
            if len(word_parts) == 1:
                # L.debug(f"Debug - Appending to Final Words: '{word}', Start: {i}, End: {o}")
                final_words.append([word, [i, o]])
                continue
            cur = i
            div = (o - i) // len(word_parts)
            for part in word_parts:
                final_words.append([part.strip(), [cur, cur + div]])
                cur += div

        # L.debug(f"Debug - Final Words after splitting and joining (Speaker {utterance['speaker']}): {final_words}")

        # Check the length and append to utterance_col
        if len(final_words) > 300:
            for i in range(0, len(final_words), 300):
                if len(final_words[i:i + 300]) > 0:
                    utterance_col.append((utterance["speaker"], final_words[i:i + 300]))
        else:
            if len(final_words) > 0:
                utterance_col.append((utterance["speaker"], final_words))

    # If we have an utterance engine, we will use that to retokenize; otherwise retokenize via scanning for punctuation
    # L.debug(f"Debug - Input to Retokenization: {utterance_col}")
    if lang == "yue":
        results = retokenize(utterance_col)
    else:
        if utterance_engine:
            results = retokenize_with_engine(utterance_col, utterance_engine)
        else:
            results = retokenize(utterance_col)

    # L.debug(f"Debug - Results after retokenization: {results}")

    final_utterances = []
    participant_map = {}  # Store the mapping between speaker and Tier

    # Process retokenized results and filter out any word with [None, None] timestamps
    for speaker, utterance in results:
        # Check if the current speaker already has a corresponding Participant
        if speaker not in participant_map:
            participant_map[speaker] = Tier(lang=lang, corpus="corpus_name", id=f"PAR{speaker}", name=f"Participant{speaker}")

        # Use the Participant corresponding to the speaker
        participant = participant_map[speaker]
        
        words = []
        for indx, (word, (start, end)) in utterance:
            # Debug: Show each word and its timestamp before filtering
            # L.debug(f"Debug - Word: '{word}', Start: {start}, End: {end}, Speaker: {speaker}")

            # Skip words with [None, None] timestamps
            if [start, end] == [None, None]:
                # L.debug(f"Debug - Skipping Word: '{word}' due to [None, None] timestamp.")
                continue

            # Add the word to the utterance content
            if word not in ENDING_PUNCT + MOR_PUNCT:
                if start is None or end is None:
                    words.append(Form(text=word, time=None))
                else:
                    words.append(Form(text=word, time=(int(start), int(end))))
            else:
                words.append(Form(text=word, time=None))  # Add punctuation marks without timestamps

        # Only add utterances with content to the final list
        if len(words) > 0:
            # L.debug(f"Debug - Final Utterance Content (Speaker {speaker}): {[form.text for form in words]}")
            final_utterances.append(Utterance(
                tier=participant,
                content=words
            ))

    # L.debug(f"Debug - Final Utterances Before Document Creation: {final_utterances}")
    doc.content = final_utterances
    # L.debug(f"Debug - Document Content After Assignment: {doc.content}")
    doc.langs = [lang]

    return doc
