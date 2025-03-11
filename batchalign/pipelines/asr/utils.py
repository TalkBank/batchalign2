import re
from batchalign.document import *
from batchalign.utils import *

from batchalign.constants import ENDING_PUNCT
from batchalign.pipelines.asr.num2chinese import num2chinese

from num2words import num2words
import pycountry


def retokenize(intermediate_output):
    """Retokenize the output of the ASR system from one giant blob to utterances

    Ideally, we'd split by utterances. For the custom-trained models that
    we ship with Batchalign, indeed that's what happens; they are trained to
    delineate sentences through utterance boundaries. However, for non-English
    ASR we rely on third party solutinos which doesn't have such facilities.

    Therefore, we at least split by sentences to get a relative tokenization.
    """

    final_outputs = []

    for speaker, utterance in intermediate_output:

        # we go through each of the words, and if the word is an
        # ending PUNCT, or if the word ENDS in a PUNCT, we split it out
        # as a sub-utterance
        tmp = []
        for word, bullet in utterance:
            word = word.replace("。", ".")
            word = word.replace("¿", " ").replace("¡", " ")
            tmp.append((word, bullet))
            if len(word) > 0 and (word in ENDING_PUNCT or word[-1] in ENDING_PUNCT):
                if word in ENDING_PUNCT:
                    final_outputs.append((speaker, tmp))
                elif word[-1] in ENDING_PUNCT:
                    # we want to seperate the ending punct out
                    final, time = tmp.pop(-1)
                    tmp.append((final[:-1], time))
                    tmp.append((final[-1], [None, None]))
                    final_outputs.append((speaker, tmp))
                tmp = []

        if len(tmp) > 0: 
            if len(tmp[-1]) > 0 and tmp[-1][0] in MOR_PUNCT:
                tmp.pop(-1)
            tmp.append((".", [None, None]))
            final_outputs.append((speaker, tmp))
            tmp = []

    return final_outputs

def retokenize_with_engine(intermediate_output, engine):
    """Retokenize `intermediate_input` given utterance engine `engine`

    This function is different from `retokenize` because it doesn't look
    at the original punctuation from the document at all; intsead, it retokenizes
    using a seperate utterance tokenization engine.

    Parameters
    ----------
    intermediate_output : List
        Rev.AI style output.
        
    engine : UtteranceEngine
        The utterance Engine to use.
    """
    
    final_outputs = []

    for speaker, utterance in intermediate_output:
        # because we are using an utterance engine, we need
        # to get rid of all the preexisting punctuation
        for i in utterance:
            for j in MOR_PUNCT+ENDING_PUNCT:
                i[0] = i[0].strip(j).lower()

        # remove everything that's now blank
        utterance = [i for i in utterance if i[0].strip() != ""]

        joined = " ".join([i[0] for i in utterance])
        joined = joined.replace("。", ".")
        split = engine(joined)

        # Initialize current index to track position in original utterance
        current_index = 0

        # align the utterance against original splits and generate final outputs
        for i in split:
            # Check if the split has ending punctuation
            if i[-1] in ENDING_PUNCT:
                new_ut, delim = (i[:-1].split(" "), i[-1])
            else:
                new_ut, delim = (i.split(" "), ".")

            tmp = []

            for s in new_ut:
                if current_index < len(utterance):
                    # Use current element and move index forward
                    tmp.append((s, utterance[current_index][1]))
                    current_index += 1
                else:
                    # Append with default timestamp if utterance is exhausted
                    tmp.append((s, [None, None]))

            final_outputs.append((speaker, tmp+[[delim, [None, None]]]))

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

    for utterance in output["monologues"]:
        # get a list of words
        words = utterance["elements"]
        # coallate words (not punct) into the shape we expect
        # which is ['word', [start_ms, end_ms]]. Yes, this would
        # involve multiplying by 1000 to s => ms
        words = [[i["value"], [round(i["ts"]*1000) if i.get("ts") != None else None,
                                round(i["end_ts"]*1000) if i.get("end_ts") != None else None]] # the shape
                for i in words # for each word
                    if i["value"].strip() != "" and
                    not re.match(r'<.*>', i["value"])] # if its text (i.e. not "pause")

        # sometimes, the system outputs two forms with a space as one single
        # word. we need to interpolate the space between them
        final_words = []
        # go through the words, if there is a space, split time in n parts
        for word, (i,o) in words:
            # if there is a dash in the beginning, we join it with the last one
            if word.strip()[0] == "-" and len(final_words) > 0:
                last = final_words.pop(-1)
                word = last[0].strip()+word.strip()
                i = last[1][0]
                o = o
            # split the word
            word_parts = word.split(" ")
            # if we only have one part, we don't interpolate
            if len(word_parts) == 1:
                final_words.append([word, [i,o]])
                continue
            # otherwise, we interpolate the itme
            cur = i
            div = (o-i)//len(word_parts)
            # for each part, add the start and end
            for part in word_parts:
                final_words.append([part.strip(), [cur, cur+div]])
                cur += div

        if lang == "yue":
            lang_2 = "yue"
        elif lang == "mys":
            lang_2 = "mys"
        else:
            lang_2 = pycountry.languages.get(alpha_3=lang).alpha_2
        def catched_num2words(i):
            if not i.isdigit():
                return i
            try:
                return num2words(i, lang=lang_2)
            except NotImplementedError:
                try:
                    if lang == "zho":
                        return num2chinese(i)
                    elif lang == "jpn":
                        return num2chinese(i, simp=False)
                    elif lang == "yue":
                        return num2chinese(i, simp=False)
                    else:
                        return i
                except:
                    return i
        final_words = [[catched_num2words(i), j] for i,j in final_words]

        # if the final words is > 300, split into n parts
        if len(final_words) > 300:
            # for each group, append
            for i in range(0, len(final_words), 300):
                # concatenate with speaker tier and append to final collection
                # not unless the crop is empty
                if len(final_words[i:i+300]) > 0:
                    utterance_col.append((utterance["speaker"], final_words[i:i+300]))
        else:
            # concatenate with speaker tier and append to final collection
            if len(final_words) > 0:
                utterance_col.append((utterance["speaker"], final_words))

    # if we have an uttetrance engine, we will use that to retokenize; otherwise
    # we retokenize via scanning for punctation
    if utterance_engine:
        results = retokenize_with_engine(utterance_col, utterance_engine)
    else:
        results = retokenize(utterance_col)

    final_utterances = []
    for speaker, utterance in results:
        participant = Tier(lang=lang, corpus="corpus_name",
                           id=f"PAR{speaker}",
                           name=f"Participant")
        words = []
        for indx, (word, (start,end)) in enumerate(utterance):
            if indx == 0:
                seen_word = False
            if word.strip() == "":
                continue
            if word not in ENDING_PUNCT+MOR_PUNCT:
                if start == None or end == None:
                    words.append(Form(text=word, time=None))
                else:
                    seen_word = True
                    words.append(Form(text=word, time=(int(start), int(end))))
            else:
                    words.append(Form(text=word, time=None))

        final_utterances.append(Utterance(
            tier=participant,
            content=words
        ))

    doc.content = final_utterances
    doc.langs = [lang]

    return doc


