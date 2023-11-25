import re
from batchalign.document import *
from batchalign.utils import *

from batchalign.constants import ENDING_PUNCT

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
            tmp.append((word, bullet))
            if word in ENDING_PUNCT or word[-1] in ENDING_PUNCT:
                final_outputs.append((speaker, tmp))
                tmp = []

    return final_outputs

def process_generation(output, lang="eng"):
    """Process Rev.AI style ASR generation

    Parameters
    ----------
    output : dict
        The raw Rev.AI style output from your ASR engine.
    lang : str
        The language ID.

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
        words = [[i["value"], [round(i["ts"]*1000),
                                round(i["end_ts"]*1000)]] # the shape
                for i in words # for each word
                    if i["type"] == "text" and
                    not re.match(r'<.*>', i["value"])] # if its text (i.e. not "pause")

        # sometimes, the system outputs two forms with a space as one single
        # word. we need to interpolate the space between them
        final_words = []
        # go through the words, if there is a space, split time in n parts
        for word, (i,o) in words:
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


    results = retokenize(utterance_col)

    final_utterances = []
    for speaker, utterance in results:
        participant = Tier(lang=lang, corpus="corpus_name",
                           id=f"PAR{speaker}",
                           name=f"Participant")
        words = []
        for word, (start,end) in utterance:
            if word not in ENDING_PUNCT:
                words.append(Form(text=word, time=(int(start), int(end))))
            else:
                words.append(Form(text=word, time=None))

        final_utterances.append(Utterance(
            tier=participant,
            content = words
        ))

    doc.content = final_utterances

    return doc


