from batchalign.document import *
from batchalign.utils.dp import align, ReferenceTarget, PayloadTarget, Match
from batchalign.formats.chat.utils import annotation_clean

def bulletize_doc(asr, doc):
    """Use ASR output to add estimated word-level timings to a doc.

    Parameters
    ----------
    asr : Dict
        The raw ASR output.
    doc : Document
        The document that needs to be timing added.

    Returns
    -------
    Document
        The corrected document.
    """

    # collect and sort the raw words together with their timings
    raw_words = []
    for i in asr["monologues"]:
        raw_words += [j for j in i["elements"]
                    if j.get("ts") and j.get("end_ts")]
    # sort and serialize
    payloads = []
    for i in sorted(raw_words, key=lambda x:x["ts"]):
        if annotation_clean(i["value"]).strip() != "":
            payloads.append(PayloadTarget(annotation_clean(i["value"]).lower(),
                                        (i["ts"], i["end_ts"])))
    # and seralize the referneces
    backplates = []
    for indx_i, i in enumerate(doc.content):
        if isinstance(i, Utterance):
            for indx_j, j in enumerate(i.content):
                if annotation_clean(j.text).strip() != "":
                    backplates.append(ReferenceTarget(annotation_clean(j.text).lower(), 
                                                    (indx_i, indx_j)))
    # aligment time!
    alignments = align(payloads, backplates, False)

    # for each aligned element, set timing
    for i in alignments:
        if isinstance(i, Match):
            a,b = i.reference_payload
            doc[a][b].time = (int(round(i.payload[0]*1000)),
                              int(round(i.payload[1]*1000)))

    return doc 

        
