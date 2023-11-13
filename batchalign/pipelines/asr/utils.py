from batchalign.document import *

def process_generation(output, lang="en"):

    doc = Document()

    for utterance in output["monologues"]:
        participant = Tier(lang=lang, corpus="corpus_name",
                        id=f"PAR{utterance['speaker']}", name=f"Participant{utterance['speaker']}")
        words = []
        for w in utterance["elements"]:
            if w["type"] == "text":
                f = Form(text=w["value"], time=(int(w["ts"]*1000), int(w["end_ts"]*1000)))
                words.append(f)
        ut = Utterance(tier=participant,
                    content=words,
                    alignment=(words[0].time[0], words[-1].time[1]))
        doc.content.append(ut)

    doc.langs = [lang]


    return doc


