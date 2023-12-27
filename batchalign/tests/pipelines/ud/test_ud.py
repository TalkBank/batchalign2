from batchalign.pipelines.morphosyntax.ud import morphoanalyze
from batchalign.document import *

import warnings

CEST_TAGGED = [{'lemma': 'ce', 'pos': 'pron', 'feats': 'Dem&S3'},
               {'lemma': 'être', 'pos': 'aux', 'feats': 'Fin&Ind&Pres&S3'}]

JUSQU_AU_TAGGED = {'text': "jusqu'au",
                   'time': None,
                   'morphology': [{'lemma': 'jusque', 'pos': 'adp', 'feats': ''},
                                  {'lemma': 'au', 'pos': 'adv', 'feats': ''}],
                   'dependency': [{'id': 4, 'dep_id': 5, 'dep_type': 'CASE'},
                                  {'id': 5, 'dep_id': 3, 'dep_type': 'ADVMOD'}],
                   'type': 0}

TRIPLE_CLITIC_FORM = {'text': "d'l'attraper", 'time': None, 'morphology': [{'lemma': 'de', 'pos': 'adp', 'feats': ''}, {'lemma': 'lui', 'pos': 'pron', 'feats': 'Prs&S3'}, {'lemma': 'attraper', 'pos': 'verb', 'feats': 'Inf&S'}], 'dependency': [{'id': 3, 'dep_id': 5, 'dep_type': 'MARK'}, {'id': 4, 'dep_id': 5, 'dep_type': 'OBJ'}, {'id': 5, 'dep_id': 2, 'dep_type': 'XCOMP'}], 'type': 0}

def test_ud_pipeline(en_doc):
    assert morphoanalyze(en_doc) == en_doc
    
# email dec092023-10:31
def test_ud_cest_mwt():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("c'est du réglisse .", lang="fra"))

        assert [i.model_dump() for i in res[0][0].morphology] == CEST_TAGGED

# email dec202023-10:02
def test_ud_jusqu():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("tu vas aller jusqu'au.", lang="fra"))
        assert res[0][-2].model_dump() == JUSQU_AU_TAGGED

# email dec222023-05:57
# based utterance
def test_ud_triple_clitics():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = morphoanalyze(Document.new("pour essayer d'l'attraper.", lang="fra"))
        assert res[0][-2].model_dump() == TRIPLE_CLITIC_FORM


# email emaildec232023-08:3 
# utterance that's fundimentally empty
def test_empty_utterance():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        from batchalign.formats.chat.parser import chat_parse_utterance

        text = "&~eu &~cho xxx+//."

        forms, delim = chat_parse_utterance(text, None, None, None, None)
        utterance = Utterance(content=forms, delim=delim)
        ut = Document(content=[utterance], langs=["fra"])

        morphoanalyze(ut)

