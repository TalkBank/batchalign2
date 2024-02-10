from batchalign.formats.chat.generator import *
from batchalign.formats.chat.file import *
from batchalign.document import *
from batchalign.errors import *

import os
import pathlib

# utterance with everything
EVERYTHING_UTTERANCE = {'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR0', 'name': 'Participant'}, 'content': [{'text': 'Um', 'time': (20780, 21330), 'morphology': [{'lemma': 'um', 'pos': 'intj', 'feats': ''}], 'dependency': [{'id': 1, 'dep_id': 3, 'dep_type': 'DISCOURSE'}]}, {'text': 'I', 'time': (21360, 21480), 'morphology': None, 'dependency': None}, {'text': 'like', 'time': (21480, 21720), 'morphology': None, 'dependency': None}, {'text': 'I', 'time': (22180, 22310), 'morphology': None, 'dependency': None}, {'text': 'like', 'time': (22310, 22710), 'morphology': None, 'dependency': None}, {'text': 'I', 'time': (22710, 22760), 'morphology': [{'lemma': 'I', 'pos': 'pron', 'feats': 'Prs-Nom-S1'}], 'dependency': [{'id': 2, 'dep_id': 3, 'dep_type': 'NSUBJ'}]}, {'text': 'like', 'time': (22760, 23270), 'morphology': [{'lemma': 'like', 'pos': 'verb', 'feats': 'Fin-Ind-1-Pres'}], 'dependency': [{'id': 3, 'dep_id': 4, 'dep_type': 'ROOT'}]}, {'text': 'beans', 'time': (23300, 24300), 'morphology': [{'lemma': 'bean', 'pos': 'noun', 'feats': 'ComNeut-Plur'}], 'dependency': [{'id': 4, 'dep_id': 3, 'dep_type': 'OBJ'}]}, {'text': '.', 'time': None, 'morphology': [{'lemma': '.', 'pos': 'PUNCT', 'feats': ''}], 'dependency': [{'id': 5, 'dep_id': 3, 'dep_type': 'PUNCT'}]}], 'text': 'Um <I like I like> [/] I like beans . \x1520780_24300\x15', 'delim': '.', 'time': (20780, 24300), 'custom_dependencies': [], 'alignment': (20780, 24300)}
EVERYTHING_CORRECT = '*PAR0:\tUm <I like I like> [/] I like beans . \x1520780_24300\x15\n%mor:\tintj|um pron|I-Prs-Nom-S1 verb|like-Fin-Ind-1-Pres noun|bean-ComNeut-Plur .\n%gra:\t1|3|DISCOURSE 2|3|NSUBJ 3|4|ROOT 4|3|OBJ 5|3|PUNCT\n%wor:\tUm \x1520780_21330\x15 I \x1521360_21480\x15 like \x1521480_21720\x15 I \x1522180_22310\x15 like \x1522310_22710\x15 I \x1522710_22760\x15 like \x1522760_23270\x15 beans \x1523300_24300\x15 .'

# utterance which doesn't have an string text field which therefore needs to be detokenized manually
DETOK_UTTERANCE = {'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'STU', 'name': 'Participant'}, 'content': [{'text': 'right', 'time': None, 'morphology': [{'lemma': 'right', 'pos': 'adv', 'feats': ''}], 'dependency': [{'id': 1, 'dep_id': 3, 'dep_type': 'JCT'}]}, {'text': ',', 'time': None, 'morphology': [{'lemma': 'cm', 'pos': 'cm', 'feats': ''}], 'dependency': [{'id': 2, 'dep_id': 1, 'dep_type': 'LP'}]}, {'text': 'right', 'time': None, 'morphology': [{'lemma': 'right', 'pos': 'co', 'feats': ''}], 'dependency': [{'id': 3, 'dep_id': 0, 'dep_type': 'INCROOT'}]}, {'type': 5, 'text': '.', 'time': None, 'morphology': [{'lemma': '.', 'pos': 'PUNCT', 'feats': ''}], 'dependency': [{'id': 4, 'dep_id': 3, 'dep_type': 'PUNCT'}]}], 'text': None, 'delim': '.', 'time': (332418, 333654), 'custom_dependencies': [], 'alignment': (332418, 333654)}
DETOK_CORRECT = '*STU:\tright , right . \x15332418_333654\x15\n%mor:\tadv|right cm|cm co|right .\n%gra:\t1|3|JCT 2|1|LP 3|0|INCROOT 4|3|PUNCT'

CORRECT_HEADER = '@Languages:\teng\n@Participants:\tPAR0 Participant\n@Options:\tmulti\n@ID:\teng|corpus_name|PAR0|||||Participant|||\n@Media:\ttest, audio'

def test_utterance_to_chat():
    assert generate_chat_utterance(Utterance.model_validate(DETOK_UTTERANCE)) == DETOK_CORRECT
    assert generate_chat_utterance(Utterance.model_validate(EVERYTHING_UTTERANCE)) == EVERYTHING_CORRECT

def test_preamble_generation():
    dir = pathlib.Path(__file__).parent.resolve()
    c = CHATFile(path=os.path.join(dir, "support", "test.cha"))

    assert generate_chat_preamble(c.doc) == CORRECT_HEADER

def test_last_line(en_doc):
    chat = CHATFile(doc=en_doc)
    chatstr = str(chat)

    assert chatstr[-5:] == "@End\n"



