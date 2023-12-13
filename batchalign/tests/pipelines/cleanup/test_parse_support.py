from batchalign.document import *
from batchalign.pipelines.cleanup.parse_support import _mark_utterance, parse, Replacement

CORRECT_PARSE = {'hello': Replacement(original='hello', main_line_replacement='is', lemma_replacement='a'), 'test': Replacement(original='test', main_line_replacement='for', lemma_replacement='the'), 'support': Replacement(original='support', main_line_replacement='parses', lemma_replacement='that')}
MARKED_MODEL = {'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR0', 'name': 'Participant'}, 'content': [{'text': 'a', 'time': (140, 610), 'morphology': [{'lemma': 'hello', 'pos': 'intj', 'feats': ''}], 'dependency': [{'id': 1, 'dep_id': 1, 'dep_type': 'ROOT'}], 'type': TokenType.CORRECTION}, {'text': '.', 'time': None, 'morphology': [{'lemma': '.', 'pos': 'PUNCT', 'feats': ''}], 'dependency': [{'id': 2, 'dep_id': 1, 'dep_type': 'PUNCT'}], 'type': TokenType.REGULAR}], 'text': 'is . \x15140_610\x15', 'delim': '.', 'time': (140, 610), 'custom_dependencies': [], 'alignment': (140, 610)}

def test_parse():
    p = parse("test.test")
    assert p == CORRECT_PARSE

def test_parse_missing():
    p = parse("test.testmmm")
    assert p == {}

def test_mark_utterance(en_utterance):
    copy = Utterance.model_validate(en_utterance.model_dump())
    _mark_utterance(copy, "test", TokenType.CORRECTION, "test")

    assert copy == Utterance.model_validate(MARKED_MODEL)



