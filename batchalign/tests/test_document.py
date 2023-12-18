from batchalign.document import *

TIER = Tier(lang="eng", corpus="corpus_name",
            id="PAR", name="Participant")
PARSED_STANDARD_UTTERANCE = [Form(text="I'm", time=(2530, 2720), morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=[Dependency(id=1, dep_id=3, dep_type='NSUBJ'), Dependency(id=2, dep_id=3, dep_type='AUX')]), Form(text='going', time=(2720, 2910), morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=[Dependency(id=3, dep_id=18, dep_type='ROOT')])]

TEST_UTTERANCE = Utterance.model_validate({
    "tier": TIER,
    "content": PARSED_STANDARD_UTTERANCE
})
CORRECT_STRINGIFICATION = "I'm going \x152530_2910\x15"

TIME_ADJUSTMENT = (2000, 3000)
TIME_ADJUSTED_STRINGIFICATION = "I'm going \x152000_3000\x15"

TOKENIZED_DOC = {'content': [{'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant'}, 'content': [{'text': 'Good', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'morning', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': '!', 'time': None, 'morphology': None, 'dependency': None, 'type': 5}], 'text': None, 'delim': '!', 'time': None, 'custom_dependencies': [], 'alignment': None}, {'tier': {'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant'}, 'content': [{'text': 'How', 'time': None, 'morphology': None, 'dependency': None, 'type':0}, {'text': 'are', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': 'ya', 'time': None, 'morphology': None, 'dependency': None, 'type': 0}, {'text': '?', 'time': None, 'morphology': None, 'dependency': None, 'type': 5}], 'text': None, 'delim': '?', 'time': None, 'custom_dependencies': [], 'alignment': None}], 'media': None, 'langs': ['eng'], 'tiers': [{'lang': 'eng', 'corpus': 'corpus_name', 'id': 'PAR', 'name': 'Participant'}]}

def test_utterance_stringification():
    assert str(TEST_UTTERANCE) == CORRECT_STRINGIFICATION
    tmp = TEST_UTTERANCE.model_copy(deep=True)
    tmp.time = TIME_ADJUSTMENT
    assert str(tmp) == TIME_ADJUSTED_STRINGIFICATION

def test_document_media_seeding():
    doc = Document.new(media_path="test-media-file.wav")
    assert doc.media.url == "test-media-file.wav"

def test_document_tokenization_seeding():
    assert (Document.new("Good morning! How are ya?") ==
            Document.model_validate(TOKENIZED_DOC))



