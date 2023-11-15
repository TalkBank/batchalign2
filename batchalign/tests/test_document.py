from batchalign.document import *

TIER = Tier(lang="eng", corpus="corpus_name",
            id="PAR", name="Participant")
# STRING_UTTERANCE = Utterance(tier=TIER, content=)

PARSED_STANDARD_UTTERANCE = [Form(text="I'm", time=(2530, 2720), morphology=[Morphology(lemma='I', pos='pron', feats='Prs-Nom-S1'), Morphology(lemma='be', pos='aux', feats='Fin-Ind-1-Pres')], dependency=[Dependency(id=1, dep_id=3, dep_type='NSUBJ'), Dependency(id=2, dep_id=3, dep_type='AUX')]), Form(text='going', time=(2720, 2910), morphology=[Morphology(lemma='go', pos='verb', feats='Part-Pres')], dependency=[Dependency(id=3, dep_id=18, dep_type='ROOT')])]

TEST_UTTERANCE = Utterance.model_validate({
    "tier": TIER,
    "content": PARSED_STANDARD_UTTERANCE
})
CORRECT_STRINGIFICATION = "I'm going \x152530_2910\x15"

TIME_ADJUSTMENT = (2000, 3000)
TIME_ADJUSTED_STRINGIFICATION = "I'm going \x152000_3000\x15"

def test_utterance_stringification():
    assert str(TEST_UTTERANCE) == CORRECT_STRINGIFICATION
    tmp = TEST_UTTERANCE.model_copy(deep=True)
    tmp.time = TIME_ADJUSTMENT
    assert str(tmp) == TIME_ADJUSTED_STRINGIFICATION

    

