from batchalign import *
import json

import logging as L 

# LOG_FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
# L.basicConfig(format=LOG_FORMAT, level=L.ERROR)
# L.getLogger("stanza").setLevel(L.ERROR)
# L.getLogger('batchalign').setLevel(L.WARN)

########

from batchalign import *

if __name__ == "__main__":
    cli()
# engine = RevEngine()
# pipe = BatchalignPipeline.new("fa,morphosyntax")
# pipe.tasks

# pipe.tasks

# with open("./batchalign/tests/support/test.json", 'r') as df:
#     d = Document.model_validate(json.load(df))

# d

# d = CHATFile(path="../talkbank-alignment/broken2/input/53.cha").doc
# # (d[12].time)[0]/1000

# d = Document.new("this um is all so crazy so crazy so so crazy so crazy, everybody everybody seem seem so famous famous I am a big scary dinosaur I am a big um um um um scary dinosaur I am a big scary dinosaur.")
# # d[0].text = str(d)

# dis = DisfluencyReplacementEngine()
# nr = NgramRetraceEngine()

# dp = nr(d)
# dp

# # forms = []
# # for utterance in d.content:
# #     for form in utterance.content:
# #         forms.append(form.text)
# # " ".join(forms)

# # tmp = CHATFile(path="./extern/tmp.cha")
# # tmp.doc.media

# # d.media.url = "./extern/tmp.wav"
# e = WhisperFAEngine()

# processing = e(d)
# # d[3]
# # d[1][0]

# CHATFile(doc=processing).write("tmp.cha")

# d[1][0]

# processing[2]

# (path=)

# doc = Document.new("howdy partner, what's your name? my name is joshua")


# doc[0]
# whisper = WhisperEngine()
# tmp = whisper("./extern/tmp.wav")
# doc.tiers
# doc.tiers[0].corpus = "hey"
# p = WhisperFAModel()
# audio = p.load("./batchalign/tests/support/test.mp3")
# transcript = "Hello . This is a test of Batchalign . I'm some body going to read wants told some random crap as I see on the screen . just to test batchline . the primary area for recording editing and me the world's gonna roll me arranging audio. MIDI and drummer regions divided into different track types . Press command slash for more info . test test . I don't know what to say . but um here's some retracing . so just in this for fun . um I like I like I like beans . beans are fun . thank you very much ."


# fa = p(audio.all(), transcript)
# fa


# doc[1].tier
# ud = UDEngine()

# nlp = BatchalignPipeline(ud, whisper)

# nlp.tasks



# <I like beans> <I like beans> <I like> I like <beans><beans>.


# res = nlp(doc)

# nlp.tasks

# tmp

# - engine capabilities: 
#  -- revise to a list of tasks each engine performs
#  -- and the pipeline takes a series of engines, and orders them in the sensical order based on their tasks
#  -- each engine can perform multiple tasks

# doc[0][0]
# from nltk.tokenize import word_tokenize
# tmp = TweetTokenizer()
# word_tokenize("¡por supuesto, maestro!")

# disf = DisfluencyEngine()
# # ore = OrthographyReplacementEngine()
# doc = disf(tmp)
# doc
# # oc = disf(ore(doc))
# from transformers import WhisperForConditionalGeneration
# tmp = WhisperForConditionalGeneration.from_pretrained("talkbank/CHATWhisper-en-large-v1")


# ud = UDEngine()
# doc = ud(doc)
# doc[0][0]
# doc[0][0]
# doc[0][0]



# # CHATFile(path="./tmp.cha").doc

# whisper = WhisperEngine(num_speakers=1)
# doc = whisper.generate("./batchalign/tests/support/test.mp3")


# doc = ud(doc)

# doc[0][0].type

# doc.model_dump()
