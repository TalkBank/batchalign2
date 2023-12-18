# TalkBank | Batchalign2

Hello! This is the working codebase for the next-generation Batchalign project. If you are a user seeking to use Batchalign, please refer to our [original repository](https://github.com/talkbank/batchalign).

Welcome! **Batchalign2** is a Python suite of language sample analysis (LSA) software from the TalkBank project. It is used to interact with conversation audio files and their transcripts, and provides a whole host of analyses within this space.

----

## 🏃 Quick Start

### Get Python
- We support Python versions 3.9, 3.10, and 3.11.  
- **We do not support Python 3.12** (no PyTorch support)
- To install Python, follow the instructions...
    -  [for macOS](https://docs.python-guide.org/starting/install3/osx/)
    -  [for Windows](https://stackoverflow.com/a/15626784)
    -  your distribution's instructions for Linux

### Install the Package
TESTING TESTING TESTING TESTING

You can get Batchalign from PyPi:

macOS/Linux:

```
pip install -U -i https://test.pypi.org/simple/ batchalign
```

Windows:

```
py -m pip install -U -i https://test.pypi.org/simple/ batchalign
```

### Rock and Roll
There are two main ways of interacting with Batchalign. Batchalign can be used as a program to batch-process CHAT (hence the name), or a Python LSA library.

- to get started on the Batchalign CHAT program, [tap here](#-quick-start-python)
- to get started on the Batchalign Python library, [tap here](#-quick-start-command-line)

## 🐍 Quick Start: Python

Let's begin!

```python
import batchalign as ba
```

### Document
The `Document` is the most basic object in Bachalign. All processing pipelines expect `Document` as input, and will spit out `Document` as output.

```python
doc = ba.Document.new("Hello, this is a transcript! I have two utterances.", 
                      media_path="audio.mp3", lang="eng")

# navigating the document
first_utterance = doc[0]
first_form = doc[0][0]
the_comma = doc[0][1]

assert the_comma.text == ','
assert the_comma.type == ba.TokenType.PUNCT

# taking a transcript
sentences = doc.transcript(include_tiers=False, strip=True)
```

Notably, if you have a Document that you haven't transcribed yet, you still can make a Document!

```python
doc = ba.Document.new(media_path="audio.mp3", lang="eng")
```

### Pipelines
<!-- You can process the language samples you got (perform ASR, forced alignment, utterance segmentation, and more!) via `BatchalignPipeline`. There are two levels of access to this API: you can either create a pipeline and use our default settings, or create and customize the underlying `BatchalignEngine`s yourself to perform processing. -->

#### Quick Pipeline
Say you wanted to perform ASR, and then tag morphology of the resulting output.

```python
nlp = ba.BatchalignPipeline.new("asr,morphosyntax", lang_code="eng", num_speakers=2)
doc = ba.Document.new(media_path="audio.mp3", lang="eng")
doc = nlp(doc) # this is equivalent to nlp("audio.mp3"), we will make the initial doc for you

first_word_pos = doc[0][0].morphology
first_word_time = doc[0][0].time
first_utterance_time = doc[0].alignment
```

The quick API (right now) has support for the following tasks, which you can pass in a comma-separated list in the first argument:

- `asr`: ASR!
- `morphosyntax`: PoS and dependency analysis
- `fa`: Forced Alignment (require utterance-level timings already)

We will support many, many, many more tasks soon with this API. For now, to gain access to the whole suite of tools, use the second pipeline API discussed below.

#### Manual Pipeline
Batchalign ships with a plurality of engines which preform the actual processing. For instance, to recreate the demo we had above using the Engines API, we would write

```python
# ASR
whisper = ba.WhisperEngine(lang_code="eng")
# retracing and disfluency analysis
retrace = ba.NgramRetraceEngine()
disfluency = ba.DisfluencyReplacementEngine()
# morphosyntax
morphosyntax = ba.StanzaEngine()

# create a pipeline
nlp = ba.BatchalignPipeline(whisper, retrace, disfluency, morphosyntax)
tasks = nlp.tasks # tasks = [<Task.ASR: 3>, 
                             <Task.UTTERANCE_SEGMENTATION: 4>, 
                             <Task.DISFLUENCY_ANALYSIS: 6>, 
                             <Task.RETRACE_ANALYSIS: 7>, 
                             <Task.MORPHOSYNTAX: 11>]
                             
# and run it!                             
doc = nlp("audio.mp3") 
```

### Formats
We currently support reading and writing two transcript formats: TalkBank CHAT, and Praat TextGrid.

#### CHAT

Here's how to read and write a CHAT file to parse a TalkBank transcript!

```python
# reading
chat = ba.CHATFile(path="chat.cha")
doc = chat.doc

# writing
chat = ba.CHATFile(doc=doc)
chat.write("chat.cha")
```

We will automatically detect audio files located within the same directory as the CHAT file, and associate it with the Batchalign Document.

#### TextGrid

Importantly, there are two ways a TextGrid could be written: we can either place each **utterance** in an individual `IntervalTier`, or each **word** in its own `IntervalTier`; we leave that decision up to you. To learn more about TextGrid, [visit this page](https://github.com/timmahrt/praatIO).

```python
# reading; recall we can either interpret each IntervalTier as a word or utterance
tg_utterance = ba.TextGridFile("utterance", path="tg_ut.TextGrid", lang="eng")
tg_word = ba.TextGridFile("word", path="tg_w.TextGrid", lang="eng")

doc1 = tg_utterance.doc
doc2 = tg_word.doc

# writing
tg_utterance = ba.TextGridFile("utterance", doc=doc1)
tg_word = ba.TextGridFile("word", doc=doc2)

tg_utterance.write("tg_ut.TextGrid")
tg_word.write("tg_w.TextGrid")
```

## 💻 Quick Start: Command Line

### Basic Usage 

Once installed, you can invoke the Batchalign CLI program via the `batchalign` command.

It is used in the following basic way:

```
batchalign [verb] [input_dir] [output_dir]
```

Where `verb` includes:

1. `transcribe` - placing only an audio of video file (`.mp3/.mp4/.wav`) in the input directory, perform ASR on the audio, diarizes utterances, identifies some basic conversational features like retracing and filled pauses, and generate word-level alignments
2. `align` - placing both an audio of video file (`.mp3/.mp4/.wav`) and an *utterance-aligned* CHAT file in the input directory, generate word-level alignments
3. `morphotag` - placing a CHAT file in the input directory, uses Stanford NLP Stanza to generate morphological and dependency analyses; note that the flag `--lang=[two letter ISO language code]`, like `--lang=en` is needed to tell Stanza what language we are working with
<!-- 4. `bulletize` - placing both an audio of video file (`.mp3/.mp4/.wav`) and an *unlinked* CHAT file in the input directory, generate utterance-level alignments through ASR -->

Follow instructions from

```
batchalign --help
```

and 

```
batchalign [verb] --help
```

to learn more about other options.

### Verbosity

Placing one or multiple `-v` *behind the word `batchalign`* (i.e. behind the `[verb]` will not work) increases the verbosity of Batchalign. The default mode and one `-v` will use the normal Batchalign interface, whereas Batchalign with more than 1 `-v` will switch to the text-based "logging" interface.

For instance, here is Batchalign running normally with a single crashing file:

```
batchalign align input output
```

<img width="649" alt="image" src="https://github.com/TalkBank/batchalign2/assets/28765741/d1c31f98-19fb-4fba-bdeb-ca9cd242b7b6">

With one `-v`: 

```
batchalign -v align input output
```

<img width="645" alt="image" src="https://github.com/TalkBank/batchalign2/assets/28765741/a516347e-5acc-4857-b0a0-9583906d2f55">

and with two `-vv`:

```
batchalign -vv align input output
```

<img width="641" alt="image" src="https://github.com/TalkBank/batchalign2/assets/28765741/9c4ffbf2-6dab-4574-a69d-ffce6e1761a8">

Once we get to 2 `-v`s, Batchalign switches to logging based interface that will give more detail under the hood.

## 🙌 Questions?
If you have any questions or concerns, please reach out! If something isn't working right, [open an issue on GitHub](https://github.com/TalkBank/batchalign2/issues); if you need support, please feel free to email `houjun@cmu.edu` and `macw@cmu.edu`.

