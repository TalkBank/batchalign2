# TalkBank | Batchalign2

Welcome! **Batchalign2** is a Python suite of language sample analysis (LSA) software from the TalkBank project. It is used to interact with conversation audio files and their transcripts, and provides a whole host of analyses within this space.

The TalkBank Project, of which Batchalign is a part, is supported by NIH grant HD082736.

----

## Quick Start

The following instructions provide a quick start to installing Batchalign. For most users aiming to process CHAT and audio with Batchalign, we recommend more detailed usage instructions: for [usage](https://talkbank.org/info/BA2-usage.pdf) and [human transcript cleanup](https://talkbank.org/info/BA2-cleanup.pdf). The following provides a quick start guide for the program.

### Get Python
- We support Python versions 3.9, 3.10, 3.11 and 3.12.  
- First, check to see if you have Python by running `python`. If it reports any of the versions above, skip the following step.
- To install Python, follow the instructions...
    -  for macOS
        1. Install Brew: [visit this link](https://brew.sh/)
        2. Install Python and Pipx: execute `brew install python@3.11 pipx`
        3. Copy the pipx path to yours: execute `pipx ensurepath` 
    -  for Windows
        1.  Install Python 3.11: [via this link](https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe) 
        2.  If later commands report `pip module not found`, [this page may help](https://github.com/TalkBank/batchalign2/wiki/Troubleshooting-Tips#get-pip-on-windows)
    -  your distribution's instructions for Linux
    
For first-time users of Python, note that if you didn't install Python 3.11 (as we recommended above), it may be complex to change Python versions downstream and may cause additional problems. We recommend explicitly installing Python 3.11 by installing it explicitly via specifying a version number as we show above.

### Install and Update the Package
You can get Batchalign from PyPi, and you can update the package in the same way:

#### macOS

Since Batchalign is a command line program, we recommend using `pipx`, the newly standard environment-based Python program installer, on macOS:

```
pipx install batchalign --python $(brew --prefix)/bin/python3.11
```

#### Linux

```
pip install -U batchalign
```

#### Windows

```
py -m pip install -U batchalign
```

Note that if your system reports `pip: command not found`, replace every use of `pip` in the instructions with `pip3` and try again.

### Rock and Roll
There are two main ways of interacting with Batchalign. Batchalign can be used as a program to batch-process CHAT (hence the name), or as a Python LSA library.

- to get started with the Batchalign program, [tap here](#quick-start-command-line)
- to get started on the Batchalign Library (assumes familiarity with Python), [tap here](#quick-start-python)

## Quick Start: Command Line

### Basic Usage 

Once installed, you can invoke the Batchalign program by typing `batchalign` into the Terminal (MacOS) or Command Prompt (Windows).

It is used in the following basic way:

```
batchalign [verb] [input_dir] [output_dir]
```

Where `verb` includes:

1. `transcribe` - by placing only an audio of video file (`.mp3/.mp4/.wav`) in the input directory, this function performs ASR on the audio, diarizes utterances, identifies some basic conversational features like retracing and filled pauses, and generates word-level alignments. You must supply a language code flag: `--lang=[three letter ISO language code]` for the ASR system to know what language the transcript is in. You can choose the flags `--rev` to use Rev.AI, a commercial ASR service, or `--whisper`, to use a local copy of OpenAI Whisper.
2. `align` - by placing both an audio of video file (`.mp3/.mp4/.wav`) and an *utterance-aligned* CHAT file in the input directory, this function recovers utterance-level time alignments (if they are not already annotated) and generates word-level alignments. The @Languages header in the CHAT file tells the program which language is in the transcript.
3. `morphotag` - by placing a CHAT file in the input directory, this function uses Stanford NLP Stanza to generate morphological and dependency analyses. The @Languages header in the CHAT file tells the program which language is in the transcript. You must supply a language code flag: `--lang=[three letter ISO language code]` for the alignment system to know what language the transcript is in. 
<!-- 4. `bulletize` - placing both an audio of video file (`.mp3/.mp4/.wav`) and an *unlinked* CHAT file in the input directory, generate utterance-level alignments through ASR -->

You can get a CHAT transcript to experiment with [at the TalkBank website](https://talkbank.org/), under any of the "Banks" that are available. You can also generate and parse a CHAT transcript via [the Python program](https://github.com/TalkBank/batchalign2?tab=readme-ov-file#chat).

### Sample Commands
For input files (CHAT and audio for `align`, CHAT only for `morphotag`, and audio only for `transcribe`), located in `~/ba_input` dumping the output to `~/ba_output`, one could write:

#### ASR + Segmentation

```
batchalign transcribe --lang=eng ~/ba_input ~/ba_output
```

#### morphosyntactic analysis

```
batchalign morphotag ~/ba_input ~/ba_output
```

#### forced alignment

```
batchalign align ~/ba_input ~/ba_output
```


-----

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

For instance, here is the instruction for running Batchalign to perform forced-alignment:

```
batchalign align input output
```

With one `-v`, you can get stack trace information about any files that crashes: 

```
batchalign -v align input output
```

and with two `-vv`, we will ditch the loading bar user interface and instead switch to a logging-based interface that has more information about what Batchalign is doing under the hood:

```
batchalign -vv align input output
```

## Quick Start: Python

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
nlp = ba.BatchalignPipeline.new("asr,morphosyntax", lang="eng", num_speakers=2)
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
whisper = ba.WhisperEngine(lang="eng")
# retracing and disfluency analysis
retrace = ba.NgramRetraceEngine()
disfluency = ba.DisfluencyReplacementEngine()
# morphosyntax
morphosyntax = ba.StanzaEngine()

# create a pipeline
nlp = ba.BatchalignPipeline(whisper, retrace, disfluency, morphosyntax)
                             
# and run it!                             
doc = nlp("audio.mp3") 
```

[Here's a list](https://github.com/TalkBank/batchalign2/blob/master/batchalign/pipelines/__init__.py) of available engines.

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
## Questions?
If you have any questions or concerns, please reach out! If something isn't working right, [open an issue on GitHub](https://github.com/TalkBank/batchalign2/issues); if you need support, please feel free to email `houjun@cmu.edu` and `macw@cmu.edu`.

