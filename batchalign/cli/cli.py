"""
cli.py
The Batchalign command-line interface
"""

import multiprocessing
import rich_click as click
import functools

import os
from glob import glob

from multiprocessing import Process, freeze_support

from batchalign.pipelines import BatchalignPipeline

from rich.traceback import install
from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from batchalign.document import *
from batchalign.formats.chat import CHATFile
from batchalign.utils import config
from rich.logging import RichHandler

from batchalign.cli.dispatch import _dispatch
from batchalign.models.training.run import cli as train

from enum import Enum

import traceback

import pyfiglet
from rich import pretty
import logging as L 
baL = L.getLogger('batchalign')

C = Console()


with open(Path(__file__).parent.parent / "version", 'r') as df:
    VERSION_NUMBER, RELEASE_DATE, RELEASE_NOTES = df.readlines()[:3]

#################### OPTIONS ################################

# common options for batchalign
def common_options(f):
    options = [
        click.argument("in_dir",
                       type=click.Path(exists=True, file_okay=False)),
        click.argument("out_dir",
                       type=click.Path(exists=True, file_okay=False))
    ]

    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)

###################### UTILS ##############################

def handle_verbosity(verbosity):
    L.shutdown()
    L.getLogger('stanza').handlers.clear()
    L.getLogger('transformers').handlers.clear()
    L.getLogger('nemo_logger').handlers.clear()
    L.getLogger("stanza").setLevel(L.INFO)
    L.getLogger('nemo_logger').setLevel(L.CRITICAL)
    L.getLogger('batchalign').setLevel(L.WARN)
    L.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(L.ERROR)

    if verbosity >= 2:
        L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
        L.getLogger('nemo_logger').setLevel(L.INFO)
        L.getLogger('batchalign').setLevel(L.INFO)
    if verbosity >= 3:
        L.getLogger('batchalign').setLevel(L.DEBUG)
    if verbosity >= 4:
        L.getLogger('batchalign').setLevel(L.DEBUG)
        L.getLogger('transformers').setLevel(L.INFO)

@click.group()
@click.pass_context
@click.option("-v", "--verbose", type=int, count=True, default=0, help="How loquacious Batchalign should be.")
def batchalign(ctx, verbose):
    """process .cha and/or audio files in IN_DIR and dumps them to OUT_DIR using recipe COMMAND"""

    ## setup commands ##
    # multiprocessing thread freeze
    freeze_support()
    # ensure that the context object is a dictionary
    ctx.ensure_object(dict)
    # verbosity
    handle_verbosity(verbose)
    # add to arguments
    ctx.obj["verbose"] = verbose
    # setup config
    ctx.obj["config"] = config.config_read(True)
    # make everything look better
    # pretty.install()
    # better tracebacks
    install()

batchalign.add_command(train, "models")

#################### ALIGN ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="For utterance timing recovery, OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--wav2vec/--whisper_fa",
              default=True, help="Use Whisper instead of Wav2Vec for English (defaults for Whisper for non-English)")
@click.option("--pauses", type=bool, default=False, help="Should we try to bullet each word or should we try to add pauses in between words by grouping them? Default: no pauses.", is_flag=True)

@click.pass_context
def align(ctx, in_dir, out_dir, whisper, wav2vec, **kwargs):
    """Align transcripts against corresponding media files."""
    def loader(file):
        return (
            CHATFile(path=os.path.abspath(file)).doc,
            {"pauses": kwargs.get("pauses", False)}
        )

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    if not wav2vec:
        _dispatch("align", "eng", 1,
                  ["cha"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  fa="whisper_fa",
                  utr="whisper_utr" if whisper else "rev_utr",
                  **kwargs)
    else:
        _dispatch("align", "eng", 1,
                  ["cha"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  fa="wav2vec_fa",
                  utr="whisper_utr" if whisper else "rev_utr",
                  **kwargs)

#################### TRANSCRIBE ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="Use OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--tencent/--rev",
              default=False, help="Use Tencent instead of Rev.AI (default).")
@click.option("--whisperx/--rev",
              default=False, help="Use WhisperX instead of Rev.AI (default). Superceeds --whisper.")
@click.option("--diarize/--nodiarize",
              default=False, help="Perform speaker diarization (this flag is ignored with Rev.AI)")
@click.option("--wor/--nowor",
              default=False, help="Should we write word level alignment line? Default to no.")
@click.option("--data",
              help="the URL of the data",
              type=str)
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def transcribe(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Create a transcript from audio files."""
    def loader(file):
        return file

    asr = "rev"
    if kwargs["whisper"]:
        asr = "whisper"
    if kwargs["whisperx"]:
        asr = "whisperx"
    if kwargs["tencent"]:
        asr = "tencent"

    def writer(doc, output):
        doc.content.insert(0, CustomLine(id="Comment", type=CustomLineType.INDEPENDENT,
                                         content=f"Batchalign {VERSION_NUMBER.strip()}, ASR Engine {asr}. Unchecked output of ASR model; do not use."))
        CHATFile(doc=doc).write(output
                                .replace(".wav", ".cha")
                                .replace(".WAV", ".cha")
                                .replace(".mp4", ".cha")
                                .replace(".MP4", ".cha")
                                .replace(".mp3", ".cha")
                                .replace(".MP3", ".cha"),
                                write_wor=kwargs.get("wor", False))

    if kwargs.get("diarize"):
        _dispatch("transcribe_s",
                  lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  asr=asr, **kwargs)
    else:
        _dispatch("transcribe",
                  lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
                  in_dir, out_dir,
                  loader, writer, C,
                  asr=asr, **kwargs)

#################### TRANSLATE ################################

@batchalign.command()
@common_options
@click.pass_context
def translate(ctx, in_dir, out_dir, **kwargs):
    """Translate the transcript to English."""

    def loader(file):
        cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
        doc = cf.doc
        # if str(cf).count("%mor") > 0:
        #     doc.ba_special_["special_mor_notation"] = True
        return doc

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("translate", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)

#################### MORPHOTAG ################################

@batchalign.command()
@common_options
@click.option("--retokenize/--keeptokens",
              default=False, help="Retokenize the main line to fit the UD tokenizations.")
@click.option("--lexicon",
              type=click.Path(exists=True,
                              file_okay=True, dir_okay=False),
              help="Comma seperated manual lexicon override")
@click.pass_context
def morphotag(ctx, in_dir, out_dir, **kwargs):
    """Perform morphosyntactic analysis on transcripts."""

    def loader(file):
        mwt = {}
        if kwargs.get("lexicon") != None and kwargs.get("lexicon", "").strip() != "":
            import csv
            raw = []
            with open(kwargs["lexicon"], 'r') as df:
                raw = [i for i in csv.reader(df)]
            for i in raw:
                mwt[i[0]] = tuple(i[1:])
        cf = CHATFile(path=os.path.abspath(file), special_mor_=True)
        doc = cf.doc
        if str(cf).count("%mor") > 0:
            doc.ba_special_["special_mor_notation"] = True
        return (
            doc,
            {"retokenize": kwargs["retokenize"], "mwt": mwt}
        )

    def writer(doc, output):
        CHATFile(doc=doc, special_mor_=doc.ba_special_.get("special_mor_notation", False)).write(output)

    _dispatch("morphotag", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)


#################### MORPHOTAG ################################

@batchalign.command(hidden=True)
@common_options
@click.pass_context
def coref(ctx, in_dir, out_dir, **kwargs):
    """Perform coreference analysis on transcripts."""

    def loader(file):
        cf = CHATFile(path=os.path.abspath(file))
        doc = cf.doc
        return doc, {}

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("coref", "eng", 1, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)


#################### UTSEG ################################

@batchalign.command()
@common_options
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def utseg(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Perform morphosyntactic analysis on transcripts."""

    def loader(file):
        return CHATFile(path=os.path.abspath(file)).doc

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("utseg", lang, num_speakers, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)

#################### BENCHMARK ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="Use OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--lang",
              help="sample language in three-letter ISO 3166-1 alpha-3 code",
              show_default=True,
              default="eng",
              type=str)
@click.option("-n", "--num_speakers", type=int, help="number of speakers in the language sample", default=2)
@click.pass_context
def benchmark(ctx, in_dir, out_dir, lang, num_speakers, whisper, **kwargs):
    """Benchmark ASR utilities for their word accuracy"""
    def loader(file):
        # try to find a .cha in the same directory
        p = Path(file)
        cha = p.with_suffix(".cha")
        # if there are not cha file found, we complain
        if not cha.exists():
            raise FileNotFoundError(f"No gold .cha transcript found, we cannot do benchmarking. audio: {p.name}, desired cha: {cha.name}, looked in: {str(cha)}.")
        # otherwise, load the goald along with the input file
        return file, {"gold": CHATFile(path=str(cha), special_mor_=True).doc}

    def writer(doc, output):
        # delete the copied cha file
        os.remove(Path(output).with_suffix(".cha"))
        # write the wer
        with open(Path(output).with_suffix(".wer.txt"), 'w') as df:
            df.write(str(doc["wer"]))
        with open(Path(output).with_suffix(".diff"), 'w') as df:
            df.write(str(doc["diff"]))
        CHATFile(doc=doc["doc"]).write(str(Path(output).with_suffix(".asr.cha")))


    _dispatch("benchmark", lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
              in_dir, out_dir,
              loader, writer, C,
              asr="whisper" if whisper else "rev", **kwargs)


#################### SETUP ################################

@batchalign.command()
@click.pass_context
def setup(ctx):
    """Reconfigure Batchalign settings, such as Rev.AI key."""

    config.interactive_setup()

#################### VERSION ################################

@batchalign.command()
@click.pass_context
def version(ctx, **kwargs):
    """Print program version info and exit."""

    ptr = (pyfiglet.figlet_format("Batchalign2")+"\n" +
           f"Version: [bold]{VERSION_NUMBER.strip()}[/bold], released {RELEASE_DATE.strip()}\n" +
           f"[italic]{RELEASE_NOTES.strip()}[/italic]"+"\n" +
           "\nDeveloped by Brian MacWhinney and Houjun Liu")
    C.print("\n\n"+ptr+"\n\n")
