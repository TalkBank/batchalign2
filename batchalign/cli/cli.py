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
                       type=click.Path(exists=True, file_okay=False)),
        click.option("--lang",
                     help="sample language in three-letter ISO 3166-1 alpha-3 code",
                     show_default=True,
                     default="eng",
                     type=str),
        click.option("-n", "--num_speakers", type=int,
                     help="number of speakers in the language sample", default=2),
        
    ]

    options.reverse()
    return functools.reduce(lambda x, opt: opt(x), options, f)

###################### UTILS ##############################

def handle_verbosity(verbosity):
    L.getLogger('stanza').handlers.clear()
    L.getLogger('transformers').handlers.clear()
    L.getLogger("stanza").setLevel(L.INFO)
    L.getLogger('batchalign').setLevel(L.WARN)
    L.getLogger('lightning.pytorch.utilities.migration.utils').setLevel(L.ERROR)

    if verbosity >= 2:
        L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
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
    pretty.install()
    install()

#################### ALIGN ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="For utterance timing recovery, OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.pass_context
def align(ctx, in_dir, out_dir, lang, num_speakers, whisper, **kwargs):
    """Align transcripts against corresponding media files."""
    def loader(file):
        return CHATFile(path=os.path.abspath(file), special_mor_=True).doc

    def writer(doc, output):
        CHATFile(doc=doc, special_mor_=True).write(output)

    _dispatch("align", lang, num_speakers,
              ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C,
              utr="whisper_utr" if whisper else "rev_utr",
              **kwargs)

#################### TRANSCRIBE ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="Use OpenAI Whisper (ASR) instead of Rev.AI (default).")
@click.option("--whisperx/--rev",
              default=False, help="Use WhisperX instead of Rev.AI (default). Superceeds --whisper.")
@click.pass_context
def transcribe(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Create a transcript from audio files."""
    def loader(file):
        return file

    def writer(doc, output):
        CHATFile(doc=doc, special_mor_=True).write(output
                                                   .replace(".wav", ".cha")
                                                   .replace(".mp4", ".cha")
                                                   .replace(".mp3", ".cha"))

    asr = "rev"
    if kwargs["whisper"]:
        asr = "whisper"
    if kwargs["whisperx"]:
        asr = "whisperx"

    _dispatch("transcribe", lang, num_speakers, ["mp3", "mp4", "wav"], ctx,
              in_dir, out_dir,
              loader, writer, C,
              asr=asr, **kwargs)

#################### MORPHOTAG ################################

@batchalign.command()
@common_options
@click.option("--retokenize/--keeptokens",
              default=False, help="Retokenize the main line to fit the UD tokenizations.")
@click.pass_context
def morphotag(ctx, in_dir, out_dir, lang, num_speakers, **kwargs):
    """Perform morphosyntactic analysis on transcripts."""

   
    def loader(file):
        return (
            CHATFile(path=os.path.abspath(file), special_mor_=True).doc,
            {"retokenize": kwargs["retokenize"]}
        )

    def writer(doc, output):
        CHATFile(doc=doc, special_mor_=True).write(output)

    _dispatch("morphotag", lang, num_speakers, ["cha"], ctx,
              in_dir, out_dir,
              loader, writer, C)


#################### BENCHMARK ################################

@batchalign.command()
@common_options
@click.option("--whisper/--rev",
              default=False, help="Use OpenAI Whisper (ASR) instead of Rev.AI (default).")
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
