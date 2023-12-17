"""
cli.py
The Batchalign command-line interface
"""

import multiprocessing
import click
import functools

import os
from glob import glob

from multiprocessing import Process, freeze_support

from batchalign.pipelines import BatchalignPipeline
from batchalign.constants import VERSION_NUMBER, RELEASE_DATE, RELEASE_NOTES

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
    L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
    L.getLogger("stanza").setLevel(L.ERROR)
    L.getLogger('batchalign').setLevel(L.ERROR)

    if verbosity >= 2:
        L.getLogger('batchalign').setLevel(L.INFO)
    if verbosity >= 3:
        L.getLogger('batchalign').setLevel(L.DEBUG)

@click.group()
@click.pass_context
@click.option("-v", "--verbose", type=int, count=True, default=0, help="How loquacious Batchalign should be.")
def batchalign(ctx, verbose):
    """process CHAT files in IN_DIR and dumps them to OUT_DIR using recipe COMMAND"""

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

#################### ALIGN ################################

@batchalign.command()
@common_options
@click.pass_context
def align(ctx, in_dir, out_dir, **kwargs):
    """Align transcripts against corresponding media files."""
    files = glob(str(Path(in_dir)/ "*.cha"))

    def loader(file):
        return CHATFile(path=os.path.abspath(file)).doc

    def writer(doc, output):
        CHATFile(doc=doc).write(output)

    _dispatch("align", files, ctx,
              in_dir, out_dir,
              loader, writer, C, **kwargs)

#################### VERSION ################################

@batchalign.command()
@click.pass_context
def version(ctx, **kwargs):
    """Print program version info and exit."""

    ptr = (pyfiglet.figlet_format("Batchalign2")+"\n" +
           f"Version: [bold]{VERSION_NUMBER}[/bold], released {RELEASE_DATE}\n" +
           f"[italic]{RELEASE_NOTES}[/italic]"+"\n" +
           "\nDeveloped by Brian MacWhinney and Houjun Liu")
    C.print("\n\n"+ptr+"\n\n")
