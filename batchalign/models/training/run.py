from batchalign.models.utterance.execute import utterance
import logging as L

import rich_click as click

import sys
from rich import pretty
from rich.traceback import install
from rich.logging import RichHandler
from multiprocessing import Process, freeze_support

@click.group()
def cli():
    """Batchalign model training utilities."""
    freeze_support()
    L.basicConfig(format="%(message)s", level=L.ERROR, handlers=[RichHandler(rich_tracebacks=True)])
    L.getLogger('batchalign').setLevel(L.DEBUG)

cli.add_command(utterance)

if __name__ == "__main__":
    cli()
