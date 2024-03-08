from batchalign.models.bert_utterance.execute import utterance
import logging as L

import rich_click as click

import sys
from rich import pretty
from rich.traceback import install
from rich.logging import RichHandler

@click.group()
def cli():
    """Batchalign model training utilities."""
    pass

cli.add_command(utterance)

if __name__ == "__main__":
    cli()
