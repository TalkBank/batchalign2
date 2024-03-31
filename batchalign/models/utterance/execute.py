from batchalign.models.training.utils import *
from batchalign.models.utterance.prep import prep as P
from batchalign.models.utterance.train import train as T

import rich_click as click

# build bert utterance model
@click.group()
def utterance():
    """Utterance segmentation model."""
    pass

@utterance.command()
@train_func_hydrate
@click.option("--lr", type=float, default=3.5e-5, help="Learning Rate", show_default=True)
@click.option("--batch_size", type=int, default=5, help="Batch Size", show_default=True)
@click.option("--epochs", type=int, default=2, help="Number of Epochs", show_default=True)
@click.option("--window", type=int, default=20, help="Size of the Utterance Merge Window", show_default=True)
@click.option("--min_length", type=int, default=10, help="Minimum length of utterance to include", show_default=True)
@click.option("--bert", type=str, default="bert-base-uncased", help="Bert model to start with", show_default=True)
def train(**kwargs):
    """Train utterance segmentation model."""
    config = create_config(P, T, None,
                           "utterance", {
        "lr": kwargs["lr"],
        "batch_size": kwargs["batch_size"],
        "epochs": kwargs["epochs"],
        "window": kwargs["window"],
        "bert_base": kwargs["bert"],
        "min_length": kwargs["min_length"],
    }, **kwargs)

    config.project.prep(config)
    config.project.train(config)
