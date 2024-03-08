from batchalign.models.training.utils import *
from batchalign.models.bert_utterance.prep import prep as P
from batchalign.models.bert_utterance.train import train as T

import rich_click as click

# weights and biases
hyperparametre_defaults = dict(
    learning_rate = 3.5e-5,
    batch_size = 5,
    epochs = 2,
    window = 20 
)

# build bert utterance model
@click.group()
def utterance():
    """Utterance segmentation model."""
    pass

@utterance.command()
@train_func_hydrate
@click.option("--lr", type=float, default=3.5e-5, help="Learning Rate")
@click.option("--batch_size", type=int, default=5, help="Batch Size")
@click.option("--epochs", type=int, default=2, help="Number of Epochs")
@click.option("--window", type=int, default=20, help="Size of the Utterance Merge Window")
@click.option("--bert_base", type=str, default="bert-base-uncased", help="Bert model to start with")
def train(**kwargs):
    """Train utterance segmentation model."""
    config = create_config(P, T, None,
                           "utterance", {
        "lr": kwargs["lr"],
        "batch_size": kwargs["batch_size"],
        "epochs": kwargs["epochs"],
        "window": kwargs["window"],
        "bert_base": kwargs["bert_base"],
    }, **kwargs)

    config.project.prep(config)
    config.project.train(config)
