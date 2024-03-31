# system utilities
import re
import string
import random

# tokenization utilitise
from batchalign.utils.utils import word_tokenize

# torch
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

# import huggingface utils
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification

# import our dataset
from batchalign.models.utterance.dataset import TOKENS, UtteranceBoundaryDataset

# tqdm
from tqdm import tqdm

# training utilities
from batchalign.models.training import *

import logging
L = logging.getLogger("batchalign")


# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def train(c):
    # get in and out dir
    model_dir, data_dir, run_name = c.resolve_data()

    if (os.path.exists(os.path.join(model_dir, run_name))):
        L.info(f"Path {os.path.join(model_dir, run_name)} exists, skipping training...")
        return 

    if c.tracker:
        import wandb

        # start wandb
        run = wandb.init(project="batchalign", name=c.tracker.run_name, entity=c.tracker.user, config=c.params)

        # set configuration
        config = dict(run.config)
    else:
        config = c.params

    # create the tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(config["bert_base"])

    # load the data (train using MICASE, test on Pitt)
    train_data = UtteranceBoundaryDataset(os.path.join(data_dir,  f"{run_name}.train.txt"),
                                          tokenizer, window=config["window"], min_length=config["min_length"])
    test_data = UtteranceBoundaryDataset(os.path.join(data_dir,  f"{run_name}.val.txt"),
                                         tokenizer, window=config["window"], min_length=config["min_length"])

    # create data collator utility on the tokenizer
    data_collator = DataCollatorForTokenClassification(tokenizer, return_tensors='pt')

    # load the data
    train_dataloader = DataLoader(train_data,
                                batch_size=config["batch_size"],
                                shuffle=True, collate_fn=lambda x:x)
    test_dataloader = DataLoader(test_data,
                                batch_size=config["batch_size"],
                                shuffle=True, collate_fn=lambda x:x)

    # create the model and tokenizer
    model = BertForTokenClassification.from_pretrained(config["bert_base"],
                                                       num_labels=len(TOKENS)).to(DEVICE)
    optim = AdamW(model.parameters(), lr=config["lr"])

    # utility to move a whole dictionary to a device
    def move_dict(d, device):
        """move a dictionary to device

        Attributes:
            d (dict): dictionary to move
            device (torch.Device): device to move to
        """

        for key, value in d.items():
            d[key] = d[key].to(device)

    # start training!
    val_data = list(iter(test_dataloader))

    # watch the model
    if c.tracker:
        run.watch(model)

    # for each epoch
    for epoch in range(config["epochs"]):
        print(f"Training epoch {epoch}")

        # for each batch
        for indx, batch in tqdm(enumerate(iter(train_dataloader)), total=len(train_dataloader)):
            # pad and conform batch
            batch = data_collator(batch)
            move_dict(batch, DEVICE)

            # train!
            output = model(**batch)
            # backprop
            output.loss.backward()
            # step
            optim.step()
            optim.zero_grad()

            # log!
            if c.tracker:
                run.log({
                    'loss': output.loss.cpu().item()
                })

            # if need to validate, validate
            if indx % 10 == 0:
                # select a val batch
                val_batch = data_collator(random.choice(val_data))
                move_dict(val_batch, DEVICE)
                # run!
                output = model(**val_batch)
                # log!
                if c.tracker:
                    run.log({
                        'val_loss': output.loss.cpu().item()
                    })

    # write model down
    model.save_pretrained(os.path.join(model_dir, run_name))
    tokenizer.save_pretrained(os.path.join(model_dir, run_name))

