# system utilities
import re
import string
import random

# tokenization utilitise
import nltk
from nltk import word_tokenize, sent_tokenize

# torch
import torch
from torch.utils.data import dataset 
from torch.utils.data.dataloader import DataLoader
from torch.optim import AdamW

# import huggingface utils
from transformers import AutoTokenizer, BertForTokenClassification
from transformers import DataCollatorForTokenClassification

# tqdm
from tqdm import tqdm

# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# seed model
class BertUtteranceModel(object):

    def __init__(self, model):
        # seed tokenizers and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = BertForTokenClassification.from_pretrained(model).to(DEVICE)

        # eval mode
        self.model.eval()

    def __call__(self, passage):
        # print(passage)
        # input passage words removed of all preexisting punctuation
        passage = passage.lower()
        passage = passage.replace('.','')
        passage = passage.replace(',','')
        passage = passage.replace('.','')

        # "tokenize" the result by just splitting by space
        input_tokenized = passage.split(' ')

        # pass it through the tokenizer and model
        tokd = self.tokenizer([input_tokenized],
                              return_tensors='pt',
                              is_split_into_words=True).to(DEVICE)

        # pass it through the model
        res = self.model(**tokd).logits

        # argmax
        classified_targets = torch.argmax(res, dim=2).cpu()

        # 0, normal word
        # 1, first capital
        # 2, period
        # 3, question mark
        # 4, exclaimation mark
        # 5, comma

        # and finally append the result
        res_toks = []
        prev_word_idx = None

        # for each word, perform the action
        wids = tokd.word_ids(0)
        for indx, elem in enumerate(wids):
            # if its none, append nothing or if we have
            # seen it before, do nothing
            if elem is None or elem == prev_word_idx:
                continue
            # otherwise, store previous index
            prev_word_idx = elem

            # otherwise, get nth prediction
            action = classified_targets[0][indx]

            # set the working variable
            w = input_tokenized[elem]

            # fix one word hanging issue
            will_action = False
            if indx < len(wids)-2 and classified_targets[0][indx+1] > 0:
                will_action = True

            if not will_action:
                # perform the edit actions
                if action == 1:
                    w = w[0].upper() + w[1:]
                elif action == 2:
                    w = w+'.'
                elif action == 3:
                    w = w+'?'
                elif action == 4:
                    w = w+'!'
                elif action == 5:
                    w = w+','


            # append
            res_toks.append(w)

        # compose final passage
        final_passage = self.tokenizer.convert_tokens_to_string(res_toks)
        # print(final_passage)
        try: 
            split_passage = sent_tokenize(final_passage)
        except LookupError:
            # we are missing punkt
            nltk.download('punkt')
            nltk.download('punkt_tab')
            # perform tokenization
            split_passage = sent_tokenize(final_passage)

        return split_passage

