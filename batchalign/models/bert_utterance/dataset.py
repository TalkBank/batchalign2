# system utilities
import re
import random

# dataset 
from torch.utils.data import dataset 

# tokenizer models
from transformers import AutoTokenizer, BertForTokenClassification

# general tokens
TOKENS = {
    "U": 0, # normal word
    "OC": 1, # first capital
    "E.": 2, # period
    "E?": 3, # question mark
    "E!": 4, # exclaimation mark
    "E,": 5, # comma
}

# marks for sentence boundaries
BOUNDARIES = [2,3,4]

# create the dataset loading function
class UtteranceBoundaryDataset(dataset.Dataset):
    raw_data: list[str]
    max_length: int
    tokenizer: AutoTokenizer
    window: int

    # initalization function (to read data, etc.)
    # max length doesn't matter
    def __init__(self, f, tokenizer, window=10, max_length=1000, min_length=10):
        # read the file
        with open(f, 'r') as df:
            d =  df.readlines()
        # store the raw data cleaned
        self.raw_data = [i.strip() for i in d]
        # store window size
        self.window = window
        # store max length
        self.max_length = max_length
        self.min_length = min_length
        # store tokenizer
        self.tokenizer = tokenizer

    # clean and conform the sentence
    def __call__(self, passage):
        """prepare passage

        Attributes:
            passage (str): the input passage
        """

        # store tokenizer
        tokenizer = self.tokenizer

        # clean sentence
        sentence_raw = re.sub(r' ?(\W)', r'\1', passage)
        # "tokenize" into words
        sentence_tokenized = sentence_raw.split(' ')

        # generate labels by scanning through words
        labels = []
        # iterate through words for labels
        for word in sentence_tokenized:
            # if the first is capitalized
            if word[0].isupper():
                labels.append(TOKENS["OC"])
            # otherwise, if the last is a punctuation, append
            elif word[-1] in ['.', '?', '!', ',']:
                labels.append(TOKENS[f"E{word[-1]}"])
            # finally, if nothing works, its just regular
            else:
                labels.append(TOKENS["U"])

        # remove symbols and lower
        sentence_tokenized = [re.sub(r'[.?!,]', r'', i) for i in sentence_tokenized]

        # tokenize time!
        tokenized = tokenizer(sentence_tokenized,
                              truncation=True,
                              is_split_into_words=True,
                              max_length=self.max_length)

        # and now, we get result
        # not sure why, but if there is multiple items for each
        # of the tokens, we only calculate error on one and leave
        # the rest as -100

        final_labels = []
        prev_word_idx = None

        # for each tokens
        for elem in tokenized.word_ids(0):
            # if its none, append nothing
            if elem is None:
                final_labels.append(-100)
            # if its not none, append something
            # if its a new index
            elif elem != prev_word_idx:
                # find the label
                final_labels.append(labels[elem])
                # set prev
                prev_word_idx = elem
            # otherwise, append skip again
            else:
                final_labels.append(-100)

        # set labels
        tokenized["labels"] = final_labels

        return tokenized

    # get a certain item
    def __getitem__(self, index):
        # get the raw data shifted by sentence
        sents = self.raw_data[index*self.window:index*self.window+random.randint(1, self.window)]
        # filter for min length
        sents = [i for i in sents if len(i) >= self.min_length]
        if len(sents) == 0:
            return self[index + 1] if index < len(self)-1 else self[index-1]
        # prepare the sentence and return
        return self(" ".join(sents))

    def __len__(self):
        return len(self.raw_data)//self.window

def calculate_acc_prec_rec_f1(preds, labs):
    """Calculates accuracy, precesion, recall, and F1 on sentence boundaries

    Arguments:
        preds (torch.Tensor): the predictions of model
        labs (torch.Tensor): the correct labels, -100 are ignored
    """

    # count tp, fp
    tp = 0
    fp = 0
    fn = 0

    # get true boundariy indicies
    boundaries = labs.clone().apply_(lambda x:x in BOUNDARIES)
    boundaries = (boundaries == 1).nonzero().tolist()

    # get true boundariy indicies
    boundaries_hat = preds.clone().apply_(lambda x:x in BOUNDARIES)
    boundaries_hat = (boundaries_hat == 1).nonzero().tolist()

    # filter hat boundaries to ignore the ignore indicies
    boundaries_hat = list(filter(lambda x: labs[x[0]][x[1]] != -100, boundaries_hat))

    # for each element, see if its a true or false positive
    for elem in boundaries_hat:
        # if its in, its true pos, else its false pos
        if elem in boundaries:
            tp += 1
        else:
            fp += 1
    # and calculate false negatives
    for elem in boundaries:
        # if its not in, its a false negative
        if elem not in boundaries_hat:
            fn += 1

    # count the number of nonzero elements
    count = len((labs!=-100).nonzero())

    # calculate false true negatives
    tn = count-(tp+fp+fn)

    # and now, acc, prec, recc
    acc = (tp+tn)/count
    prec = tp/(tp+fp)
    recc = tp/(tp+fn)

    # and f1
    f1 = 2*((prec*recc)/(prec+recc))

    return acc, prec, recc, f1

