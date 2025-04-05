import re
import string
import random

# tokenization utilities
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

import logging
L = logging.getLogger("batchalign")

# seed device and tokens
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# seed model
class BertCantoneseUtteranceModel(object):

    def __init__(self, model):
        # seed tokenizers and model
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = BertForTokenClassification.from_pretrained(model).to(DEVICE)
        self.max_length = 512
        self.overlap = 20

        # eval mode
        self.model.eval()
        L.debug(f"Model and tokenizer initialized on device: {DEVICE}")
        L.debug(f"Max length set to {self.max_length} with overlap of {self.overlap}")

    def __call__(self, passage):
        # Step 1: Clean up passage
        passage = passage.lower()
        passage = passage.replace('.','')
        passage = passage.replace(',','')
        passage = passage.replace('!','')
        passage = passage.replace('！','')
        passage = passage.replace('？','')
        passage = passage.replace('。','')
        passage = passage.replace('，','')
        passage = passage.replace('?','')
        passage = passage.replace('（','')
        passage = passage.replace('）','')
        passage = passage.replace('：','')
        passage = passage.replace('＊','')
        passage = passage.replace('ｌ','')
        
        
        # Step 2: Define keywords and split the passage based on them
        keywords = ['呀', '啦', '喎', '嘞', '㗎喇', '囉', '㗎', '啊', '嗯']  # Replace with your desired keywords
        
        chunks = []
        start = 0

        while start < len(passage):
            # Find the position of each keyword in the passage starting from the current `start`
            keyword_positions = [(keyword, passage.find(keyword, start)) for keyword in keywords]
            # Filter out keywords that are not found (find() returns -1 if not found)
            keyword_positions = [kp for kp in keyword_positions if kp[1] != -1]
            
            if keyword_positions:
                # Find the keyword that appears first in the passage from current start
                first_keyword, keyword_pos = min(keyword_positions, key=lambda x: x[1])
                chunk = passage[start:keyword_pos + len(first_keyword)]
                chunks.append(chunk)
                start = keyword_pos + len(first_keyword)
            else:
                # No more keywords found, add the rest of the passage as the last chunk
                chunks.append(passage[start:])
                break
        
        # Debugging: Print number of chunks and their content
        L.debug(f"Created {len(chunks)} chunks based on keywords.")
        for i, chunk in enumerate(chunks):
            L.debug(f"Chunk {i + 1}: {chunk[:100]}...")  # Print the first 100 characters of each chunk
        
        # Step 3: Process each chunk and restore punctuation
        final_passage = []
        for chunk_index, chunk in enumerate(chunks):
            L.debug(f"Processing chunk {chunk_index + 1}/{len(chunks)}...")

            # Step 3.1: Split chunk by characters (Chinese tokenization)
            tokenized_chunk = list(chunk)  # Simply split by characters for Chinese text
            
            # Step 3.2: Pass chunk through the tokenizer and model
            tokd = self.tokenizer.batch_encode_plus([tokenized_chunk],
                                  return_tensors='pt',
                                  truncation=True,
                                  padding=True,
                                  max_length=self.max_length,
                                  is_split_into_words=True).to(DEVICE)

            try:
                # Pass it through the model
                res = self.model(**tokd).logits
            except Exception as e:
                L.error(f"Error during model inference: {e}")
                return []

            # Argmax for classification
            classified_targets = torch.argmax(res, dim=2).cpu()

            # Initialize result tokens list for the current chunk
            res_toks = []
            prev_word_idx = None

            # Iterate over tokenized words
            wids = tokd.word_ids(0)
            for indx, elem in enumerate(wids):
                if elem is None or elem == prev_word_idx:
                    continue

                prev_word_idx = elem
                action = classified_targets[0][indx]

                # Get the word corresponding to the token
                w = tokenized_chunk[elem]  # Use tokenized chunk here

                # Fix one word hanging issue (if needed)
                will_action = False
                if indx < len(wids) - 2 and classified_targets[0][indx + 1] > 0:
                    will_action = True

                if not will_action:
                    # Perform the edits based on model predictions
                    if action == 1:  # First capital letter
                        w = w[0].upper() + w[1:]
                    elif action == 2:  # Add period
                        w = w + '.'
                    elif action == 3:  # Add question mark
                        w = w + '?'
                    elif action == 4:  # Add exclamation mark
                        w = w + '!'
                    elif action == 5:  # Add comma
                        w = w + ','

                # Append modified word to result list
                res_toks.append(w)

            # Convert list of tokens back to string and append to final_passage
            final_passage.append(self.tokenizer.convert_tokens_to_string(res_toks))

        # Step 4: Join processed chunks together into the final passage
        final_passage = ' '.join(final_passage)

        L.info("Text processing completed. Generating final output...")
        
        # Optionally, tokenize the final text into sentences based on punctuation
        def custom_sent_tokenize(text):
            # Regular expression to match sentence-ending punctuation marks (. ! ?)
            sentence_endings = re.compile(r'([.!?])')  
            split_passage = []
        
            # Split the passage based on punctuation marks and keep them
            parts = re.split(sentence_endings, text) 
            
            # Debug: Output the parts after splitting
            L.debug(f"Parts after splitting: {parts}")
        
            # Combine parts and punctuation together
            for i in range(0, len(parts) - 1, 2):
                sentence = parts[i] + parts[i + 1]  # Join sentence with punctuation
                L.debug(f"Sentence formed: {sentence}")  # Debug: Output the current sentence
                
                if sentence.strip():  # Only add non-empty sentences (check for non-whitespace content)
                    split_passage.append(sentence)
        
            # If the last part doesn't have punctuation, we handle it here
            if len(parts) % 2 != 0:  # If there's no punctuation at the end
                last_part = parts[-1].strip()
                L.debug(f"Last part without punctuation: {last_part}")  # Debug: Output the last part
                
                if last_part:  # Only add non-empty sentences
                    split_passage.append(last_part)
        
            # Final output
            L.debug(f"Final split passage: {split_passage}")
            return split_passage
            
        split_passage = custom_sent_tokenize(final_passage)

        # Debugging: Output the sentences after splitting
        L.debug(f"Final sentences: {split_passage}")

        return split_passage
