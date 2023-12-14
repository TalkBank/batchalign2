from transformers import WhisperProcessor, WhisperTokenizer, WhisperForConditionalGeneration

import torch
from torchaudio import load
from torchaudio import transforms as T
from scipy.ndimage import median_filter
from transformers.models.whisper.modeling_whisper import _dynamic_time_warping as dtw
from transformers.models.whisper.modeling_whisper import _median_filter as median_filter

import numpy as np

base = "openai/whisper-large-v2"

transcript = "Hello . This is a test of Batchalign . I'm some body going to read wants told some random crap as I see on the screen . just to test batchline . the primary area for recording editing and me the world's gonna roll me arranging audio. MIDI and drummer regions divided into different track types . Press command slash for more info . test test . I don't know what to say . but um here's some retracing . so just in this for fun . um I like I like I like beans . beans are fun . thank you very much ."

audio = "./batchalign/tests/support/test.mp3"

# sample rate
sample_rate = 16000

processor = WhisperProcessor.from_pretrained(base)
model = WhisperForConditionalGeneration.from_pretrained(base)
time_precision = 0.02

###### 

# function: load and resample audio
audio_arr, rate = load(audio)
duration = audio_arr.shape[-1]/rate

# resample if needed
if rate != sample_rate:
    audio_arr = T.Resample(rate, sample_rate)(audio_arr)

# transpose and mean
resampled = torch.mean(audio_arr.transpose(0,1), dim=1)

# input features
features = processor(audio=resampled, text=transcript,
                     sampling_rate=sample_rate, return_tensors='pt')
tokens = features["labels"][0]

# perform inference to get cached qs
with torch.inference_mode():
    output = model(**features, output_attentions=True)

# get decoder layer across attentions
# which has shape layers x heads x output_tokens x input_frames
cross_attentions = torch.cat(output.cross_attentions)

# get the attention of alignment heads we care about only
weights = torch.stack([cross_attentions[l][h]
                       for l, h in model.generation_config.alignment_heads])

# normalize the attentino activations
std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
weights = (weights - mean) / std

# perform smoothing on attention activations + scale them
weights = median_filter(weights, model.config.median_filter_width)
matrix = weights.mean(axis=0)

# its dynamic time warping time
text_idx, time_idx = dtw(-matrix)
jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
jump_times = time_idx[jumps] * time_precision
timestamped_tokens = [(processor.decode(i),j) for i,j in zip(tokens, jump_times)]
timestamped_tokens




# [i.remove() for i in handles]

# QKs

# weights = torch.cat(QKs)
# [i for i in QKs if i == None]

# # QKs[0]

# # model = Whi
