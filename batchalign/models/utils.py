import torch
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as _dynamic_time_warping
from transformers.models.whisper.generation_whisper import _median_filter as _median_filter


from transformers import WhisperForConditionalGeneration

import numpy as np

import logging
L = logging.getLogger("batchalign")

def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
    """
    Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
    map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
    cross-attentions will be cropped before applying DTW.

    Returns:
        tensor containing the timestamps in seconds for each predicted token
    """
    L.debug("Collecting and normalizing activations...")

    # Create a list with `decoder_layers` elements, each a tensor of shape
    # (batch size, attention_heads, output length, input length).
    cross_attentions = []
    for i in range(self.config.decoder_layers):
        cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

    # Select specific cross-attention layers and heads. This is a tensor
    # of shape (batch size, num selected, output length, input length).
    weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    if num_frames is not None:
        weights = weights[..., : num_frames // 2]

    # Normalize and smoothen the weights.
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = _median_filter(weights, self.config.median_filter_width)

    # Average the different cross-attention heads.
    matrix = weights.mean(dim=1)

    timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)

    # Perform dynamic time warping on each element of the batch.
    for batch_idx in range(timestamps.shape[0]):
        text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].float().cpu().numpy())
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        timestamps[batch_idx, 1:] = torch.tensor(jump_times)

    return timestamps
WhisperForConditionalGeneration._extract_token_timestamps = _extract_token_timestamps

def attn_dynamic_timewarp(output, alignment_heads, median_filter_width):
    """Apply the Dynamic Timewraping Algorithm on Whisper Attentions

    Parameters
    ----------
    output : dict
        Whisper inference output.
    alignment_heads : list
        Whisper model's specific heads used for alignment.
    median_filter_width : float?
        Width of the median filter used .

    Returns
    -------
    List[float]
        Timestamped tokens.
    """
    
    L.debug("Collecting and normalizing activations...")
    # get decoder layer across attentions
    # which has shape layers x heads x output_tokens x input_frames
    cross_attentions = torch.cat(output.cross_attentions).cpu()

    # get the attention of alignment heads we care about only
    weights = torch.stack([cross_attentions[l][h]
                           for l, h in alignment_heads])

    # normalize the attentino activations
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std

    L.debug("Applying median filter...")

    # perform smoothing on attention activations + scale them
    weights = median_filter(weights, median_filter_width)
    # average weights across heads
    matrix = weights.mean(axis=0)
    matrix[0] = matrix.mean() # Jack's jank way of fixing weird 0th padding token output
    # see: https://media.discordapp.net/attachments/870073176380563460/1185486042753679390/image.png?ex=658fc8e9&is=657d53e9&hm=28ba60b6035fd8976e44f3e53628558d59d5f578917d36ad22c3d63b5563582e&=&format=webp&quality=lossless&width=1050&height=1164
    # essentially, the 0th token (<sos>) gets attention jammed across the entire rest of the sequence
    # because its padding; which screws everything else up

    L.debug("Applying dynamic time warping...")

    # its dynamic time warping time
    text_idx, time_idx = dtw(-matrix)
    jumps = np.pad(np.diff(text_idx), (1, 0), constant_values=1).astype(bool)
    jump_times = time_idx[jumps] * 0.02

    return jump_times
