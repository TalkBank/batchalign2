import torch
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as _dynamic_time_warping
from transformers.models.whisper.generation_whisper import _median_filter as _median_filter

import numpy as np

def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
    """
    Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
    map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
    cross-attentions will be cropped before applying DTW.

    Returns:
        tensor containing the timestamps in seconds for each predicted token
    """
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

