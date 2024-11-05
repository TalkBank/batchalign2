import torch
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as _dynamic_time_warping
from transformers.models.whisper.generation_whisper import _median_filter as _median_filter

from dataclasses import dataclass
import numpy as np

def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None, num_input_ids=None):
    """
    Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
    map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
    cross-attentions will be cropped before applying DTW.

    Returns:
        tensor containing the timestamps in seconds for each predicted token
    """
    # Create a list with `decoder_layers` elements, each a tensor of shape
    # (batch_size, attention_heads, output_length, input_length).
    cross_attentions = []
    for i in range(self.config.decoder_layers):
        cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

    # Select specific cross-attention layers and heads. This results in a tensor
    # of shape (batch_size, num_selected_heads, output_length, input_length).
    weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
    weights = weights.permute([1, 0, 2, 3])
    if num_frames is not None:
        weights = weights[..., : num_frames // 2]

    # Normalize and smoothen the weights.
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    weights = (weights - mean) / std
    weights = _median_filter(weights, self.config.median_filter_width)

    # Average the different cross-attention heads to get a matrix of shape
    # (batch_size, output_length, input_length).
    matrix = weights.mean(dim=1)

    # Initialize the timestamps tensor with the correct size.
    # We'll find the maximum length of `jump_times` across the batch.
    batch_size = generate_outputs.sequences.size(0)
    max_jump_length = 0
    batch_jump_times = []

    # First pass: Compute `jump_times` and find the maximum length.
    for batch_idx in range(batch_size):
        text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].float().cpu().numpy())
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        batch_jump_times.append(jump_times)
        if len(jump_times) > max_jump_length:
            max_jump_length = len(jump_times)

    # Initialize timestamps tensor with appropriate size.
    # Adding 1 to account for the initial zero (timestamps[:, 0]).
    timestamps = torch.zeros((batch_size, max_jump_length + 1), dtype=torch.float32)

    # Second pass: Assign `jump_times` to the timestamps tensor.
    for batch_idx, jump_times in enumerate(batch_jump_times):
        length = len(jump_times)
        # Assign `jump_times` to the appropriate slice in `timestamps`.
        timestamps[batch_idx, 1:1+length] = torch.tensor(jump_times, dtype=torch.float32)

    return timestamps



@dataclass
class ASRAudioFile:
    file : str
    tensor : torch.Tensor
    rate : int

    def chunk(self,begin_ms, end_ms):
        """Get a chunk of the audio.

        Parameters
        ----------
        begin_ms : int
            Milliseconds of the start of the slice.
        end_ms : int
            Milliseconds of the end of the slice.

        Returns
        -------
        torch.Tensor
            The returned chunk to supply to the ASR engine.
        """

        data = self.tensor[int(round((begin_ms/1000)*self.rate)):
                           int(round((end_ms/1000)*self.rate))]

        return data

    def all(self):
        """Get the audio in its entirety

        Notes
        -----
        like `chunk()` but all of the audio
        """

        return self.tensor

