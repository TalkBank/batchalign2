import torch
from transformers.models.whisper.generation_whisper import _dynamic_time_warping as _dynamic_time_warping
from transformers.models.whisper.generation_whisper import _median_filter as _median_filter

from dataclasses import dataclass
import numpy as np

def _extract_token_timestamps(
        self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None, num_input_ids=None
    ):
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

        weight_length = None

        if "beam_indices" in generate_outputs:
            # If beam search has been used, the output sequences may have been generated for more timesteps than their sequence_lengths
            # since the beam search strategy chooses the most probable sequences at the end of the search.
            # In that case, the cross_attentions weights are too long and we have to make sure that they have the right output_length
            weight_length = (generate_outputs.beam_indices != -1).sum(-1).max()
            weight_length = weight_length if num_input_ids is None else weight_length + num_input_ids

            # beam search takes `decoder_input_ids` into account in the `beam_indices` length
            # but forgot to shift the beam_indices by the number of `decoder_input_ids`
            beam_indices = torch.zeros_like(generate_outputs.beam_indices[:, :weight_length], dtype=torch.float32)
            # we actually shif the beam indices here
            beam_indices[:, num_input_ids:] = generate_outputs.beam_indices[:, : weight_length - num_input_ids]

            weights = weights[:, :, :weight_length]

            # If beam index is still -1, it means that the associated token id is EOS
            # We need to replace the index with 0 since index_select gives an error if any of the indexes is -1.
            beam_indices = beam_indices.masked_fill(beam_indices == -1, 0)

            # Select the cross attention from the right beam for each output sequences
            weights = torch.stack(
                [
                    torch.index_select(weights[:, :, i, :], dim=0, index=beam_indices[:, i])
                    for i in range(beam_indices.shape[1])
                ],
                dim=2,
            )

        # make sure timestamps are as long as weights
        input_length = weight_length or cross_attentions[0].shape[2]
        batch_size = generate_outputs.sequences.shape[0]
        timestamps = torch.zeros(
            (batch_size, input_length + 1), dtype=torch.float32, device=generate_outputs.sequences.device
        )

        if num_frames is not None:
            # two cases:
            # 1. num_frames is the same for each sample -> compute the DTW matrix for each sample in parallel
            # 2. num_frames is different, compute the DTW matrix for each sample sequentially

            # we're using np.unique because num_frames can be int/list/tuple
            if isinstance(num_frames, int):
                weights = weights[..., : num_frames // 2]

            elif isinstance(num_frames, (list, tuple, np.ndarray)) and len(np.unique(num_frames)) == 1:
                weights = weights[..., : num_frames[0] // 2]

            elif isinstance(num_frames, (torch.Tensor)) and len(torch.unique(num_frames)) == 1:
                weights = weights[..., : num_frames[0] // 2]

            else:
                # num_frames is of shape (batch_size,) whereas batch_size is truely batch_size*num_return_sequences
                repeat_time = batch_size if isinstance(num_frames, int) else batch_size // len(num_frames)
                num_frames = num_frames.cpu() if isinstance(num_frames, (torch.Tensor)) else num_frames
                num_frames = np.repeat(num_frames, repeat_time)

        if num_frames is None or isinstance(num_frames, int):
            # Normalize and smoothen the weights.
            std = torch.std(weights, dim=-2, keepdim=True, unbiased=False)
            mean = torch.mean(weights, dim=-2, keepdim=True)
            weights = (weights - mean) / std
            weights = _median_filter(weights, self.config.median_filter_width)

            # Average the different cross-attention heads.
            weights = weights.mean(dim=1)

        # Perform dynamic time warping on each element of the batch.
        for batch_idx in range(batch_size):
            if num_frames is not None and isinstance(num_frames, (tuple, list, np.ndarray, torch.Tensor)):
                matrix = weights[batch_idx, ..., : num_frames[batch_idx] // 2]

                # Normalize and smoothen the weights.
                std = torch.std(matrix, dim=-2, keepdim=True, unbiased=False)
                mean = torch.mean(matrix, dim=-2, keepdim=True)
                matrix = (matrix - mean) / std
                matrix = _median_filter(matrix, self.config.median_filter_width)

                # Average the different cross-attention heads.
                matrix = matrix.mean(dim=0)
            else:
                matrix = weights[batch_idx]

            text_indices, time_indices = _dynamic_time_warping(-matrix.cpu().double().numpy())
            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
            jump_times = time_indices[jumps] * time_precision
            timestamps[batch_idx, 1:] = torch.tensor(jump_times)

        return timestamps

# def _extract_token_timestamps(self, generate_outputs, alignment_heads, time_precision=0.02, num_frames=None):
#     """
#     Calculates token-level timestamps using the encoder-decoder cross-attentions and dynamic time-warping (DTW) to
#     map each output token to a position in the input audio. If `num_frames` is specified, the encoder-decoder
#     cross-attentions will be cropped before applying DTW.

#     Returns:
#         tensor containing the timestamps in seconds for each predicted token
#     """
#     # Create a list with `decoder_layers` elements, each a tensor of shape
#     # (batch size, attention_heads, output length, input length).
#     cross_attentions = []
#     for i in range(self.config.decoder_layers):
#         cross_attentions.append(torch.cat([x[i] for x in generate_outputs.cross_attentions], dim=2))

#     # Select specific cross-attention layers and heads. This is a tensor
#     # of shape (batch size, num selected, output length, input length).
#     weights = torch.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
#     weights = weights.permute([1, 0, 2, 3])
#     if num_frames is not None:
#         weights = weights[..., : num_frames // 2]

#     # Normalize and smoothen the weights.
#     std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
#     weights = (weights - mean) / std
#     weights = _median_filter(weights, self.config.median_filter_width)

#     # Average the different cross-attention heads.
#     matrix = weights.mean(dim=1)

#     timestamps = torch.zeros_like(generate_outputs.sequences, dtype=torch.float32)

#     # Perform dynamic time warping on each element of the batch.
#     for batch_idx in range(timestamps.shape[0]):
#         text_indices, time_indices = _dynamic_time_warping(-matrix[batch_idx].float().cpu().numpy())
#         jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
#         jump_times = time_indices[jumps] * time_precision
#         timestamps[batch_idx, 1:] = torch.tensor(jump_times)

#     return timestamps


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

