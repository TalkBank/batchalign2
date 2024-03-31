import torch
import torch.nn.functional as F

def conv_scale_weights(self, ms_avg_embs_perm, ms_emb_seq_single):
    """
    Use multiple Convnet layers to estimate the scale weights based on the cluster-average embedding and
    input embedding sequence.

    Args:
        ms_avg_embs_perm (Tensor):
            Tensor containing cluster-average speaker embeddings for each scale.
            Shape: (batch_size, length, scale_n, emb_dim)
        ms_emb_seq_single (Tensor):
            Tensor containing multi-scale speaker embedding sequences. ms_emb_seq_single is input from the
            given audio stream input.
            Shape: (batch_size, length, num_spks, emb_dim)

    Returns:
        scale_weights (Tensor):
            Weight vectors that determine the weight of each scale.
            Shape: (batch_size, length, num_spks, emb_dim)
    """
    ms_cnn_input_seq = torch.cat([ms_avg_embs_perm, ms_emb_seq_single], dim=2)
    ms_cnn_input_seq = ms_cnn_input_seq.unsqueeze(2).flatten(0, 1)

    conv_out = self.conv_forward(
        ms_cnn_input_seq, conv_module=self.conv[0], bn_module=self.conv_bn[0], first_layer=True
    )
    for conv_idx in range(1, self.conv_repeat + 1):
        conv_out = self.conv_forward(
            conv_input=conv_out,
            conv_module=self.conv[conv_idx],
            bn_module=self.conv_bn[conv_idx],
            first_layer=False,
        )

    # reshape / view
    lin_input_seq = conv_out.reshape(self.batch_size, self.length, self.cnn_output_ch * self.emb_dim)
    hidden_seq = self.conv_to_linear(lin_input_seq)
    hidden_seq = self.dropout(F.leaky_relu(hidden_seq))
    scale_weights = self.softmax(self.linear_to_weights(hidden_seq))
    scale_weights = scale_weights.unsqueeze(3).expand(-1, -1, -1, self.num_spks)
    return scale_weights

