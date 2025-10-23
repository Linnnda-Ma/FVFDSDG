import torch
import torch.nn as nn
import numpy as np
from transformer.Layers import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    def __init__(self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super(Encoder, self).__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):
        enc_slf_attn_list = []

        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class TransformerImageMix(nn.Module):
    def __init__(self, n_layers, n_head, d_k, d_v, d_model, dropout=0.1):
        super(TransformerImageMix, self).__init__()

        # Define the Transformer encoder for image mixing
        self.encoder = Encoder(
            n_src_vocab=512,  # Can be the number of channels or some predefined value
            d_word_vec=d_model, 
            n_layers=n_layers, 
            n_head=n_head, 
            d_k=d_k, 
            d_v=d_v, 
            d_model=d_model, 
            d_inner=2048, 
            pad_idx=0, 
            dropout=dropout
        )

    def forward(self, image1, image2):
        """Mix two images using Transformer-based self-attention."""
        *_, c, h, w = image1.shape

        # Flatten the image to treat each pixel as a token (shape: batch_size, c, h * w)
        image1_tokens = image1.view(image1.size(0), c, -1).permute(0, 2, 1)  # (batch_size, h * w, c)
        image2_tokens = image2.view(image2.size(0), c, -1).permute(0, 2, 1)  # (batch_size, h * w, c)

        # Create masks for padding (can be adjusted based on input)
        pad_mask = torch.ones(image1_tokens.size(0), image1_tokens.size(1), dtype=torch.bool, device=image1.device)

        # Pass through the transformer encoder (same encoder for both images)
        enc_output1, _ = self.encoder(image1_tokens, pad_mask)
        enc_output2, _ = self.encoder(image2_tokens, pad_mask)

        # Combine the outputs using some form of attention or mixing strategy
        mixed_output = (enc_output1 + enc_output2) / 2  # Simple mix, could use more complex strategies

        # Reshape back to image format (batch_size, c, h, w)
        mixed_image = mixed_output.permute(0, 2, 1).view(image1.size(0), c, h, w)

        return mixed_image

