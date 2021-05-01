import copy
import torch
import torch.nn as nn


# num queries : 예측할 object 갯수.

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.d_model = d_model

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

    def forward(self, src, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, layer_num, norm=None):
        super().__init__()
        self.layers = clone_layer(encoder_layer, layer_num)
        self.norm = norm

    def forward(self, src, pos=None):
        out = src
        for layer in self.layer:
            out = layer(out, pos)

        if self.norm:
            out = self.norm(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, layer_num, norm=None):
        super().__init__()
        self.layers = clone_layer(decoder_layer, layer_num)
        self.norm = norm

    def forward(self, tgt, memory, pos=None, query_pos=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, pos, query_pos)
        if self.norm:
            out = self.norm(out)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos=None):
        q = k = self.pos_add(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

    def pos_add(self, src, pos):
        return src if pos == None else src + pos


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.mh_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, pos=None, query_pos=None):
        q = k = self.pos_add(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.mh_attn(self.pos_add(tgt, query_pos), self.pos_add(memory, pos), value=memory)[0]
        tgt = tgt + self.dropout2(tgt)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def pos_add(self, tensor, pos):
        return tensor if pos == None else tensor + pos


def clone_layer(layer, layer_num):
    return [copy.deepcopy(layer) for _ in range(layer_num)]