import torch
import torchviz
from torch import nn
import json
import math


#class PositionalEncoding(nn.Module):
#    def __init__(self, d_model, max_len=5000, device='cpu'):
#        super().__init__()
#        self.dropout = torch.nn.Dropout(p=0.1)
#        self.d_model = d_model
#        self.device = device
#
#        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
#        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / d_model))
#        pe = torch.zeros(max_len, d_model, device=device)
#        pe[:, 0::2] = torch.sin(position * div_term)
#        pe[:, 1::2] = torch.cos(position * div_term)
#        self.pe = pe.unsqueeze(0).to(device)
#
#    def forward(self, x):
#        return x + self.pe[:, :x.size(1)].to(self.device)

class AddressEncoding(nn.Module):
    def __init__(self, max_len=5000, device='cpu'):
        super(AddressEncoding, self).__init__()
        self.max_len = max_len
        self.device = device

        self.positions = nn.Parameter(torch.arange(max_len).long().unsqueeze(1), requires_grad=False)

    def forward(self, x):
        x = x.unsqueeze(0)

        size = x.size()
        positions = self.positions[:size[1], 0].unsqueeze(0)
        positions = positions.float() / (self.max_len - 1)
        positions = positions.unsqueeze(-1)

        ret = torch.cat([x, positions.to(self.device)], dim=-1)
        return ret

class MainUnit(nn.Module):
    def __init__(self, d_model, d_word, num_heads, image_size, hidden_size, dropout=0.1, device='cpu'):
        super().__init__()
        self.d_model = d_model
        self.d_word = d_word
        self.device = device

        self.mh_attention = nn.MultiheadAttention(d_word, 4, dropout=dropout)
        self.layer_norm = nn.LayerNorm([d_model, d_word])

        self.ff_input_size = (image_size * image_size) + (d_model * (d_word))
        self.seq = nn.Sequential(
            nn.Linear(self.ff_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model * (d_word)),
            nn.ReLU()
        )

        self.layer_norm2 = nn.LayerNorm([d_model, d_word])

    def forward(self, x, image, word_num):
        # shape(1, 256, 128) = (1, stroke size, word_size)
        mask = torch.zeros((x.shape[1], 1)).bool()
        mask[word_num:, :] = True
        mask = mask.float().to(self.device)
        x = x.to(self.device)

        y, _ = self.mh_attention(x, x, x, key_padding_mask=mask)
        y = y + x

        layer_norm_y = self.layer_norm(y)

        seq_input = torch.cat((image, layer_norm_y.view(-1)))
        seq_input = seq_input.unsqueeze(0)

        ff_y = self.seq(seq_input).view(1, self.d_model, self.d_word)

        return self.layer_norm2(layer_norm_y + ff_y)


class Network(nn.Module):
    def __init__(self, param, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(param["d_dict"], param["d_word"] - 1)
        #self.pos_encoder = PositionalEncoding(param["d_model"],
        #                                      max_len = param["stroke_max"],
        #                                      device = device)
        self.pos_encoder = AddressEncoding(max_len = param["stroke_max"],
                                              device = device)
        self.main_unit = MainUnit(
                param["d_model"],
                param["d_word"],
                num_heads = param["main_layer"]["num_head"],
                image_size = param["d_image"],
                hidden_size = param["main_layer"]["d_hidden"],
                device = device)

        input_size = param["d_model"] * (param["d_word"])
        output_size = param["d_dict"]
        self.seq = nn.Sequential(
            nn.Linear(input_size, param["output_layer"]["d_hidden"]),
            nn.ReLU(),
            nn.Linear(param["output_layer"]["d_hidden"], output_size),
            nn.ReLU()
        )
        # Output network
        self.softmax = nn.Softmax()

    def forward(self, x, image, word_num):
        # Input network
        y = self.embedding(x)
        y = self.pos_encoder(y)

        # MainUnit
        y = self.main_unit(y, image, word_num).view(-1)

        y = self.softmax(self.seq(y))

        y[0] = 0.0
        return y / torch.sum(y)
