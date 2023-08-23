import torch
import torchviz
from torch import nn
import json
import math


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

        #出力サイズ = (入力サイズ - フィルターサイズ + 2 × パディング) ÷ ストライド + 1
        self.d_in1 = (image_size - 3 + 2 * 0) // 1 + 1
        self.d_in2 = (2*image_size - 3 + 2 * 0) // 1 + 1
        self.conv_channel = 3

        self.in1 = nn.Sequential(
            nn.Conv2d(1, self.conv_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.conv_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(self.conv_channel * self.d_in1 * self.d_in1, image_size*image_size),
            nn.ReLU(),
        )

        self.in2 = nn.Sequential(
            nn.Conv2d(1, self.conv_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.conv_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(self.conv_channel * self.d_in2 * self.d_in2, 2*image_size*2*image_size),
            nn.ReLU(),
        )

        self.ff_input_size = (image_size * image_size) + (2*image_size * 2*image_size)
        self.seq = nn.Sequential(
            nn.Linear(self.ff_input_size, d_model * (d_word)),
            nn.ReLU(),
        )

        self.layer_norm2 = nn.LayerNorm([d_model, d_word])

    def forward(self, x, image, canvas, word_num):
        yi1 = self.in1(canvas)
        yi2 = self.in2(image)

        yi1 = yi1.view(-1)
        yi2 = yi2.view(-1)

        seq_input = torch.cat((yi1, yi2))
        seq_input = seq_input.unsqueeze(0)
        ff_y = self.seq(seq_input).view(1, self.d_model, self.d_word)
        y = ff_y.view(1, self.d_model, self.d_word)
        return y


class Network(nn.Module):
    def __init__(self, param, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(param["d_dict"], param["d_word"] - 1)
        self.pos_encoder = AddressEncoding(max_len = param["stroke_max"],
                                              device = device)

        self.main_unit1 = MainUnit(
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

    def forward(self, x, image, canvas, word_num):
        # Input network
        y = self.embedding(x)
        y = self.pos_encoder(y)

        # MainUnit
        y = self.main_unit1(y, image, canvas, word_num)
        y = y.view(-1)

        y = self.softmax(self.seq(y))

        y[0] = 0.0
        return y / torch.sum(y)
