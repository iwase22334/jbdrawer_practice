import torch
import torchviz
from torch import nn
import json
import math
import jbutils as jb;

class MainUnit(nn.Module):
    def __init__(self, param, image_size, device='cpu'):
        super().__init__()

        self.d_out = param["d_out"]
        self.image_size = image_size

        self.device = device

        #出力サイズ = (入力サイズ - フィルターサイズ + 2 × パディング) ÷ ストライド + 1
        self.d_in1 = (image_size - 3 + 2 * 0) // 1 + 1
        self.d_in2 = (image_size - 3 + 2 * 0) // 1 + 1
        self.conv_channel = param["conv_channel"]

        self.in1 = nn.Sequential(
            nn.Conv2d(1, self.conv_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.conv_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(self.conv_channel * self.d_in1 * self.d_in1, image_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.in2 = nn.Sequential(
            nn.Conv2d(1, self.conv_channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(self.conv_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            nn.Flatten(),
            nn.Linear(self.conv_channel * self.d_in2 * self.d_in2, image_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.ff_input_size = image_size + image_size

        self.seq = nn.Sequential(
            nn.Linear(self.ff_input_size, self.d_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, image, subject):
        yi1 = image
        yi2 = subject

        yi1 = torch.nn.functional.interpolate(yi1, scale_factor=0.5, mode="area")
        yi2 = torch.nn.functional.interpolate(yi2, scale_factor=0.5, mode="area")

        #jb.display_image_async(yi1.squeeze(0))
        #jb.display_image_async(yi2.squeeze(0))

        yi1 = self.in1(yi1)
        yi2 = self.in2(yi2)

        yi1 = yi1.view(-1, self.image_size)
        yi2 = yi2.view(-1, self.image_size)

        seq_input = torch.cat((yi1, yi2), dim=1)
        #seq_input = seq_input.unsqueeze(0)
        y = self.seq(seq_input)
        return y


class DQN(nn.Module):
    def __init__(self, param, batch_size=1, device='cpu'):
        super().__init__()
        self.device = device

        self.main_unit1 = MainUnit(param["main_layer"], param["d_image"], device = device)

        print(param)
        input_size = param["d_in"]
        output_size = param["d_dict"]
        hidden_size = param["output_layer"]["d_hidden"]

        self.seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )


    def forward(self, image, canvas):
        # MainUnit
        y = self.main_unit1(image, canvas)
        y = self.seq(y)

        return y
