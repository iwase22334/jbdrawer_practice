import torch
import torchviz
from torch import nn
import json
import math
import jbutils as jb;

class DQN(nn.Module):
    def __init__(self, param, batch_size=1, device='cpu'):
        super().__init__()
        self.device = device

        hidden = param["d_hidden"]
        image_size = param["d_image"]
        channel1 = param["conv_channel1"]
        channel2 = param["conv_channel2"]
        channel3 = param["conv_channel3"]
        output_size = param["d_dict"]

        #出力サイズ = (入力サイズ - フィルターサイズ + 2 × パディング) ÷ ストライド + 1
        d_in1 = (image_size - 3 + 2 * 1) // 1 + 1

        self.seq = nn.Sequential(
            nn.Conv2d(2, channel1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(channel1, channel2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(channel2, channel3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Flatten(),

            nn.Linear(channel3 * d_in1 * d_in1, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, output_size)
        )

    def forward(self, image, subject):
        # MainUnit

        yi1 = image
        yi2 = subject

        yi1 = torch.nn.functional.interpolate(yi1, scale_factor=0.5, mode="area")
        yi2 = torch.nn.functional.interpolate(yi2, scale_factor=0.5, mode="area")

        #yi = torch.cat(yi1, yi2)
        yi = torch.cat((yi1, yi2), dim=1)
        y = self.seq(yi)

        return y
