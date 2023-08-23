import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.inlayer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.inlayer2 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.main = nn.Sequential(
            nn.Conv2d(2 * 32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),

            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
            nn.Linear(62 * 62, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, image):
        x1 = self.inlayer1(x)
        x2 = self.inlayer2(image)
        inp = torch.cat((x1, x2), dim=1)

        return self.main(inp)

