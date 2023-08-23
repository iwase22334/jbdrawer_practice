import numpy as np
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import jbutils as jb;
from stroke_to_image import convert
import model
import trainer


model_parameter = {
    "d_dict": 19,
    "d_model": 128,
    "d_image": 64,
    "stroke_max": 300,
    "main_layer": {
        "d_hidden": 1024,
        "num_head": 4
    },
    "output_layer": {
        "d_hidden": 1024
    }
}

def zero_extpand(x, target_size):
    ext = torch.zeros(target_size - x.shape[0], dtype=torch.int)
    return torch.cat((x, ext))

if __name__ == "__main__":
    stroke = [1]
    ## dummy image
    net = model.Network(model_parameter)
    image = torch.randn(model_parameter["d_image"] ** 2)

    for i in range(64):
        ## Calcurate word embedding
        x = zero_extpand(torch.tensor(stroke), model_parameter["d_model"])
        y = net(x, image)

        stroke.append(torch.argmax(y).item())

    print(stroke)

    img = convert(stroke, model_parameter["d_image"])

    jb.display_image(img)

