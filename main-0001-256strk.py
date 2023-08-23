import os

import numpy as np
from tqdm import tqdm

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchvision
from torchvision import datasets, transforms

import jbutils as jb;
from stroke_to_image import convert
import model
import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'output/main/0001-256strk/'
os.makedirs(path, exist_ok=True)

hyper_parameter = {
    "num_epoch": 1024,
    "learning_rate": 0.0002,
    "num_words": 256
}

model_parameter = {
    "d_dict": 19,
    "d_model": hyper_parameter["num_words"],
    "d_word": 128,
    "d_image": 64,
    "stroke_max": 300,
    "main_layer": {
        "d_hidden": 516,
        "num_head": 4
    },
    "output_layer": {
        "d_hidden": 516
    }
}

def zero_expand(x, width):
    ext = torch.zeros(width - x.shape[0], dtype=torch.int)
    return torch.cat((x, ext))


def Gl(G, x):
    stroke = [1]
    x = x.flatten(start_dim=2).squeeze()

    for i in range(hyper_parameter["num_words"]):
        strk = zero_expand(torch.tensor(stroke), hyper_parameter["num_words"])
        strk, x = strk.to(device), x.to(device)
        y = G(strk, x, len(stroke))

        stroke.append(torch.argmax(y).item())

    #print(stroke)
    return convert(stroke, model_parameter["d_image"]).unsqueeze(0).unsqueeze(0)


def data_pair_generator():
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    batch_size = 2 
    dataset = torchvision.datasets.ImageFolder("data/images", transform = transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

def data_seq_generator():
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    batch_size = 1  
    dataset = torchvision.datasets.ImageFolder("data/images", transform = transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def D_train(G, D, D_optim, x):
    # (2, H, W) -> (1, H, W), (1, H, W)
    t_data, f_data = torch.split(x, [1, 1], dim=0)
    t_data, f_data = t_data.to(device), f_data.to(device)

    img = Gl(G, t_data).to(device)
    y_pred = D(img, t_data)
    y_real = torch.full_like(y_pred, 1, device=device)
    loss_real = nn.BCELoss()(y_pred, y_real)

    D.zero_grad()
    loss_real.backward()
    D_optim.step()

    #print(img)
    #print(t_data)
    #jb.display_image(img[0])
    #jb.display_image(t_data[0])

    img = Gl(G, t_data).to(device)

    y_pred = D(img, f_data)
    y_fake = torch.full_like(y_pred, 0, device=device)
    loss_fake = nn.BCELoss()(y_pred, y_fake)

    D.zero_grad()
    loss_fake.backward()
    D_optim.step()

    loss = loss_real + loss_fake
    return float(loss)


def G_train(G, D, G_optim, x):
    # (2, H, W) -> (1, H, W), (1, H, W)
    x1, x2 = torch.split(x, [1, 1], dim=0)
    x1, x2 = x1.to(device), x2.to(device)

    img = Gl(G, x1).to(device)
    y_pred = D(img, x1)
    y = torch.full_like(y_pred, 1, device=device)
    loss = nn.BCELoss()(y_pred, y)

    G.zero_grad()
    loss.backward()
    G_optim.step()

    img = Gl(G, x2).to(device)
    y_pred = D(img, x2)
    y = torch.full_like(y_pred, 1, device=device)
    loss = nn.BCELoss()(y_pred, y)

    G.zero_grad()
    loss.backward
    G_optim.step()

    return float(loss)


def train(D, G, n_epoch):
    D.to(device)
    G.to(device)

    D_optim = optim.Adam(D.parameters(), lr=hyper_parameter["learning_rate"])
    G_optim = optim.Adam(G.parameters(), lr=hyper_parameter["learning_rate"])

    D.train()
    G.train()

    history = []

    for epoch in tqdm(range(n_epoch)):
        D_losses, G_losses = [], []

        for x, _ in data_pair_generator():
            x = x.to(device)
            noise = torch.randn_like(x) * 0.3 - 0.5
            x = x + noise
            x = torch.clamp(x, max=1.0, min=0.0)

            loss_d = D_train(G, D, D_optim, x)
            loss_g = G_train(G, D, G_optim, x)

            print("D:", float(loss_d))
            print("G:", float(loss_g))

            D_losses.append(loss_d)
            G_losses.append(loss_g)


        # 途中経過を記録する。
        history.append({
            "epoch": epoch + 1,
            "D_loss": np.mean(D_losses),
            "G_loss": np.mean(G_losses),
        })

        if epoch % 10 == 0:
            for i, (x, _) in enumerate(data_seq_generator()):
                x = x.to(device)
                img = Gl(G, x).squeeze()
                jb.save_image(img, path + "%s-%s-generate.png" % ((epoch + 1), i))

    history = pd.DataFrame(history)

    return history


if __name__ == "__main__":
    G = model.Network(model_parameter, device)
    D = trainer.Discriminator()

    history = train(D,G, hyper_parameter["num_epoch"])
    jb.plot_history(history, path + "history.png")

