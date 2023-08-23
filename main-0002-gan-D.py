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
import stroke_to_image
import model002
import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'output/main/0002-gan-D/'
os.makedirs(path, exist_ok=True)

hyper_parameter = {
    "num_pre_epoch": 64,
    "num_epoch": 512,
    "learning_rate": 0.0002,
    "num_words": 256,
    "num_words_fragment": 256
}

model_parameter = {
    "d_dict": 19,
    "d_model": hyper_parameter["num_words"],
    "d_word": 128,
    "d_image": 64,
    "stroke_max": 1024,
    "main_layer": {
        "d_hidden": 256,
        "num_head": 4
    },
    "output_layer": {
        "d_hidden": 256
    }
}

def zero_expand(x, width):
    ext = torch.zeros(width - x.shape[0], dtype=torch.int)
    return torch.cat((x, ext))


def roll_and_fill(img, x, y):

    moved_img = torch.roll(img, shifts=(x, y), dims=(1, 2))
    if x > 0:
        moved_img[:, :x, :] = 1
    elif x < 0:
        moved_img[:, x:, :] = 1
    if y > 0:
        moved_img[:, :, :y] = 1
    elif y < 0:
        moved_img[:, :, y:] = 1

    return moved_img


def Gl(G, x):
    frag_num = hyper_parameter["num_words_fragment"]

    canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"])
    canvas = canvas.to(device)
    for i in range(0, hyper_parameter["num_words"] + 1, frag_num):
        canvas += Gl_step(G, x, canvas, frag_num).to(device)
        canvas = torch.clamp(canvas, max=0, min=-1)

    #print(stroke)
    return canvas.unsqueeze(0).unsqueeze(0)


def Gl_step(G, x, canvas, num_word):
    stroke = [1]
    px0 = model_parameter["d_image"] // 2
    py0 = model_parameter["d_image"] // 2

    px = px0
    py = py0

    x = torch.nn.functional.pad(x, (
        model_parameter["d_image"]//2,
        model_parameter["d_image"]//2,
        model_parameter["d_image"]//2,
        model_parameter["d_image"]//2), mode='constant', value=1)

    noise = torch.randn_like(x) * 0.3 - 0.5
    x = x + noise
    x0 = torch.clamp(x, max=1.0, min=0.0).squeeze(0)

    for i in range(num_word):
        x1 = roll_and_fill(x0, px0 - px, py0 - py)

        img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
        x2 = roll_and_fill(img.unsqueeze(0), px0 - px, py0 - py)

        x1 = x1.flatten(start_dim=1).squeeze()
        x2 = x2.flatten(start_dim=1).squeeze()
        strk = zero_expand(torch.tensor(stroke), hyper_parameter["num_words"])

        strk, x1, x2 = strk.to(device), x1.to(device), x2.to(device)
        y = G(strk, x1, x2, len(stroke))
        y = torch.argmax(y).item()

        canvas, px, py = stroke_to_image.draw(canvas, y, (px, py), model_parameter["d_image"], device)

        stroke.append(y)

    #print(stroke)

    return canvas


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

#def D_train(G, D, D_optim, x):
#    # (2, H, W) -> (1, H, W), (1, H, W)
#    t_data, f_data = torch.split(x, [1, 1], dim=0)
#    t_data, f_data = t_data.to(device), f_data.to(device)
#    frag_num = hyper_parameter["num_words_fragment"]
#
#    canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"])
#    canvas = canvas.to(device)
#    for i in range(0, hyper_parameter["num_words"] + 1, frag_num):
#        y = Gl_step(G, t_data, canvas, frag_num).to(device)
#
#        canvas = canvas + y
#        canvas = torch.clamp(canvas, max=0, min=-1).to(device)
#
#        img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
#        img = img.unsqueeze(0).unsqueeze(0).to(device)
#        y_pred = D(img, t_data)
#        y_real = torch.full_like(y_pred, 1, device=device)
#        loss_real = nn.BCELoss()(y_pred, y_real)
#
#        D.zero_grad()
#        loss_real.backward()
#        D_optim.step()
#
#    canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"])
#    canvas = canvas.to(device)
#    for i in range(0, hyper_parameter["num_words"] + 1, frag_num):
#        canvas += Gl_step(G, t_data, canvas, frag_num).to(device)
#        canvas = torch.clamp(canvas, max=0, min=-1)
#
#        img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
#        img = img.unsqueeze(0).unsqueeze(0).to(device)
#        y_pred = D(img, f_data)
#        y_fake = torch.full_like(y_pred, 0, device=device)
#        loss_fake = nn.BCELoss()(y_pred, y_fake)
#
#        D.zero_grad()
#        loss_fake.backward()
#        D_optim.step()
#
#    loss = loss_real + loss_fake + D_train_boost(D, D_optim, x)
#    return float(loss)


def D_train_boost(D, D_optim, x):
    t_data, f_data = torch.split(x, [1, 1], dim=0)
    t_data, f_data = t_data.to(device), f_data.to(device)

    y_pred = D(t_data, t_data)
    y_real = torch.full_like(y_pred, 1, device=device)
    loss_real = nn.BCELoss()(y_pred, y_real)

    D.zero_grad()
    loss_real.backward()
    D_optim.step()

    y_pred = D(t_data, f_data)
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
    frag_num = hyper_parameter["num_words_fragment"]

    canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"]).to(device)
    canvas = canvas.to(device)
    for i in range(0, hyper_parameter["num_words"] + 1, frag_num):
        y = Gl_step(G, x1, canvas, frag_num).to(device)
        canvas += y
        canvas = torch.clamp(canvas, max=0, min=-1)

        img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        y_pred = D(img, x1)
        y = torch.full_like(y_pred, 1, device=device)
        loss = nn.BCELoss()(y_pred, y)

        G.zero_grad()
        loss.backward()
        G_optim.step()

    canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"])
    canvas = canvas.to(device)
    for i in range(0, hyper_parameter["num_words"] + 1, frag_num):
        canvas += Gl_step(G, x1, canvas, frag_num).to(device)
        canvas = torch.clamp(canvas, max=0, min=-1)

        img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
        img = img.unsqueeze(0).unsqueeze(0).to(device)
        y_pred = D(img, x2)
        y = torch.full_like(y_pred, 1, device=device)
        loss = nn.BCELoss()(y_pred, y)

        G.zero_grad()
        loss.backward
        G_optim.step()

    return float(loss)



def train(D, G, n_epoch, pre_n_epoch):
    D.to(device)
    G.to(device)

    D_optim = optim.Adam(D.parameters(), lr=hyper_parameter["learning_rate"])
    G_optim = optim.Adam(G.parameters(), lr=hyper_parameter["learning_rate"])

    D.train()
    G.train()

    pre_history = []
    for epoch in tqdm(range(pre_n_epoch)):
        pre_D_losses = []
        for x, _ in data_pair_generator():
            x = x.to(device)
            loss_d = D_train_boost(D, D_optim, x)
            pre_D_losses.append(loss_d)
            print("D:", float(loss_d))

        # 途中経過を記録する。
        pre_history.append({
            "epoch": epoch + 1,
            "D_loss": np.mean(pre_D_losses),
            "G_loss": 0,
        })

    pre_history = pd.DataFrame(pre_history)
    jb.plot_history(pre_history, path + "pre_history.png")


    history = []
    for epoch in tqdm(range(n_epoch)):
        D_losses, G_losses = [], []

        for x, _ in data_pair_generator():
            x = x.to(device)

            #loss_d = D_train(G, D, D_optim, x)
            loss_g = G_train(G, D, G_optim, x)

            #print("D:", float(loss_d))
            print("G:", float(loss_g))

            #D_losses.append(loss_d)
            D_losses.append(0)
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
                canvas = Gl(G, x).squeeze()
                img = canvas + torch.ones(model_parameter["d_image"], model_parameter["d_image"]).to(device)
                jb.save_image(img, path + "%s-%s-generate.png" % ((epoch + 1), i))

    history = pd.DataFrame(history)

    return history


if __name__ == "__main__":
    G = model002.Network(model_parameter, device)
    D = trainer.Discriminator()

    history = train(D,G, hyper_parameter["num_epoch"], hyper_parameter["num_pre_epoch"])
    jb.plot_history(history, path + "history.png")

