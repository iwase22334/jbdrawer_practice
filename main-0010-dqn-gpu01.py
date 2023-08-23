import os
import random

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
import model010
import trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = 'output/main/0010-dqn-gpu01/'
os.makedirs(path, exist_ok=True)

hyper_parameter = {
    "pre_num_epoch": 256,
    "num_epoch": 1024,
    "pre_learning_rate": 0.0001,
    "learning_rate": 0.0010,
    "num_words": 200,
    "discount_factor": 0.99,
    "epsilon_decay": 0.97,
    "epsilon_min": 0.05,
    "batch_size": 128,
    "memory_size": 10000,
    "d_joint": 128,
}

model_parameter = {
    "d_dict": 19,
    "d_in": hyper_parameter["d_joint"],
    "d_image": 64,
    "main_layer": {
        "conv_channel": 4,
        "d_out": hyper_parameter["d_joint"]
    },
    "output_layer": {
        #"d_hidden": 256
        "d_hidden": 64
    }
}

def data_generator(bnum=1, shuffle=True):
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    batch_size = bnum
    dataset = torchvision.datasets.ImageFolder("data/images", transform = transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


## Train Discriminator
def D_train(D, D_optim, x):
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


# Train D
D = trainer.Discriminator().to(device)
D_optim = optim.Adam(D.parameters(), lr=hyper_parameter["pre_learning_rate"])

pre_history = []
for epoch in tqdm(range(hyper_parameter["pre_num_epoch"])):
    pre_D_losses = []
    for x, _ in data_generator(2, True):
        noise = torch.randn_like(x) * 0.2
        x = x + noise
        x = torch.clamp(x, max=1.0, min=0.0).squeeze(0)

        loss_d = D_train(D, D_optim, x.to(device))
        pre_D_losses.append(loss_d)
        print("D:", float(loss_d))

    # 途中経過を記録する。
    pre_history.append({
        "epoch": epoch + 1,
        "D_loss": np.mean(pre_D_losses),
        "G_loss": 0,
    })

del D_optim
torch.cuda.empty_cache()

pre_history = pd.DataFrame(pre_history)
jb.plot_history(pre_history, path + "pre_history.png")


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


## Train Genertor
class Environment:
    def __init__(self, D, subject_img):
        #self.stroke = [1]
        self.px0 = model_parameter["d_image"] // 2
        self.py0 = model_parameter["d_image"] // 2
        self.dx = 0
        self.dy = 0
        self.D = D

        noise = torch.randn_like(subject_img) * 0.1
        subject_img = subject_img + noise
        # (1, 1, W, H), (1, W, H)
        self.subject_img = torch.clamp(subject_img, max=1.0, min=0.0).squeeze(0)

        # (1, W, H)
        canvas = torch.zeros(model_parameter["d_image"], model_parameter["d_image"]).unsqueeze(0)
        self.canvas = canvas.to(device)

    def _canvas_to_img(self, canvas):
        ## canvas to image
        return canvas + 1

    def first_state(self):
        ## canvas to image
        img = self._canvas_to_img(self.canvas)

        st1 = torch.nn.functional.pad(img, (
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2), mode='constant', value=1).to(device)

        st2 = torch.nn.functional.pad(self.subject_img, (
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2), mode='constant', value=1).to(device)

        # (B, 1, 2*width, 2*height) -> (B, 1, 2*width, 2*height)
        st1 = roll_and_fill(st1, -self.dx, -self.dy)
        # (B, 1, 2*width, 2*height) -> (B, 1, 2*width, 2*height)
        st2 = roll_and_fill(st2, -self.dx, -self.dy)

        return (st1.detach(), st2.detach())

    def get_canvas(self):
        return self.canvas.detach()

    def step(self, action):
        ## Update env

        # (1, W, H) -> (W, H)
        #self.canvas = self.canvas.view(model_parameter["d_image"], model_parameter["d_image"])

        px = self.px0 + self.dx
        py = self.py0 + self.dy

        canvas = self.canvas.squeeze(0)
        px, py, canvas = stroke_to_image.draw(canvas, action, (px, py), model_parameter["d_image"], device)
        self.canvas = canvas.unsqueeze(0)

        self.dx = px - self.px0
        self.dy = py - self.py0

        print("px,py = {},{}, action = {} ".format(px, py, action))

        ## canvas to image
        img = self._canvas_to_img(self.canvas)

        ## (1, width, height) -> (1, 1, width, height)
        di1 = img.unsqueeze(0).detach().to(device)
        ## (1, width, height) -> (1, 1, width, height)
        di2 = self.subject_img.unsqueeze(0).detach().to(device)

        reward = self.D(di1, di2)

        # (W, H) -> (1, W, H)
        #self.canvas = self.canvas.unsqueeze(0)

        st1 = torch.nn.functional.pad(img, (
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2), mode='constant', value=1).to(device)

        st2 = torch.nn.functional.pad(self.subject_img, (
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2,
            model_parameter["d_image"]//2), mode='constant', value=1).to(device)

        # (B, 1, 2*width, 2*height) -> (B, 1, 2*width, 2*height)
        st1 = roll_and_fill(st1, -self.dx, -self.dy)
        # (B, 1, 2*width, 2*height) -> (B, 1, 2*width, 2*height)
        st2 = roll_and_fill(st2, -self.dx, -self.dy)

        return (st1.detach(), st2.detach()), reward.detach()


def predict_Q(state):
    img, subject_img = zip(*state)
    img = torch.cat(img)

    subject_img = torch.cat(subject_img)

    # (4 , width, height) -> (4, 1 , width, height)
    x1 = img.unsqueeze(1)
    x2 = subject_img.unsqueeze(1)

    x1, x2 = x1.to(device), x2.to(device)
    q = G(x1, x2)

    #print(q)
    return q


def G_train(G, G_optim, states, actions, rewards, next_states, dones):
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards).unsqueeze(1).to(device)
    dones   = torch.tensor(dones).unsqueeze(1).to(device)

    Q_next = predict_Q(next_states).detach().max(1)[0].unsqueeze(1).to(device)
    yi = rewards + hyper_parameter["discount_factor"] * Q_next * (1 - done)
    #yi = yi.squeeze(0)[0]

    Q_expected = predict_Q(states).gather(1, actions)

    loss = nn.functional.mse_loss(yi, Q_expected)
    loss.backward()



# Init meta variables
memory = []
G = model010.DQN(model_parameter, device).to(device)
G_optim = optim.Adam(G.parameters(), lr=hyper_parameter["learning_rate"])

# learning loop
history = []
epsilon = 1.0

for epoch in range(hyper_parameter["num_epoch"]):
    for subject_num, (subject, _) in enumerate(data_generator(1, False)):

        env = Environment(D, subject)
        state = env.first_state()
        done = False
        total_reward = 0
        reward_pre = 0
        prev_reward = 0

        reward_hist = []

        for nw in range(hyper_parameter["num_words"]):
            # Decide action
            if np.random.rand() <= epsilon:
                action = torch.randint(low=0, high=19, size=(1,)).item()
            else:
                action = predict_Q([state]).argmax(1).item()

            # Play action
            next_state, reward = env.step(action)
            reward1 = reward - reward_pre if reward_pre == 0 else 0
            reward_pre = reward
            total_reward += reward1
            #next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            #total_reward += reward

            # append experience to memory for boosting learning
            done = nw == hyper_parameter["num_words"]

            ## !TODO   which is the correct?
            memory.append((state, action, reward1, next_state, done))
            #memory.append((state, action, reward, next_state, done))
            #memory.append((state, action, total_reward, next_state, done))
            if len(memory) > hyper_parameter["memory_size"]:
                memory.pop(0)

            # G_train
            if len(memory) >= hyper_parameter["batch_size"]:
                print("start learning ... >", end="", flush=True)
                minibatch = random.sample(memory, hyper_parameter["batch_size"])
                G_optim.zero_grad()

                states, actions, rewards, next_states, dones = zip(*minibatch)
                G_train(G, G_optim, states, actions, rewards, next_states, dones)

                ## This function takes may time
                G_optim.step()
                print("<")

            state = next_state

            print("Episode {}, subject {}, Word {:03d}: Total Reward = {:.6f}, Epsilon = {:.2f}, Action = {}".format(
                epoch + 1, subject_num + 1, nw + 1, total_reward.detach()[0].item(), epsilon, action))

            reward_hist.append(total_reward.detach()[0].item())


        if epoch % 4 == 0 and subject_num == 0:
            for img_num, (x, _) in enumerate(data_generator(1, False)):
                ## save image
                env = Environment(D, subject)
                state = env.first_state()
                for _ in range(hyper_parameter["num_words"]):
                    #action = predict_Q([state]).unsqueeze(0).argmax(1).item()
                    qs = predict_Q([state])
                    action = predict_Q([state]).argmax(1).item()
                    print(action, ", ", end="", flush=True)
                    # Play action
                    state, _ = env.step(action)

                canvas = env.get_canvas()
                img = canvas + 1
                jb.save_image(img, path + "%s-%s-generate.png" % ((epoch + 1), img_num))
                print("image saved", path + "%s-%s-generate.png" % ((epoch + 1), img_num))

                if img_num == 3:
                    break

        history.append({
            "epoch": epoch + 1,
            "rewards": np.mean(reward_hist),
        })

    # update epsilon
    if epsilon > hyper_parameter["epsilon_min"]:
        epsilon *= hyper_parameter["epsilon_decay"]


## Plot
import matplotlib.pyplot as plt
history = pd.DataFrame(history)
fig, ax = plt.subplots()
ax.set_title("rewards")
ax.plot(history["epoch"], history["rewards"], label="rewards")
ax.set_xlabel("Epoch")
ax.legend()
fig.savefig(path + "history.png")


