import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms


import stroke_to_image2
import model013
import trainer

from torch.utils.tensorboard import SummaryWriter
path = 'output/main/0013/'
os.makedirs(path, exist_ok=True)
tb_log = path + "tblog"
writer = SummaryWriter(log_dir=tb_log)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


hyper_parameter = {
    "pre_num_epoch": 256,
    "pre_learning_rate": 0.0001,
    "num_epoch": 8192,
    "learning_rate": 0.0001,
    "num_words": 200,
    # DQN Param
    "discount_factor": 0.999,
    # Fixed target q-network
    # "tau": 0.005,
    "tau": 0.02,
    # For Epsilon Greedy
    "epsilon_decay": 0.995,
    "epsilon_min": 0.05,
    # Experience Replay
    "memory_size": 30000,
}

model_parameter = {
    "d_image": 64,
    "d_model": 128,
    "seq_len": hyper_parameter["num_words"],
    "n_layer": 8,
    "n_head": 8,
    "d_vocab": stroke_to_image2.N_WORD
}

lossfunc = nn.SmoothL1Loss()


def data_generator(bnum=1, shuffle=True):
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
    )

    batch_size = bnum
    dataset = torchvision.datasets.ImageFolder("data/images", transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Train Discriminator
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

    writer.add_scalar('Loss/D_train', np.mean(pre_D_losses), epoch)

D.eval()
del D_optim
torch.cuda.empty_cache()


# Train Genertor
class Environment:
    def __init__(self, D, subject_img, num_words):
        self.px0 = model_parameter["d_image"] // 2
        self.py0 = model_parameter["d_image"] // 2
        self.dx = 0
        self.dy = 0
        self.timestep = 0
        self.num_words = num_words
        self.D = D

        noise = torch.randn_like(subject_img) * 0.1
        subject_img = subject_img + noise
        # (1, 1, W, H) -> (1, W, H)
        self.subject_img = torch.clamp(subject_img, max=1.0, min=0.0).squeeze(0)

        # (1, W, H)
        self.canvas = torch.zeros(model_parameter["d_image"],
                                  model_parameter["d_image"],
                                  requires_grad=False).unsqueeze(0)

        self.last_reward = 0
        self.initialized = False

    def _canvas_to_img(self, canvas):
        return canvas + 1

    def _roll_and_fill(self, img, x, y):
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

    def _observation(self):
        img = self._canvas_to_img(self.canvas)

        st1 = torch.nn.functional.pad(img, (
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2),
            mode='constant', value=1).to(device)

        st2 = torch.nn.functional.pad(self.subject_img, (
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2,
            model_parameter["d_image"] // 2),
            mode='constant', value=1).to(device)

        st1 = self._roll_and_fill(st1, -self.dx, -self.dy)
        st2 = self._roll_and_fill(st2, -self.dx, -self.dy)

        # (1, W, H) x (1, W, H) -> (2, W, H)
        return torch.cat((st1, st2), dim=0)

    def first_state(self):
        return self._observation()

    def get_subject_img(self):
        return self.subject_img

    def get_img(self):
        return self._canvas_to_img(self.canvas.detach())

    def step(self, action):
        px = self.px0 + self.dx
        py = self.py0 + self.dy

        canvas = self.canvas.squeeze(0)
        px, py, canvas = stroke_to_image2.draw(canvas, action, (px, py), model_parameter["d_image"], device)
        self.canvas = canvas.unsqueeze(0)

        self.dx = px - self.px0
        self.dy = py - self.py0

        img = self._canvas_to_img(self.canvas)

        # (1, width, height) -> (1, 1, width, height)
        di1 = img.unsqueeze(0).detach().to(device)
        # (1, width, height) -> (1, 1, width, height)
        di2 = self.subject_img.unsqueeze(0).detach().to(device)

        # Calcurate reward
        reward_swap = self.D(di1, di2)
        reward = reward_swap[0] - self.last_reward
        self.last_reward = reward_swap[0]

        self.timestep = self.timestep + 1
        done = 1 if self.timestep == self.num_words - 1 else 0

        # Calcurate reward
        return self._observation(), reward.detach(), self.timestep, done


def to_attention_mask(timesteps):
    tlen = hyper_parameter['num_words']
    cstep = timesteps[-1]
    msk = torch.ones(tlen)
    msk[:cstep + 1] = torch.zeros(cstep + 1, dtype=torch.long)

    return msk.bool()


def memory_to_batch(states, actions, timesteps):
    tlen = hyper_parameter['num_words']
    cstep = timesteps[-1]

    # [(2, W, H) ...] -> (n, 2, W, H)
    states = torch.stack(states)
    b_states = torch.zeros(tlen, states.shape[1], states.shape[2], states.shape[3])
    # (n, 2, W, H) -> (seq_len, 2, W, H)
    b_states[:cstep + 1, :, :, :] = states

    # [int, ...] -> (n, 1)
    actions = torch.Tensor(actions).reshape(-1, 1)
    b_actions = torch.zeros(tlen, actions.shape[1])
    # (n, 1) -> (seq_len, 1)
    b_actions[:cstep + 1, :] = actions
    b_actions = b_actions.to(torch.int64)

    # [int, ...] -> (n, 1)
    timesteps = torch.Tensor(timesteps).reshape(-1, 1)
    b_timesteps = torch.zeros(tlen, timesteps.shape[1])
    # (n, 1) -> (seq_len, 1)
    b_timesteps[:cstep + 1, :] = timesteps
    b_timesteps = b_timesteps.to(torch.int64)

    b_states = b_states.unsqueeze(0).to(device)
    b_actions = b_actions.unsqueeze(0).to(device)
    b_timesteps = b_timesteps.unsqueeze(0).to(device)

    return b_states, b_actions, b_timesteps, cstep


def DQN_train(DQN_policy, DQN_target, DQN_optim, memory):
    states, actions, rewards, next_states, dones, timesteps = zip(*memory)

    rewards = torch.tensor([rewards[-1]]).to(device)
    dones = torch.tensor([dones[-1]]).to(device)

    with torch.no_grad():
        b_states, b_actions, b_timesteps, cstep = memory_to_batch(next_states, actions, timesteps)
        b_atn_msk = to_attention_mask(timesteps).unsqueeze(0).to(device)

        Q_next = DQN_target(b_states, b_actions, b_timesteps, b_atn_msk, cstep)
        Q_next = Q_next.max(1)[0]

    yi = rewards + hyper_parameter["discount_factor"] * Q_next * (1 - dones)

    b_states, b_actions, b_timesteps, cstep = memory_to_batch(states[:-1], actions[:-1], timesteps[:-1])
    b_atn_msk = to_attention_mask(timesteps[:-1]).unsqueeze(0).to(device)

    Q_expected = DQN_policy(b_states, b_actions, b_timesteps, b_atn_msk, cstep)

    actions = torch.tensor(actions).unsqueeze(1).to(device)
    # !TODO is this correct???
    Q_expected = Q_expected.gather(1, actions[-1:]).squeeze(0)

    loss = lossfunc(yi, Q_expected)
    loss.backward()

    return torch.mean(loss)


# Init meta variables
DQN_policy = model013.DecisionTransformer(model_parameter).to(device)
DQN_target = model013.DecisionTransformer(model_parameter).to(device)
DQN_target.load_state_dict(DQN_policy.state_dict())
DQN_optim = optim.Adam(DQN_policy.parameters(), lr=hyper_parameter["learning_rate"])


def soft_update_DQN_target(DQN_target, DQN_policy):
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_dict = DQN_target.state_dict()
    policy_dict = DQN_policy.state_dict()
    tau = hyper_parameter["tau"]
    for key in policy_dict:
        target_dict[key] = policy_dict[key] * tau + target_dict[key] * (1 - tau)
    DQN_target.load_state_dict(target_dict)


def test_current_agent(D, DQN_target):
    for img_num, (x, _) in enumerate(data_generator(1, False)):
        # save image
        env = Environment(D, x, hyper_parameter["num_words"])
        state = env.first_state()
        memory = []
        memory.append((state, 0, 0, state, 0, 0))
        done = 0
        while done == 0:
            with torch.no_grad():
                states, actions, rewards, next_states, dones, timesteps = zip(*memory)
                b_states, b_actions, b_timesteps, cstep = memory_to_batch(states, actions, timesteps)
                b_atn_msk = to_attention_mask(timesteps).unsqueeze(0).to(device)
                action = DQN_target(b_states, b_actions, b_timesteps, b_atn_msk, cstep).argmax(1).item()
            # Play action
            next_state, reward, timestep, done = env.step(action)
            memory.append((state, action, reward, next_state, done, timestep))

            state = next_state

        timg = env.get_subject_img()
        img = env.get_img()

        gtimg = torchvision.utils.make_grid(timg, nrow=1).to('cpu')
        gimg = torchvision.utils.make_grid(img, nrow=1).to('cpu')

        grid_image = torch.cat((gtimg, gimg), dim=2).to('cpu')

        name = f'image/ts_{img_num+1}'
        writer.add_image(name, grid_image, epoch, dataformats='CHW')


# learning loop
epsilon = 1.0
replay_memory = []

for epoch in range(hyper_parameter["num_epoch"]):
    for subject_num, (subject, _) in enumerate(data_generator(1, False)):

        env = Environment(D, subject, hyper_parameter["num_words"])
        state = env.first_state()
        memory = []
        memory.append((state, 0, 0, state, 0, 0))

        done = 0
        timestep = 0
        total_reward = torch.zeros((1, 1))

        reward_hist = []

        while done == 0:
            # Decide action
            if np.random.rand() <= epsilon:
                action = torch.randint(low=0, high=stroke_to_image2.N_WORD, size=(1,)).item()
            else:
                states, actions, rewards, next_states, dones, timesteps = zip(*memory)
                b_states, b_actions, b_timesteps, cstep = memory_to_batch(states, actions, timesteps)
                b_atn_msk = to_attention_mask(timesteps).unsqueeze(0).to(device)
                action = DQN_policy(b_states, b_actions, b_timesteps, b_atn_msk, cstep).argmax(1).item()

            # Tensorboard. add sample state view.
            if epoch == 0 and subject_num == 0:
                states = state.unsqueeze(0)
                grid_image = torchvision.utils.make_grid(states, nrow=2)
                name = 'comp/x'
                writer.add_image(name, grid_image, timestep)

            # Play action
            next_state, reward, timestep, done = env.step(action)
            total_reward += reward.to('cpu')

            memory.append((state, action, reward, next_state, done, timestep))

            # DQN_train
            DQN_optim.zero_grad()

            states, actions, rewards, next_states, dones, timesteps = zip(*memory)
            loss_mean = DQN_train(DQN_policy, DQN_target, DQN_optim, memory)

            if subject_num == 0:
                writer.add_scalar('meta/LossDQN', loss_mean, epoch)

            DQN_optim.step()

            state = next_state

            print('Episode {}, subject {}, Word {:03d}: Total Reward = {:.6f}, Epsilon = {:.2f}, Action = {}, Done = {}, Device = {}'.format(
                epoch, subject_num, timestep, total_reward.item(), epsilon, action, done, device))

            reward_hist.append(total_reward.item())

            soft_update_DQN_target(DQN_target, DQN_policy)

        writer.add_scalar('reward/DQN', np.mean(reward_hist), epoch)

    if epoch % 20 == 0:
        test_current_agent(D, DQN_target)

    if epoch % 100 == 0:
        torch.save(DQN_target.state_dict(), path + "model-%s.pth" % (epoch))

    # update epsilon
    if epsilon > hyper_parameter["epsilon_min"]:
        epsilon *= hyper_parameter["epsilon_decay"]
        writer.add_scalar('meta/epsilon', epsilon, epoch)
