import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ハイパーパラメータの設定
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
batch_size = 64
memory_size = 10000
episodes = 500

# 環境の生成
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Q-networkの構築
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, state):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = QNetwork()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# メモリの初期化
memory = []

# 学習ループ
for e in range(episodes):
    state = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0)
    done = False
    total_reward = 0

    while not done:
        # 行動の選択
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = model(state).argmax(1).item()

        # 環境の実行
        next_state, reward, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        total_reward += reward

        # メモリに経験を追加
        memory.append((state, action, reward, next_state, done))
        if len(memory) > memory_size:
            memory.pop(0)

        # 学習
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*minibatch)
            states = torch.cat(states)
            actions = torch.tensor(actions).unsqueeze(1)
            rewards = torch.tensor(rewards).unsqueeze(1)
            next_states = torch.cat(next_states)
            dones = torch.tensor(dones).unsqueeze(1)

            Q_targets_next = model(next_states).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + discount_factor * Q_targets_next * (1 - dones)

            Q_expected = model(states).gather(1, actions)
            loss = nn.functional.mse_loss(Q_expected, Q_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state

    # Epsilonの減衰
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # 結果の表示
    print("Episode {}: Total Reward = {}, Epsilon = {:.2f}".format(e + 1, total_reward, epsilon))

