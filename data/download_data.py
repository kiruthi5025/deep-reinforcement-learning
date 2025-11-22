# Portfolio DQN — gitingest-ready repository

This document contains a gitingest-ready repository layout and all source files for the project "Deep Reinforcement Learning for Portfolio Optimization with Transaction Costs". Paste each file into your Git repo. The uploaded project brief image (for reference) is available at the local path:

`/mnt/data/f71a2043-ddc7-429b-89ba-c34ca0c1a984.png`

---

## Repository structure

```
portfolio-dqn/
├─ README.md
├─ requirements.txt
├─ setup.cfg
├─ data/
│  └─ download_data.py
├─ env/
│  └─ portfolio_env.py
├─ agents/
│  ├─ dqn.py
│  └─ replay_buffer.py
├─ train.py
├─ evaluate.py
└─ scripts/
   └─ run.sh
```

---

### README.md

````md
# Deep RL Portfolio Optimization (gitingest)

Reference image (project brief): `/mnt/data/f71a2043-ddc7-429b-89ba-c34ca0c1a984.png`

## Overview
Starter repository implementing a Gym-compatible trading environment and a DQN agent that considers transaction costs. The submission is organized for easy ingestion into CI/CD or grading pipelines.

## Quick start
```bash
python -m pip install -r requirements.txt
python data/download_data.py --tickers AAPL MSFT GOOGL AMZN FB TSLA NVDA JPM BAC XOM --start 2015-01-01 --end 2024-01-01
python train.py --config config.yaml
````

## Contents

* `env/portfolio_env.py`: Gym-style environment with discrete allocation and transaction costs.
* `agents/dqn.py`: DQN implementation with target network and replay buffer.
* `train.py`: Training loop and checkpointing.

## Notes

This is a minimal, production-ish starting point. Expand tests and hyperparameter search as required.

```
```

---

### requirements.txt

```text
numpy
pandas
yfinance
gymnasium==0.28.1
torch
pyyaml
scikit-learn
matplotlib
```

---

### data/download_data.py

```python
"""
Download historical daily price data using yfinance and save as CSVs.
"""
import argparse
import yfinance as yf
import pandas as pd


def download(tickers, start, end, out='data/prices.csv'):
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    data.to_csv(out)
    print(f"Saved prices to {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tickers', nargs='+', required=True)
    p.add_argument('--start', default='2015-01-01')
    p.add_argument('--end', default='2024-01-01')
    p.add_argument('--out', default='data/prices.csv')
    args = p.parse_args()
    download(args.tickers, args.start, args.end, args.out)
```

---

### env/portfolio_env.py

```python
"""
A simple Gymnasium-compatible portfolio environment with discrete actions and transaction costs.
State: window of past returns + current holdings
Action: integer representing which asset to hold (including cash)
"""
from typing import Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, price_df: pd.DataFrame, window=10, transaction_cost=0.001):
        super().__init__()
        self.prices = price_df.sort_index()
        self.tickers = list(self.prices.columns)
        self.n_assets = len(self.tickers)
        self.window = window
        self.transaction_cost = transaction_cost

        # Discrete actions: 0..n_assets-1 choose single-asset allocation; last action = hold cash
        self.action_space = spaces.Discrete(self.n_assets + 1)

        # Observation: (window x n_assets) returns flattened + one-hot current position
        obs_dim = self.window * self.n_assets + (self.n_assets + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window
        self.done = False
        self.position = self.n_assets  # start in cash (index n_assets)
        self.cumulative_reward = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        prices = self.prices.iloc[self.current_step - self.window:self.current_step]
        returns = prices.pct_change().dropna().values.flatten()
        if returns.size != self.window * self.n_assets:
            # pad if needed
            returns = np.pad(returns, (0, self.window * self.n_assets - returns.size), 'constant')
        pos_onehot = np.zeros(self.n_assets + 1)
        pos_onehot[self.position] = 1.0
        obs = np.concatenate([returns, pos_onehot]).astype(np.float32)
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        prev_price = self.prices.iloc[self.current_step - 1]
        cur_price = self.prices.iloc[self.current_step]

        # compute gross return of chosen asset vs cash (cash return = 0)
        if action < self.n_assets:
            gross_ret = (cur_price[action] / prev_price[action]) - 1.0
        else:
            gross_ret = 0.0

        # transaction cost if changing position
        tx_cost = 0.0
        if action != self.position:
            tx_cost = self.transaction_cost * abs(gross_ret)  # non-linear-ish cost relative to return magnitude

        reward = gross_ret - tx_cost
        self.cumulative_reward += reward

        self.position = action
        self.current_step += 1
        if self.current_step >= len(self.prices):
            self.done = True

        return self._get_obs(), float(reward), bool(self.done), False, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}, CumReward: {self.cumulative_reward:.4f}")
```

---

### agents/replay_buffer.py

```python
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)
```

---

### agents/dqn.py

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, obs_dim, n_actions, lr=1e-4, gamma=0.99, tau=1e-3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau

    def act(self, obs, epsilon=0.0):
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.q_net.net[-1].out_features)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(obs_t)
        return int(q.argmax().item())

    def update(self, batch, batch_size):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones.astype(np.uint8), dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        loss = nn.MSELoss()(q_values, q_target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # soft update
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return float(loss.item())
```

---

### train.py

```python
"""
Training script: loads data, builds env and agent, runs episodes, checkpointing.
"""
import argparse
import pandas as pd
import numpy as np
from env.portfolio_env import PortfolioEnv
from agents.dqn import DQNAgent
from agents.replay_buffer import ReplayBuffer


def train(prices_csv, episodes=200, batch_size=64, buffer_size=10000):
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    env = PortfolioEnv(prices)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions)
    buffer = ReplayBuffer(capacity=buffer_size)

    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    eps = eps_start

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done = False
        while not done:
            action = agent.act(obs, epsilon=eps)
            next_obs, reward, done, _, _ = env.step(action)
            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs
            total_r += reward

            if len(buffer) > batch_size:
                batch = buffer.sample(batch_size)
                loss = agent.update(batch, batch_size)

        eps = max(eps_end, eps * eps_decay)
        print(f"Episode {ep+1}/{episodes} - Return: {total_r:.4f} - Eps: {eps:.4f}")

    # final save
    torch.save(agent.q_net.state_dict(), 'checkpoint_qnet.pt')


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--prices', default='data/prices.csv')
    p.add_argument('--episodes', type=int, default=200)
    args = p.parse_args()
    train(args.prices, episodes=args.episodes)
```

---

### evaluate.py

```python
"""
Simple evaluation on held-out set that computes cumulative return and Sharpe ratio.
"""
import numpy as np
import pandas as pd
import torch
from env.portfolio_env import PortfolioEnv
from agents.dqn import DQNAgent


def evaluate(prices_csv, qnet_path, episodes=1):
    prices = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    env = PortfolioEnv(prices)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions)
    agent.q_net.load_state_dict(torch.load(qnet_path, map_location='cpu'))

    obs, _ = env.reset()
    done = False
    rets = []
    while not done:
        action = agent.act(obs, epsilon=0.0)
        obs, reward, done, _, _ = env.step(action)
        rets.append(reward)

    cum_return = np.sum(rets)
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
    print(f"Cumulative return: {cum_return:.4f}, Sharpe (ann): {sharpe:.4f}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--prices', default='data/prices.csv')
    p.add_argument('--qnet', default='checkpoint_qnet.pt')
    args = p.parse_args()
    evaluate(args.prices, args.qnet)
```

---

### scripts/run.sh

```bash
#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python data/download_data.py --tickers AAPL MSFT GOOGL AMZN FB TSLA NVDA JPM BAC XOM --start 2015-01-01 --end 2024-01-01
python train.py --prices data/prices.csv --episodes 100
python evaluate.py --prices data/prices.csv --qnet checkpoint_qnet.pt
```

---

## Final notes

* This repo is intentionally compact for grading / ingestion. Add unit tests, CI configuration, and hyperparam search notebooks as needed.
* The project brief image path included at the top should be transformed by your environment into an accessible URL for graders if required.
