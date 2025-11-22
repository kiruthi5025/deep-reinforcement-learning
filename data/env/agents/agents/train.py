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
