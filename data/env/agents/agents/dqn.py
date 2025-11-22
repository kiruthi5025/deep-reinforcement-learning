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
