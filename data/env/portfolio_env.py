"""
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
self.position = self.n_assets # start in cash (index n_assets)
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
tx_cost = self.transaction_cost * abs(gross_ret) # non-linear-ish cost relative to return magnitude


reward = gross_ret - tx_cost
self.cumulative_reward += reward


self.position = action
self.current_step += 1
if self.current_step >= len(self.prices):
self.done = True


return self._get_obs(), float(reward), bool(self.done), False, {}


def render(self, mode='human'):
print(f"Step: {self.current_step}, Position: {self.position}, CumReward: {self.cumulative_reward:.4f}")
