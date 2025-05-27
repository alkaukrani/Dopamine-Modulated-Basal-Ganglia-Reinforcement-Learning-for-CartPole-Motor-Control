import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hyperparameters
BATCH_SIZE    = 64
GAMMA         = 0.99
EPSILON_START = 1.0
EPSILON_END   = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
MEMORY_SIZE   = 10000
TARGET_UPDATE = 10
EPISODES      = 500

# Dopamine sweep: Parkinson‑like, normal, high
dopamine_levels = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 1e2, 1e4, 1e6]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basal ganglia model with log‑normalized DA

class BasalGangliaModel:
    def __init__(self, dopamine, n_actions):
        # map log10(DA) from [-6,6] into [0,1]
        logd     = np.log10(dopamine)
        dop_norm = np.clip((logd + 6.0) / 12.0, 0.0, 1.0)
        self.dop_norm = dop_norm
        
        # pathway weights
        self.direct_w      = np.ones(n_actions) * 1.0
        self.indirect_w    = np.ones(n_actions) * 0.2
        self.hyperdirect_w = np.ones(n_actions) * 0.1
        
        # amplified modulation strength
        self.mod_direct   = 1.0 + 1.5 * dop_norm
        self.mod_indirect = 1.0 - 1.5 * dop_norm

    def process(self, q_values):
        q = np.array(q_values)
        go   = q * self.direct_w      * self.mod_direct
        nogo = (1.0 - q) * self.indirect_w * self.mod_indirect
        hdo  = self.hyperdirect_w
        net  = go - nogo - hdo
        expn = np.exp(net - np.max(net))
        return expn / np.sum(expn)


# 2) DQN network
def init_linear(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

class DQN(nn.Module):
    def __init__(self, in_sz, out_sz):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_sz)
        self.apply(init_linear)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Agent integrating BG model + DQN

class BasalGangliaDQNAgent:
    def __init__(self, state_size, action_size, dopamine):
        self.state_size  = state_size
        self.action_size = action_size
        
        # nonlinear ε vs. DA
        logd = np.log10(dopamine)
        norm = np.clip((logd + 6.0) / 12.0, 0.0, 1.0)
        self.epsilon = 0.05 + 0.9 * (norm ** 1.5)
        self.epsilon_decay = EPSILON_DECAY
        
        self.bg = BasalGangliaModel(dopamine, action_size)
        
        # DQN networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory    = deque(maxlen=MEMORY_SIZE)
        self.explore_ct = 0
        self.exploit_ct = 0

    def act(self, state):
        if random.random() < self.epsilon:
            self.explore_ct += 1
            return random.randrange(self.action_size)
        else:
            self.exploit_ct += 1
        with torch.no_grad():
            qv = self.policy_net(torch.FloatTensor(state).unsqueeze(0).to(device)).cpu().numpy()[0]
        probs = self.bg.process(qv)
        return int(np.argmax(probs))

    def remember(self, s,a,r,ns,d):
        self.memory.append((s,a,r,ns,d))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        s,a,r,ns,d = zip(*batch)
        s  = torch.FloatTensor(s).to(device)
        a  = torch.LongTensor(a).unsqueeze(1).to(device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(device)
        ns = torch.FloatTensor(ns).to(device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(device)
        
        curr_q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].detach().unsqueeze(1)
        targ_q = r + (1 - d) * GAMMA * next_q
        
        loss = nn.MSELoss()(curr_q, targ_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(EPSILON_END, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Motor control metrics

def calculate_smoothness(seq):
    if len(seq) < 3: return np.nan
    cart = np.array([s[0] for s in seq])
    pole = np.array([s[2] for s in seq])
    return np.var(np.diff(cart,3)) + np.var(np.diff(pole,3))

def calculate_stability(seq):
    if not seq: return np.nan
    return np.var([s[2] for s in seq])


# Single-run training function

def train_agent(dopamine):
    env = gym.make('CartPole-v1')
    agent = BasalGangliaDQNAgent(env.observation_space.shape[0],
                                 env.action_space.n,
                                 dopamine)
    rewards, mov_avg, smooths, stabs = [], [], [], []
    solve_ep = EPISODES

    for ep in range(EPISODES):
        state,_= env.reset()
        seq, total = [], 0
        for _ in range(500):
            a = agent.act(state)
            ns, r, done, trunc, _ = env.step(a)
            agent.remember(state,a,r,ns,done or trunc)
            state, total = ns, total + r
            seq.append(ns)
            if done or trunc:
                break

        agent.replay()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        rewards.append(total)
        window = rewards[-10:] if ep>=9 else rewards
        mov_avg.append(np.mean(window))
        if solve_ep==EPISODES and mov_avg[-1]>=195:
            solve_ep = ep

        smooths.append(calculate_smoothness(seq))
        stabs .append(calculate_stability(seq))

    env.close()
    return {
        'moving_avg'   : np.array(mov_avg),
        'solve_eps'    : solve_ep,
        'smoothness'   : np.nanmean(smooths[-50:]),
        'stability'    : np.nanmean(stabs[-50:]),
        'explore_ratio': agent.explore_ct / (agent.explore_ct + agent.exploit_ct)
    }


#Fixed plotting: five separate panels

def plot_all(results):
    dvals = sorted(results.keys())
    # 1) Learning curves
    plt.figure(figsize=(8,5))
    for da in dvals:
        plt.plot(results[da]['moving_avg'], label=f'DA={da:.0e}', alpha=0.6)
    plt.title("Learning Curves vs Dopamine")
    plt.xlabel("Episode")
    plt.ylabel("Moving Avg Reward (10-ep window)")
    plt.legend(bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.show()

    # 2) Episodes to Solve
    solve_eps = [results[da]['solve_eps'] for da in dvals]
    plt.figure(figsize=(6,4))
    plt.plot(dvals, solve_eps, 'o-')
    plt.xscale('log')
    plt.title("Episodes to First Solve vs Dopamine")
    plt.xlabel("Dopamine (log scale)")
    plt.ylabel("First Episode ≥195 Reward")
    plt.tight_layout()
    plt.show()

    # 3) Stability
    stab = [results[da]['stability'] for da in dvals]
    plt.figure(figsize=(6,4))
    plt.plot(dvals, stab, 'o-')
    plt.xscale('log')
    plt.title("Stability (Pole Variance) vs Dopamine")
    plt.xlabel("Dopamine (log scale)")
    plt.ylabel("Pole‑Angle Variance")
    plt.tight_layout()
    plt.show()

    # 4) Smoothness
    smo = [results[da]['smoothness'] for da in dvals]
    plt.figure(figsize=(6,4))
    plt.plot(dvals, smo, 'o-')
    plt.xscale('log')
    plt.title("Smoothness (Jerk Variance) vs Dopamine")
    plt.xlabel("Dopamine (log scale)")
    plt.ylabel("Jerk Variance")
    plt.tight_layout()
    plt.show()

    # 5) Exploration Ratio
    exp_r = [results[da]['explore_ratio'] for da in dvals]
    plt.figure(figsize=(6,4))
    plt.plot(dvals, exp_r, 'o-')
    plt.xscale('log')
    plt.title("Exploration Ratio vs Dopamine")
    plt.xlabel("Dopamine (log scale)")
    plt.ylabel("Exploration / Total Actions")
    plt.tight_layout()
    plt.show()


# execution

if __name__ == '__main__':
    all_results = {}
    for da in dopamine_levels:
        print(f'Running DA={da:.0e}')
        all_results[da] = train_agent(da)
    plot_all(all_results)
