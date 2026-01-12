import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from collections import deque
import pickle
from datetime import datetime

# ======================== Replay Buffer =========================
class ReplayBuffer:
    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(np.array(d)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)

# ======================== Actor (tanh-squashed Gaussian) =========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)
        nn.init.constant_(self.log_std.bias, -2.0)

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = self.backbone(state)
        mu = self.mu(x)  # no tanh here
        log_std = self.log_std(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()
        a = torch.tanh(u)

        logp = dist.log_prob(u).sum(-1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp

    def get_action(self, state):
        mu, _ = self.forward(state)
        return torch.tanh(mu)

# ======================== Critic =========================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)

# ======================== CQL-SAC Agent =========================
class CQLAgent:
    """
    Offline CQL-SAC (continuous actions).
    CQL regularizer: logsumexp(Q(s,a)) over sampled actions - Q(s,a_data)
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device="cpu",
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.05,              # SAC entropy temperature (fixed; offline建议先固定)
        auto_tune_alpha=False,
        target_entropy_coef=0.1,

        actor_hidden_dim=256,
        critic_hidden_dim=256,

        # CQL settings
        cql_alpha=5.0,           # CQL强度：1~10 常见；不稳就往上
        cql_n_random=10,         # 每个state采样多少个random action
        cql_importance_sample=True,  # 使用 policy action 的 logp 做校正（更接近论文）

        # Optional BC (super helpful offline)
        bc_weight=0.0,           # 0.1~1.0 常见；纯CQL也能跑，建议先 0.1
        q_grad_clip=10.0
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = float(alpha)
        self.auto_tune_alpha = auto_tune_alpha
        self.q_grad_clip = q_grad_clip

        self.action_dim = action_dim
        self.cql_alpha = float(cql_alpha)
        self.cql_n_random = int(cql_n_random)
        self.cql_importance_sample = bool(cql_importance_sample)

        self.bc_weight = float(bc_weight)

        self.actor = Actor(state_dim, action_dim, hidden_dim=actor_hidden_dim).to(device)

        self.q1 = QNetwork(state_dim, action_dim, hidden_dim=critic_hidden_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim=critic_hidden_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dim=critic_hidden_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dim=critic_hidden_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q_optim = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # optional auto alpha
        if auto_tune_alpha:
            self.target_entropy = -action_dim * target_entropy_coef
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = float(self.log_alpha.exp().item())
            print(f"[CQL] auto alpha ON | target_entropy={self.target_entropy:.3f}")
        else:
            self.log_alpha = None
            self.alpha_optim = None
            self.target_entropy = None
            print(f"[CQL] auto alpha OFF | alpha={self.alpha:.4f}")

        self.step = 0

    @torch.no_grad()
    def act(self, state, deterministic=False):
        self.actor.eval()
        s = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if deterministic:
            a = self.actor.get_action(s)
        else:
            a, _ = self.actor.sample(s)
        self.actor.train()
        return a.cpu().numpy().flatten()

    def _soft_update(self, src, tgt):
        for p, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _cql_regularizer(self, states, q_net, q_data):
        """
        Compute CQL regularizer for one critic:
          cql = logsumexp(Q(s, a_samples)) - Q(s, a_data)
        Samples include:
          - random uniform actions in [-1,1]
          - current policy actions
        """
        B = states.shape[0]
        A = self.action_dim
        N = self.cql_n_random

        # random actions: (B*N, A)
        rand_actions = torch.empty(B * N, A, device=self.device).uniform_(-1.0, 1.0)
        states_rep = states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        q_rand = q_net(states_rep, rand_actions).view(B, N)  # (B,N)

        # policy actions: sample N actions per state
        states_rep2 = states.unsqueeze(1).repeat(1, N, 1).view(B * N, -1)
        pol_actions, pol_logp = self.actor.sample(states_rep2)  # (B*N,A), (B*N,1)
        q_pol = q_net(states_rep2, pol_actions).view(B, N)      # (B,N)
        pol_logp = pol_logp.view(B, N)                          # (B,N)

        # combine
        if self.cql_importance_sample:
            # subtract logp to approximate sampling correction
            cat_q = torch.cat([q_rand, q_pol - pol_logp], dim=1)  # (B,2N)
        else:
            cat_q = torch.cat([q_rand, q_pol], dim=1)

        # logsumexp over samples
        lse = torch.logsumexp(cat_q, dim=1, keepdim=True)  # (B,1)

        # regularizer
        cql = (lse - q_data).mean()
        return cql

    def update(self, replay_buffer, batch_size=256, reward_scale=1.0):
        if len(replay_buffer) < batch_size:
            return {}

        s, a, r, s2, d = replay_buffer.sample(batch_size)
        s = s.to(self.device)
        a = a.to(self.device)
        r = (r * float(reward_scale)).to(self.device)
        s2 = s2.to(self.device)
        d = d.to(self.device)

        # ---------- target Q ----------
        with torch.no_grad():
            a2, logp2 = self.actor.sample(s2)
            q1n = self.q1_target(s2, a2)
            q2n = self.q2_target(s2, a2)
            qn = torch.min(q1n, q2n)
            y = r + (1.0 - d) * self.gamma * (qn - self.alpha * logp2)

        # ---------- critic losses ----------
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        bellman1 = nn.MSELoss()(q1, y)
        bellman2 = nn.MSELoss()(q2, y)

        # CQL regularizer
        cql1 = self._cql_regularizer(s, self.q1, q1)
        cql2 = self._cql_regularizer(s, self.q2, q2)

        q_loss = bellman1 + bellman2 + self.cql_alpha * (cql1 + cql2)

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.q_grad_clip)
        self.q_optim.step()

        # ---------- actor loss ----------
        new_a, logp = self.actor.sample(s)
        q1_new = self.q1(s, new_a)
        q2_new = self.q2(s, new_a)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * logp - q_new).mean()

        # Optional BC term (recommended for offline)
        if self.bc_weight > 0:
            bc_loss = (new_a - a).pow(2).mean()
            actor_loss = actor_loss + self.bc_weight * bc_loss
        else:
            bc_loss = torch.tensor(0.0, device=self.device)

        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optim.step()

        # ---------- alpha (optional) ----------
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_tune_alpha:
            alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = float(self.log_alpha.exp().item())

        # ---------- target update ----------
        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        self.step += 1
        return {
            "bellman1": float(bellman1.item()),
            "bellman2": float(bellman2.item()),
            "cql1": float(cql1.item()),
            "cql2": float(cql2.item()),
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "alpha": float(self.alpha),
            "alpha_loss": float(alpha_loss.item()) if self.auto_tune_alpha else 0.0,
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
        }

    def save(self, path):
        payload = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "alpha": self.alpha,
            "step": self.step,
        }
        if self.auto_tune_alpha:
            payload["log_alpha"] = self.log_alpha.detach().cpu()
        torch.save(payload, path)

# ======================== Data Loading (with robust states concat) =========================
def load_offline_data(data_root, goal_pos, scaler=None, capacity=1_000_000):
    rb = ReplayBuffer(capacity=capacity)
    raw_list = []

    episode_dirs = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])
    print(f"Found {len(episode_dirs)} episode directories")

    goal_pos = np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)

    for ep_name in episode_dirs:
        ep_dir = os.path.join(data_root, ep_name)
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files:
            continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)
        obs_raw = np.asarray(data["obs"], dtype=np.float32)
        rewards = np.asarray(data["reward"], dtype=np.float32)

        states = np.asarray(data["states"])
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        elif states.ndim > 2:
            states = states.reshape(states.shape[0], -1)

        T = min(obs_raw.shape[0], rewards.shape[0], states.shape[0])
        if T <= 2:
            continue

        obs_raw = obs_raw[:T]
        rewards = rewards[:T]
        states = states[:T]

        goal_block = np.repeat(goal_pos, T, axis=0)          # (T,3)
        bread_pos = obs_raw[:, 23:26]                        # (T,3)
        goal_to_bread = goal_block - bread_pos               # (T,3)

        obs_enhanced = np.concatenate([obs_raw, goal_block, goal_to_bread, states], axis=1)

        # Actions
        action_infos = data["action_infos"]
        actions = [np.asarray(ai["actions"], dtype=np.float32) for ai in action_infos]

        T_trans = min(len(actions), obs_enhanced.shape[0] - 1, rewards.shape[0] - 1)
        if T_trans <= 1:
            continue

        for t in range(T_trans):
            s = obs_enhanced[t]
            s2 = obs_enhanced[t + 1]
            a = actions[t]
            r = rewards[t]
            done = float(t == T_trans - 1)  # 如果你有真实done字段，强烈建议换成真实done

            raw_list.append(s)
            rb.add(s, a, float(r), s2, done)

    if len(rb) == 0:
        raise RuntimeError("Loaded 0 transitions. Check data_root and file format.")

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(np.array(raw_list, dtype=np.float32))

    print("Applying state normalization...")
    rb2 = ReplayBuffer(capacity=rb.capacity)
    for s, a, r, s2, d in rb.buffer:
        s = scaler.transform(s.reshape(1, -1)).astype(np.float32).flatten()
        s2 = scaler.transform(s2.reshape(1, -1)).astype(np.float32).flatten()
        rb2.add(s, a, r, s2, d)

    print(f"Loaded {len(rb2)} transitions | state_dim={rb2.buffer[0][0].shape[0]}")
    return rb2, scaler

# ======================== Main =========================
def main():
    DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05RL"
    GOAL_POS = np.array([0.197, 0.159, 0.938], dtype=np.float32)

    BATCH = 256
    STEPS = 80_000
    LOG_FREQ = 1000
    SAVE_FREQ = 10_000

    # Offline CQL defaults (稳健起点)
    LR = 1e-4
    GAMMA = 0.99
    TAU = 0.005
    ALPHA = 0.05
    AUTO_ALPHA = False

    CQL_ALPHA = 5.0        # 不稳就 10.0
    CQL_N_RANDOM = 10

    BC_WEIGHT = 0.1        # 离线强烈建议先开：0.1~1.0

    REWARD_SCALE = 1.0     # Q 爆炸就试 0.1；Q 太小试 10
    ACTOR_H = 256
    CRITIC_H = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rb, scaler = load_offline_data(DATA_ROOT, GOAL_POS)

    s, a, _, _, _ = rb.sample(1)
    state_dim = s.shape[1]
    action_dim = a.shape[1]
    print("state_dim:", state_dim, "action_dim:", action_dim)

    agent = CQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        lr=LR,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        auto_tune_alpha=AUTO_ALPHA,
        actor_hidden_dim=ACTOR_H,
        critic_hidden_dim=CRITIC_H,
        cql_alpha=CQL_ALPHA,
        cql_n_random=CQL_N_RANDOM,
        cql_importance_sample=True,
        bc_weight=BC_WEIGHT,
        q_grad_clip=10.0
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model/cql_offline_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    log_path = os.path.join(save_dir, "training_log.txt")
    log_file = open(log_path, "w", buffering=1)
    log_file.write(f"DATA_ROOT={DATA_ROOT}\nsteps={STEPS} batch={BATCH}\n")
    log_file.write(f"LR={LR} gamma={GAMMA} tau={TAU} alpha={ALPHA} auto_alpha={AUTO_ALPHA}\n")
    log_file.write(f"CQL_ALPHA={CQL_ALPHA} CQL_N_RANDOM={CQL_N_RANDOM} BC_WEIGHT={BC_WEIGHT}\n")
    log_file.write(f"REWARD_SCALE={REWARD_SCALE}\nstate_dim={state_dim} action_dim={action_dim}\n\n")

    stats = {k: [] for k in ["bellman1","bellman2","cql1","cql2","q_loss","actor_loss","bc_loss","alpha","q1_mean"]}

    for step in range(1, STEPS + 1):
        m = agent.update(rb, batch_size=BATCH, reward_scale=REWARD_SCALE)
        for k in stats:
            if k in m:
                stats[k].append(m[k])

        if step % LOG_FREQ == 0 and len(stats["q_loss"]) >= LOG_FREQ:
            avg = {k: float(np.mean(v[-LOG_FREQ:])) for k, v in stats.items() if len(v) > 0}
            msg = (
                f"Step {step:6d}/{STEPS} | "
                f"Bellman({avg['bellman1']:.3f},{avg['bellman2']:.3f}) | "
                f"CQL({avg['cql1']:.3f},{avg['cql2']:.3f}) | "
                f"Qloss {avg['q_loss']:.3f} | "
                f"Actor {avg['actor_loss']:.3f} | BC {avg['bc_loss']:.3f} | "
                f"Alpha {avg['alpha']:.3f} | Qmean {avg['q1_mean']:.2f}"
            )
            print(msg)
            log_file.write(msg + "\n")

        if step % SAVE_FREQ == 0:
            ckpt = os.path.join(save_dir, f"checkpoint_{step}.pt")
            agent.save(ckpt)
            print("✓ saved:", ckpt)

    final = os.path.join(save_dir, "final_model.pt")
    agent.save(final)
    np.savez(os.path.join(save_dir, "training_stats.npz"), **{k: np.array(v) for k, v in stats.items()})
    log_file.close()
    print("✓ done. saved to:", save_dir)

if __name__ == "__main__":
    main()
