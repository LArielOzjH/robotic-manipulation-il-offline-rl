import os
import argparse
import pickle
from datetime import datetime
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# =========================================================
# Utils (ReplayBuffer)  — 结构对齐 sfujim/BCQ utils.py 的味道
# =========================================================
class ReplayBuffer:
    def __init__(self, capacity=2_000_000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        return (
            torch.as_tensor(np.asarray(s, dtype=np.float32)),
            torch.as_tensor(np.asarray(a, dtype=np.float32)),
            torch.as_tensor(np.asarray(r, dtype=np.float32)).unsqueeze(1),
            torch.as_tensor(np.asarray(s2, dtype=np.float32)),
            torch.as_tensor(np.asarray(d, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Offline dataset loader — 按你 npz episode 文件夹组织方式
# =========================================================
def load_offline_dataset_npz(
    data_root: str,
    goal_pos: np.ndarray,
    use_states: bool = True,
    normalize_states: bool = True,
    scaler: StandardScaler | None = None,
):
    """
    data_root/
      episode_x/
        xxx.npz

    npz expected keys:
      - obs: (T, obs_dim)
      - reward: (T,) or (T,1)
      - action_infos: array/list length T or T-1, each has ["actions"]
        OR actions: (T-1, act_dim)
      - states: (T, state_dim)  (optional)
    """
    rb_raw = ReplayBuffer()
    feats_for_scaler = []

    goal_pos = np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)

    ep_dirs = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])
    print(f"[Data] Found {len(ep_dirs)} episode directories")

    for ep_dir in ep_dirs:
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files:
            continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)

        obs_raw = np.asarray(data["obs"], dtype=np.float32)
        rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
        T_obs = obs_raw.shape[0]

        # actions
        if "action_infos" in data:
            action_infos = data["action_infos"]
            actions = np.stack([np.asarray(ai["actions"], dtype=np.float32) for ai in action_infos], axis=0)
        elif "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
        else:
            raise KeyError(f"npz missing 'action_infos' or 'actions': {os.path.join(ep_dir, npz_files[0])}")

        # optional states
        states = None
        if use_states:
            if "states" not in data:
                raise KeyError(f"use_states=True but npz missing 'states': {os.path.join(ep_dir, npz_files[0])}")
            states = np.asarray(data["states"], dtype=np.float32)
            if states.ndim == 1:
                states = states.reshape(-1, 1)
            elif states.ndim > 2:
                states = states.reshape(states.shape[0], -1)

        # align lengths
        T = min(T_obs - 1, actions.shape[0], len(rewards) - 1 if len(rewards) > 1 else actions.shape[0])
        if T <= 1:
            continue

        # feature engineering (你之前那套：obs + goal + goal_to_bread + states)
        goal_block = np.repeat(goal_pos, T_obs, axis=0)                 # (T_obs,3)
        bread_pos = obs_raw[:, 23:26]                                   # (T_obs,3) 你原本假设
        goal_to_bread = goal_block - bread_pos                          # (T_obs,3)

        obs_enhanced = np.concatenate([obs_raw, goal_block, goal_to_bread], axis=1)

        if use_states:
            # align states length to T_obs
            if states.shape[0] != T_obs:
                states = states[:T_obs]
            obs_enhanced = np.concatenate([obs_enhanced, states], axis=1)

        # push transitions
        for t in range(T):
            s = obs_enhanced[t]
            a = actions[t]
            r = rewards[t]
            s2 = obs_enhanced[t + 1]
            done = 1.0 if (t == T - 1) else 0.0   # 如果你有真实 done 字段，强烈建议替换这里

            rb_raw.add(s, a, float(r), s2, float(done))
            feats_for_scaler.append(s)

    if len(rb_raw) == 0:
        raise RuntimeError("[Data] Loaded 0 transitions. Check data_root & npz contents.")

    print(f"[Data] Loaded {len(rb_raw)} transitions (raw)")

    if not normalize_states:
        return rb_raw, None

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(np.asarray(feats_for_scaler, dtype=np.float32))

    rb = ReplayBuffer(capacity=rb_raw.capacity)
    for s, a, r, s2, d in rb_raw.buffer:
        s_n = scaler.transform(s.reshape(1, -1)).astype(np.float32).squeeze(0)
        s2_n = scaler.transform(s2.reshape(1, -1)).astype(np.float32).squeeze(0)
        rb.add(s_n, a, r, s2_n, d)

    print("[Data] Applied StandardScaler to states")
    return rb, scaler


# =========================================================
# BCQ Networks (continuous) — 结构与官方实现一致：VAE + perturb actor + twin critic
# =========================================================
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=None, hidden_dim=750, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim if latent_dim is not None else action_dim * 2
        self.max_action = float(max_action)

        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, self.latent_dim)
        self.log_std = nn.Linear(hidden_dim, self.latent_dim)

        self.d1 = nn.Linear(state_dim + self.latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.e1(x))
        x = self.relu(self.e2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        recon = self.decode(state, z)
        return recon, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim), device=state.device)
        x = torch.cat([state, z], dim=1)
        x = self.relu(self.d1(x))
        x = self.relu(self.d2(x))
        a = torch.tanh(self.d3(x)) * self.max_action
        return a


class PerturbationActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400, phi=0.05, max_action=1.0):
        super().__init__()
        self.phi = float(phi)
        self.max_action = float(max_action)
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        delta = torch.tanh(self.l3(x)) * self.phi
        return (action + delta).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super().__init__()
        # Q1
        self.q1_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.q2_l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        q1 = self.relu(self.q1_l1(sa))
        q1 = self.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = self.relu(self.q2_l1(sa))
        q2 = self.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)

        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = self.relu(self.q1_l1(sa))
        q1 = self.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)
        return q1


# =========================================================
# BCQ Agent (continuous) — 核心训练逻辑对齐官方 continuous BCQ 思路
# =========================================================
class BCQAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        lmbda=0.75,
        phi=0.05,
        n_action=10,
        lr=3e-4,
        vae_hidden=750,
        critic_hidden=400,
        actor_hidden=400,
    ):
        self.device = device
        self.action_dim = action_dim
        self.max_action = float(max_action)
        self.discount = float(discount)
        self.tau = float(tau)
        self.lmbda = float(lmbda)
        self.phi = float(phi)
        self.n_action = int(n_action)

        self.vae = VAE(state_dim, action_dim, hidden_dim=vae_hidden, max_action=max_action).to(device)
        self.actor = PerturbationActor(state_dim, action_dim, hidden_dim=actor_hidden, phi=phi, max_action=max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)

        self.actor_target = PerturbationActor(state_dim, action_dim, hidden_dim=actor_hidden, phi=phi, max_action=max_action).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim=critic_hidden).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.vae_opt = optim.Adam(self.vae.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.mse = nn.MSELoss()
        self.total_updates = 0

    def train(self, replay_buffer: ReplayBuffer, batch_size=256, reward_scale=1.0):
        if len(replay_buffer) < batch_size:
            return {}

        self.total_updates += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = (reward * float(reward_scale)).to(self.device)  # reward scaling 在这里
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        # 1) VAE update
        recon, mean, std = self.vae(state, action)
        recon_loss = self.mse(recon, action)
        kl_loss = 0.5 * torch.mean(torch.sum(
            mean.pow(2) + std.pow(2) - 2.0 * torch.log(std + 1e-6) - 1.0, dim=1
        ))
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_opt.zero_grad(set_to_none=True)
        vae_loss.backward()
        self.vae_opt.step()

        # 2) Critic target: sample n actions at s', perturb, take max over candidates
        with torch.no_grad():
            ns_rep = next_state.repeat_interleave(self.n_action, dim=0)
            cand = self.vae.decode(ns_rep)
            cand = self.actor_target(ns_rep, cand)

            tq1, tq2 = self.critic_target(ns_rep, cand)
            tq_min = torch.min(tq1, tq2)
            tq_max = torch.max(tq1, tq2)
            # weighted clipped double-Q
            tq = self.lmbda * tq_min + (1.0 - self.lmbda) * tq_max
            tq = tq.view(batch_size, self.n_action)
            max_tq, _ = torch.max(tq, dim=1, keepdim=True)

            target_q = reward + (1.0 - done) * self.discount * max_tq

        cq1, cq2 = self.critic(state, action)
        critic_loss = self.mse(cq1, target_q) + self.mse(cq2, target_q)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # 3) Actor update: sample from VAE(s), perturb, maximize Q1
        sampled = self.vae.decode(state)
        perturbed = self.actor(state, sampled)
        actor_loss = -self.critic.q1(state, perturbed).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        # 4) soft update targets
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            "vae": float(vae_loss.item()),
            "recon": float(recon_loss.item()),
            "kl": float(kl_loss.item()),
            "critic": float(critic_loss.item()),
            "actor": float(actor_loss.item()),
            "q1": float(cq1.mean().item()),
            "tq": float(target_q.mean().item()),
        }

    def _soft_update(self, net, net_targ):
        for p, pt in zip(net.parameters(), net_targ.parameters()):
            pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)

    def save(self, path: str):
        torch.save({
            "vae": self.vae.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "total_updates": self.total_updates,
        }, path)


# =========================================================
# main() — 模仿 continuous_BCQ/main.py 的 CLI / logging 风格
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="offline dataset root (episode dirs)")
    parser.add_argument("--goal_pos", type=float, nargs=3, required=True, help="goal position x y z")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # training
    parser.add_argument("--max_updates", type=int, default=200_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_freq", type=int, default=1000)
    parser.add_argument("--save_freq", type=int, default=10_000)

    # BCQ hyperparams (paper-style knobs)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--lmbda", type=float, default=0.75)
    parser.add_argument("--phi", type=float, default=0.05)
    parser.add_argument("--n_action", type=int, default=10)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reward_scale", type=float, default=1.0)

    # data
    parser.add_argument("--use_states", action="store_true", help="concat npz['states'] into observation")
    parser.add_argument("--no_norm", action="store_true", help="disable StandardScaler state normalization")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print("[Run] device:", device)

    # load dataset
    replay, scaler = load_offline_dataset_npz(
        data_root=args.data_root,
        goal_pos=np.array(args.goal_pos, dtype=np.float32),
        use_states=args.use_states,
        normalize_states=(not args.no_norm),
        scaler=None
    )

    s1, a1, _, _, _ = replay.sample(1)
    state_dim = s1.shape[1]
    action_dim = a1.shape[1]
    print(f"[Run] state_dim={state_dim}, action_dim={action_dim}")

    agent = BCQAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        max_action=1.0,
        discount=args.discount,
        tau=args.tau,
        lmbda=args.lmbda,
        phi=args.phi,
        n_action=args.n_action,
        lr=args.lr,
    )

    # save dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("model", f"bcq_offline_{ts}_seed{args.seed}")
    os.makedirs(save_dir, exist_ok=True)

    if scaler is not None:
        with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    log_path = os.path.join(save_dir, "train.log")
    log_f = open(log_path, "w", buffering=1)

    print("[Run] save_dir:", save_dir)
    log_f.write(f"data_root={args.data_root}\n")
    log_f.write(f"goal_pos={args.goal_pos}\n")
    log_f.write(f"state_dim={state_dim} action_dim={action_dim}\n")
    log_f.write(f"updates={args.max_updates} batch={args.batch_size}\n")
    log_f.write(f"discount={args.discount} tau={args.tau} lambda={args.lmbda} phi={args.phi} n={args.n_action}\n")
    log_f.write(f"lr={args.lr} reward_scale={args.reward_scale}\n")
    log_f.write(f"use_states={args.use_states} norm={not args.no_norm}\n\n")

    # train loop
    history = {k: [] for k in ["vae", "critic", "actor", "q1", "tq"]}
    for t in range(1, args.max_updates + 1):
        m = agent.train(replay, batch_size=args.batch_size, reward_scale=args.reward_scale)
        if not m:
            continue
        for k in history:
            history[k].append(m[k])

        if t % args.log_freq == 0 and len(history["critic"]) >= args.log_freq:
            msg = (
                f"Upd {t:7d}/{args.max_updates} | "
                f"VAE {np.mean(history['vae'][-args.log_freq:]):.4f} | "
                f"Critic {np.mean(history['critic'][-args.log_freq:]):.4f} | "
                f"Actor {np.mean(history['actor'][-args.log_freq:]):.4f} | "
                f"Q1 {np.mean(history['q1'][-args.log_freq:]):.2f} | "
                f"TQ {np.mean(history['tq'][-args.log_freq:]):.2f}"
            )
            print(msg)
            log_f.write(msg + "\n")

        if t % args.save_freq == 0:
            ckpt = os.path.join(save_dir, f"checkpoint_{t}.pt")
            agent.save(ckpt)
            print("[Run] saved:", ckpt)

    final_path = os.path.join(save_dir, "final.pt")
    agent.save(final_path)
    log_f.close()
    print("[Run] done. final saved:", final_path)


if __name__ == "__main__":
    main()
