import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, device, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


	def save(self, save_folder):
		np.save(f"{save_folder}_state.npy", self.state[:self.size])
		np.save(f"{save_folder}_action.npy", self.action[:self.size])
		np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
		np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
		np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
		np.save(f"{save_folder}_ptr.npy", self.ptr)


	def load(self, save_folder, size=-1):
		reward_buffer = np.load(f"{save_folder}_reward.npy")
		
		# Adjust crt_size if we're using a custom size
		size = min(int(size), self.max_size) if size > 0 else self.max_size
		self.size = min(reward_buffer.shape[0], size)

		self.state[:self.size] = np.load(f"{save_folder}_state.npy")[:self.size]
		self.action[:self.size] = np.load(f"{save_folder}_action.npy")[:self.size]
		self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy")[:self.size]
		self.reward[:self.size] = reward_buffer[:self.size]
		self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy")[:self.size]

from sklearn.preprocessing import StandardScaler
import os
import numpy as np

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim > 2:
        return x.reshape(x.shape[0], -1)
    return x

def load_from_npz_dataset(
    replay_buffer,
    data_root: str,
    goal_pos: np.ndarray,
    use_states: bool = True,
    normalize_state: bool = True,
):
    """
    Fill an existing ReplayBuffer (official style) from your offline npz dataset.

    Expected per npz:
      - obs: (T, obs_dim)
      - reward: (T,)
      - action_infos: list/array length T or T-1, each element has ["actions"]
      - states: (T, state_dim)  (optional, if use_states=True)

    Feature:
      obs_enhanced = [obs_raw, goal_block(3), goal_to_bread(3), states(optional)]
    """
    goal_pos = np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)

    all_states = []
    tmp_transitions = []

    episode_dirs = sorted([
        os.path.join(data_root, d)
        for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    print(f"[Data] Found {len(episode_dirs)} episode dirs under {data_root}")

    for ep_dir in episode_dirs:
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files:
            continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)

        obs_raw = np.asarray(data["obs"], dtype=np.float32)          # (T, obs_dim)
        rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)  # (T,)
        T_obs = obs_raw.shape[0]

        # actions
        if "action_infos" not in data:
            raise KeyError(f"{ep_dir} missing action_infos")
        action_infos = data["action_infos"]
        actions = np.stack([np.asarray(ai["actions"], dtype=np.float32) for ai in action_infos], axis=0)  # (Ta, act_dim)

        # optional states
        states = None
        if use_states:
            if "states" not in data:
                raise KeyError(f"{ep_dir} use_states=True but missing states")
            states = _ensure_2d(np.asarray(data["states"], dtype=np.float32))  # (Ts, state_dim)

        # align lengths for feature construction
        T_feat = min(T_obs, len(rewards), states.shape[0] if states is not None else T_obs)
        if T_feat <= 2:
            continue
        obs_raw = obs_raw[:T_feat]
        rewards = rewards[:T_feat]
        if states is not None:
            states = states[:T_feat]

        # feature engineering
        goal_block = np.repeat(goal_pos, T_feat, axis=0)             # (T,3)
        bread_pos = obs_raw[:, 23:26]                                 # (T,3) 你原本的索引假设
        goal_to_bread = goal_block - bread_pos                        # (T,3)

        obs_enhanced = np.concatenate([obs_raw, goal_block, goal_to_bread], axis=1)
        if use_states:
            obs_enhanced = np.concatenate([obs_enhanced, states], axis=1)

        # transitions length
        T_trans = min(actions.shape[0], obs_enhanced.shape[0] - 1, rewards.shape[0] - 1)
        if T_trans <= 1:
            continue

        for t in range(T_trans):
            s = obs_enhanced[t]
            a = actions[t]
            s2 = obs_enhanced[t + 1]
            r = rewards[t]
            done = 1.0 if (t == T_trans - 1) else 0.0  # 如果你有真实done字段，务必替换这里

            tmp_transitions.append((s, a, s2, r, done))
            all_states.append(s)

    if len(tmp_transitions) == 0:
        raise RuntimeError("[Data] Loaded 0 transitions from npz dataset.")

    print(f"[Data] Parsed transitions: {len(tmp_transitions)}")

    # normalize state (recommended)
    if normalize_state:
        scaler = StandardScaler()
        scaler.fit(np.asarray(all_states, dtype=np.float32))
        print("[Data] Fitting StandardScaler done.")

        for s, a, s2, r, done in tmp_transitions:
            s_n = scaler.transform(s.reshape(1, -1)).astype(np.float32).squeeze(0)
            s2_n = scaler.transform(s2.reshape(1, -1)).astype(np.float32).squeeze(0)
            replay_buffer.add(s_n, a, s2_n, float(r), float(done))
        return scaler
    else:
        for s, a, s2, r, done in tmp_transitions:
            replay_buffer.add(s, a, s2, float(r), float(done))
        return None
