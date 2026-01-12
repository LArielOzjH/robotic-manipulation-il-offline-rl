import argparse
import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

import BCQ
import utils


def ensure_2d(x: np.ndarray) -> np.ndarray:
	x = np.asarray(x)
	if x.ndim == 1:
		return x.reshape(-1, 1)
	if x.ndim > 2:
		return x.reshape(x.shape[0], -1)
	return x


def infer_dims_from_one_npz(data_root: str, use_states: bool) -> tuple[int, int, int]:

	ep_dirs = sorted([
		os.path.join(data_root, d)
		for d in os.listdir(data_root)
		if os.path.isdir(os.path.join(data_root, d))
	])
	if len(ep_dirs) == 0:
		raise RuntimeError("No episode directories found under data_root.")

	npz_files = [f for f in os.listdir(ep_dirs[0]) if f.endswith(".npz")]
	if len(npz_files) == 0:
		raise RuntimeError("No .npz file found in the first episode directory.")

	data = np.load(os.path.join(ep_dirs[0], npz_files[0]), allow_pickle=True)
	obs_raw = np.asarray(data["obs"], dtype=np.float32)
	obs_dim = obs_raw.shape[1]

	ai = data["action_infos"]
	action_dim = np.asarray(ai[0]["actions"], dtype=np.float32).shape[0]
	obs_dim = 20
	state_dim = obs_dim + 3 + 3
	if use_states:
		st = ensure_2d(np.asarray(data["states"], dtype=np.float32))
		state_dim += st.shape[1]

	return state_dim, action_dim, obs_dim


def load_npz_dataset_to_buffer(
	replay_buffer: utils.ReplayBuffer,
	data_root: str,
	goal_pos: np.ndarray,
	use_states: bool = True,
	print_every_ep: int = 20
):

	goal_pos = np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)

	episode_dirs = sorted([
		os.path.join(data_root, d)
		for d in os.listdir(data_root)
		if os.path.isdir(os.path.join(data_root, d))
	])

	total_added = 0
	all_rewards = []
	all_actions_min = None
	all_actions_max = None
	done_count = 0

	print(f"[Data] Found {len(episode_dirs)} episode dirs under {data_root}")

	for epi, ep_dir in enumerate(episode_dirs, start=1):
		npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
		if not npz_files:
			continue

		data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)

		obs_raw = np.asarray(data["obs"], dtype=np.float32)
		rewards = np.asarray(data["reward"], dtype=np.float32).reshape(-1)
		T_obs = obs_raw.shape[0]

		action_infos = data["action_infos"]
		actions = np.stack([np.asarray(ai["actions"], dtype=np.float32) for ai in action_infos], axis=0)

		states = None
		if use_states:
			states = ensure_2d(np.asarray(data["states"], dtype=np.float32))

		T_feat = min(
			T_obs,
			len(rewards),
			states.shape[0] if states is not None else T_obs
		)
		if T_feat <= 2:
			continue

		obs_raw = obs_raw[:T_feat]
		rewards = rewards[:T_feat]
		if states is not None:
			states = states[:T_feat]

		goal_block = np.repeat(goal_pos, T_feat, axis=0)
		bread_pos = obs_raw[:, 23:26]
		eef_pos = obs_raw[:, 14:17]
		eef_to_bread = eef_pos - bread_pos
		goal_to_bread = goal_block - bread_pos
		obs_enhanced = obs_raw[:, 14:]
		obs_enhanced = np.concatenate([obs_enhanced[:, :7], obs_enhanced[:, 9:]], axis=1)	
		
		obs_enhanced = np.concatenate([obs_enhanced, goal_block, goal_to_bread, eef_to_bread], axis=1)
		if use_states:
			obs_enhanced = np.concatenate([obs_enhanced, states], axis=1)

		T_trans = min(actions.shape[0], obs_enhanced.shape[0] - 1, rewards.shape[0] - 1)
		if T_trans <= 1:
			continue

		r_seg = rewards[:T_trans]
		a_seg = actions[:T_trans]

		a_min = a_seg.min(axis=0)
		a_max = a_seg.max(axis=0)
		all_actions_min = a_min if all_actions_min is None else np.minimum(all_actions_min, a_min)
		all_actions_max = a_max if all_actions_max is None else np.maximum(all_actions_max, a_max)

		for t in range(T_trans):
			s = obs_enhanced[t]
			a = actions[t]
			s2 = obs_enhanced[t + 1]
			r = float(rewards[t])
			done = 1.0 if (t == T_trans - 1) else 0.0   

			replay_buffer.add(s, a, s2, r, done)
			total_added += 1
			done_count += int(done)
			all_rewards.append(r)

		if epi % print_every_ep == 0:
			print(
				f"[Data] ep {epi:4d}/{len(episode_dirs)} | "
				f"T_obs={T_obs}, T_feat={T_feat}, T_trans={T_trans} | "
				f"reward(mean={r_seg.mean():.3f}, min={r_seg.min():.3f}, max={r_seg.max():.3f}) | "
				f"action(min={a_min.min():.3f}, max={a_max.max():.3f}) | "
				f"state_dim={obs_enhanced.shape[1]}"
			)

	all_rewards = np.asarray(all_rewards, dtype=np.float32)
	print("=" * 80)
	print(f"[Data] Total transitions added: {total_added}")
	print(f"[Data] Done count: {done_count} | done ratio ~ {done_count / max(total_added,1):.6f}")
	if all_actions_min is not None:
		print(f"[Data] Action global range: min={all_actions_min.min():.3f}, max={all_actions_max.max():.3f}")
	if total_added > 0:
		print(f"[Data] Reward global: mean={all_rewards.mean():.4f}, std={all_rewards.std():.4f}, min={all_rewards.min():.4f}, max={all_rewards.max():.4f}")
	print("=" * 80)
@torch.no_grad()
def eval_bcq_signals(policy, replay_buffer, batch_size: int, device):
	policy.actor.eval()
	policy.critic.eval()
	policy.vae.eval()

	state, action, next_state, reward, done = replay_buffer.sample(batch_size)

	q1, q2 = policy.critic(state, action)
	q_min_mean = torch.min(q1, q2).mean().item()

	recon, mean, std = policy.vae(state, action)
	vae_recon_mse = ((recon - action) ** 2).mean().item()

	try:
		a_cand = policy.vae.decode(state)  # [B, act_dim]
	except TypeError:
		a_cand = recon

	pi = policy.actor(state, a_cand)

	pi_abs_mean = pi.abs().mean().item()
	pi_l2 = (pi ** 2).mean().sqrt().item()

	q1_pi, q2_pi = policy.critic(state, pi)
	q_pi_min_mean = torch.min(q1_pi, q2_pi).mean().item()

	return {
		"q_min_mean": q_min_mean,
		"q_pi_min_mean": q_pi_min_mean,
		"vae_recon_mse": vae_recon_mse,
		"pi_abs_mean": pi_abs_mean,
		"pi_l2": pi_l2,
	}

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_root", type=str, default="../PoS0_05RL")
	parser.add_argument("--goal_pos", type=float, nargs=3, default=(0.197, 0.159, 0.938))

	parser.add_argument("--use_states", action="store_true")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--device", type=str, default="cuda")

	# train
	parser.add_argument("--max_updates", type=int, default=200000)
	parser.add_argument("--eval_freq", type=int, default=5000)    
	parser.add_argument("--batch_size", type=int, default=100)
	parser.add_argument("--log_every", type=int, default=1000)    
	parser.add_argument("--print_every_ep", type=int, default=20)

	# BCQ hyperparams
	parser.add_argument("--discount", type=float, default=0.99)
	parser.add_argument("--tau", type=float, default=0.005)
	parser.add_argument("--lmbda", type=float, default=0.75)
	parser.add_argument("--phi", type=float, default=0.05)

	# buffer size
	parser.add_argument("--max_size", type=int, default=int(2e6))

	args = parser.parse_args()

	# seeds
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	device = torch.device(args.device if (torch.cuda.is_available() and args.device.startswith("cuda")) else "cpu")
	print("[Run] device:", device)
	print("[Run] seed:", args.seed)

	# infer dims
	state_dim, action_dim, obs_dim = infer_dims_from_one_npz(args.data_root, args.use_states)
	max_action = 1.0  
	print(f"[Run] inferred obs_dim={obs_dim}, state_dim={state_dim}, action_dim={action_dim}, max_action={max_action}")

	# init policy & buffer
	policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)
	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device, max_size=args.max_size)

	# fill buffer
	load_npz_dataset_to_buffer(
		replay_buffer=replay_buffer,
		data_root=args.data_root,
		goal_pos=np.array(args.goal_pos, dtype=np.float32),
		use_states=args.use_states,
		print_every_ep=args.print_every_ep
	)
	print(f"[Run] replay_buffer.size={replay_buffer.size}")

	# save dir
	os.makedirs("../model/bcq", exist_ok=True)
	tag = f"offline_bcq_seed{args.seed}"
	save_path = f"../model/bcq/{tag}.pt"

	# ========= logging =========
	run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	log_path_npz = f"../model/bcq/{tag}_logs_{run_stamp}.npz"
	log_path_csv = f"../model/bcq/{tag}_logs_{run_stamp}.csv"
	fig_path_png = f"../model/bcq/{tag}_convergence_{run_stamp}.png"
	fig_path_pdf = f"../model/bcq/{tag}_convergence_{run_stamp}.pdf"

	log = {
		"updates": [],
		"q_min_mean": [],
		"q_pi_min_mean": [],
		"vae_recon_mse": [],
		"pi_abs_mean": [],
		"pi_l2": [],
	}



	# train loop
	updates = 0
	while updates < args.max_updates:
		print(f"\n[Run] Train chunk: {updates}->{min(updates+args.eval_freq, args.max_updates)}")
		policy.train(
			replay_buffer,
			iterations=int(args.eval_freq),
			batch_size=args.batch_size,
			log_every=args.log_every
		)
		updates += int(args.eval_freq)

		# ---- eval signals ----
		if replay_buffer.size >= args.batch_size:
			signals = eval_bcq_signals(policy, replay_buffer, args.batch_size, device)
			log["updates"].append(updates)
			for k, v in signals.items():
				log[k].append(v)

			print("[Eval] " + " | ".join([f"{k}={signals[k]:.4f}" for k in signals]))


		# checkpoint
		ckpt = f"../model/bcq/{tag}_{updates}.pt"
		torch.save({
			"actor": policy.actor.state_dict(),
			"critic": policy.critic.state_dict(),
			"vae": policy.vae.state_dict(),
		}, ckpt)
		print(f"[Run] saved: {ckpt}")

	# final
	torch.save({
		"actor": policy.actor.state_dict(),
		"critic": policy.critic.state_dict(),
		"vae": policy.vae.state_dict(),
	}, save_path)
	print(f"[Run] done. final saved: {save_path}")

	# ========= save logs =========
	# 1) npz
	np.savez(
		log_path_npz,
		**{k: np.asarray(v, dtype=np.float32) for k, v in log.items()}
	)

	# 2) csv（便于你用excel/latex pgfplots）
	with open(log_path_csv, "w") as f:
		keys = list(log.keys())
		f.write(",".join(keys) + "\n")
		for i in range(len(log["updates"])):
			row = [str(log[k][i]) for k in keys]
			f.write(",".join(row) + "\n")

	print(f"[Log] saved: {log_path_npz}")
	print(f"[Log] saved: {log_path_csv}")

	# ========= plot =========
	def moving_avg(x, w=5):
		x = np.asarray(x, dtype=np.float32)
		if len(x) < w:
			return x
		k = np.ones(w, dtype=np.float32) / w
		return np.convolve(x, k, mode="same")

	u = np.asarray(log["updates"], dtype=np.float32)
	if len(u) > 0:
		plt.figure(figsize=(9, 6))
		plt.plot(u, moving_avg(log["q_pi_min_mean"], 7), label="min Q(s, pi(s)) mean")
		plt.plot(u, moving_avg(log["vae_recon_mse"], 7), label="VAE recon MSE")
		plt.plot(u, moving_avg(log["pi_abs_mean"], 7), label="|pi(s)| mean")

		plt.xlabel("Gradient updates")
		plt.ylabel("Value")
		plt.title("BCQ training convergence (signals)")
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.tight_layout()
		plt.savefig(fig_path_png, dpi=200)
		plt.savefig(fig_path_pdf)
		plt.close()
		print(f"[Plot] saved: {fig_path_png}")
		print(f"[Plot] saved: {fig_path_pdf}")



if __name__ == "__main__":
	main()
