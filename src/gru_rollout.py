##########################################
# Title: gru_rollout
# Declaration: execute the trained GRU policy
##########################################

import argparse
import time
from pathlib import Path
from collections import deque

import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper

import joblib
import torch
import torch.nn as nn
from trajectories_record import PandaPickPlace, PandaPickPlaceBread

# ---------------------------- Model Class ----------------------------
class GRUPolicy(nn.Module):
    """
    Input:  (B, T, D)
    Output: (B, action_dim)
    """
    def __init__(self, state_dim: int, action_dim: int = 7, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        _, h_n = self.gru(x)      # (num_layers, B, hidden_dim)
        h_last = h_n[-1]          # (B, hidden_dim)
        return self.head(h_last)


# ---------------------------- Config ----------------------------
goal_pos = np.array((0.197, 0.159, 0.938), dtype=np.float32)

keep_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    # "robot0_gripper_qpos",
    "Bread_pos",
    "Bread_quat",
    "Bread_to_robot0_eef_pos",
]

# MUST match training
SEQ_LEN = 7

# Paths MUST match training save dir
MODEL_DIR = Path("../model/gru_policy")
SCALER_PATH = MODEL_DIR / "state_scaler.pkl"
WEIGHTS_PATH = MODEL_DIR / "gru_policy.pt"


def get_needed_obs(obs_dict):
    return np.concatenate([obs_dict[k].flatten() for k in keep_keys]).astype(np.float32)


def build_features(obs_dict):
    """
    Must match training feature engineering:
    base = get_needed_obs()
    then append goal_to_bread and bread_to_goal (6 dims).
    """
    base = get_needed_obs(obs_dict)  # shape (D0,)

    # NOTE: positions in base depend on keep_keys order:
    # base = [eef_pos(3), eef_quat(4), bread_pos(3), bread_quat(4), bread_to_eef(3)]
    eef_pos = base[0:3]
    bread_pos = base[7:10]

    goal_to_bread = goal_pos - bread_pos          # (3,)
    bread_to_goal = bread_pos - goal_pos          # (3,)

    feat = np.concatenate([base, goal_pos, goal_to_bread], axis=0).astype(np.float32)
    return feat


def scaler_transform_sequence(scaler, seq_feats):
    """
    seq_feats: (T, D)
    StandardScaler expects (N, D)
    """
    return scaler.transform(seq_feats)


# ---------------------------- Main ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--arm", type=str, default="right")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller name / json file / None",
    )
    parser.add_argument("--max_fr", default=20, type=int)
    args = parser.parse_args()

    # ---- env ----
    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0])
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None

    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=50,
        hard_reset=False,
        z_rotation=0,
    )
    env = VisualizationWrapper(env, indicator_configs=None)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load scaler & model ----
    scaler = joblib.load(str(SCALER_PATH))

    # Infer state_dim from a dummy obs once we reset
    obs_raw = env.reset()
    first_feat = build_features(obs_raw)
    state_dim = first_feat.shape[0]
    action_dim = 7

    model = GRUPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=64, num_layers=1).to(device)
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location=device))
    model.eval()

    # ---- init queue ----
    obs_queue = deque(maxlen=SEQ_LEN)
    for _ in range(SEQ_LEN):
        obs_queue.append(first_feat.copy())

    env.render()
    done = False
    t = 0

    while not done:
        with torch.no_grad():
            seq = np.stack(list(obs_queue), axis=0)          # (T, D)
            seq = scaler_transform_sequence(scaler, seq)     # (T, D)
            seq_t = torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, D)

            action = model(seq_t).detach().cpu().numpy()[0]
            action[:6] *= 0.8  # velocity scaling (keep your choice)

        obs_raw, reward, done, _ = env.step(action)
        feat = build_features(obs_raw)
        obs_queue.append(feat)

        t += 1
        if env._check_success():
            print("Succeed.")
            break

        time.sleep(0.005)

    print(f"Totally {t} steps")
