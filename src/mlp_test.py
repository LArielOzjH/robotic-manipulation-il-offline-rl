##########################################
# Title: mlp_rollout_eval
# Declaration: multi-rollout evaluation + success rate (NO VIDEO)
##########################################

import argparse
import time
from collections import deque
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import VisualizationWrapper
import joblib
import torch
from trajectories_record import PandaPickPlace, PandaPickPlaceBread
from mlp_train import MLPPolicy

# ========================= Config =========================
NUM_TEST = 50
MAX_STEPS = 10000     
FPS = 20

scaler = joblib.load("../model/mlp_policy/state_scaler.pkl")

goal_pos = np.array((0.197, 0.159, 0.938))

keep_keys = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "Bread_pos",
    "Bread_quat",
    "Bread_to_robot0_eef_pos",
]

# ========================= Utils =========================
def get_needed_obs(obs_dict):
    return np.concatenate([obs_dict[k].flatten() for k in keep_keys])

# ========================= Main =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda")
    parser.add_argument("--controller", type=str, default=None)
    args = parser.parse_args()

    controller_config = load_composite_controller_config(
        controller=args.controller,
        robot=args.robots[0],
    )

    env = suite.make(
        env_name=args.environment,
        robots=args.robots,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False, 
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=180,
        hard_reset=False,
        z_rotation=0,
    )

    env = VisualizationWrapper(env)
    np.set_printoptions(precision=3, suppress=True)

    # ------------------ Load policy ------------------
    model = MLPPolicy()
    model.load_state_dict(
        torch.load("../model/mlp_policy/mlp_policy.pt", map_location="cpu")
    )
    model.eval()

    # ===================== Evaluation =====================
    success_count = 0
    success_steps = []

    K = 1

    for ep in range(NUM_TEST):
        print(f"\n========== Rollout {ep + 1}/{NUM_TEST} ==========")

        obs_raw = env.reset()
        obs = get_needed_obs(obs_raw)

        bread_pos = obs[7:10]
        eef_pos = obs[0:3]
        goal_to_bread = goal_pos - bread_pos
        goal_to_eef = goal_pos - eef_pos
        obs = np.concatenate([obs, goal_pos, goal_to_bread])

        obs_queue = deque([obs] * K, maxlen=K)

        success = False

        for t in range(1, MAX_STEPS + 1):

            with torch.no_grad():
                state = np.concatenate(list(obs_queue))
                state = scaler.transform(state.reshape(1, -1)).flatten()
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = model(state_t).cpu().numpy()[0]
                # action = model.get_action(state_t).detach().cpu().numpy()[0]

            obs_raw, reward, done, _ = env.step(action)

            obs = get_needed_obs(obs_raw)
            bread_pos = obs[7:10]
            eef_pos = obs[0:3]
            goal_to_bread = goal_pos - bread_pos
            goal_to_eef = goal_pos - eef_pos
            obs = np.concatenate([obs, goal_pos, goal_to_bread])
            obs_queue.append(obs)

            if env._check_success():
                success = True
                success_count += 1
                success_steps.append(t)
                print(f"✅ Success in {t} steps")
                break

            # time.sleep(1.0 / FPS)

        if not success:
            print(f"❌ Failed (reach {MAX_STEPS} steps)")

    # ===================== Summary =====================
    print("\n================== FINAL RESULT ==================")
    print(f"Success rate: {success_count}/{NUM_TEST} = {success_count / NUM_TEST:.2%}")

    if success_steps:
        print(f"Avg steps (success only): {np.mean(success_steps):.1f}")
        print(f"Min steps: {np.min(success_steps)}")
        print(f"Max steps: {np.max(success_steps)}")
