##########################################
# Title: mlp_rollout
# Declaration: execute the trained policy
# Powered by LArielO
##########################################

import argparse
import time
import os
from pathlib import Path
from collections import deque
from glob import glob
import numpy as np
import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import VisualizationWrapper
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from trajectories_record import PandaPickPlace, PandaPickPlaceBread
from mlp_train import MLPPolicy
from rl_sac import Actor

#-------------------------------------Config---------------------------------------
scaler = joblib.load("../model/mlp_policy/state_scaler.pkl")
# scaler = joblib.load("../23d_0s_128b_32n_3l_005p_00875g/state_scaler.pkl")
goal_pos = np.array((0.1975, 0.1575, 0.855))
body_names = [
    "gripper0_right_eef",
    "Bread_main",
    "robot0_right_hand",
    "gripper0_right_leftfinger",
    "gripper0_right_finger_joint1_tip",
    "gripper0_right_rightfinger",
    "gripper0_right_finger_joint2_tip",
]
# ⬆︎⬇︎
keep_keys = [
    "robot0_eef_pos", 
    "robot0_eef_quat", 
    "Bread_pos", 
    "Bread_quat", 
    "Bread_to_robot0_eef_pos"
]
model = MLPPolicy()
load_path = Path("../model/mlp_policy") / "mlp_policy.pt"
# load_path = Path("../23d_0s_128b_32n_3l_005p_00875g") / "mlp_policy.pt"
model.load_state_dict(torch.load(str(load_path), map_location="cpu"))
model.eval()

def get_needed_obs(obs_dict):
    return np.concatenate([obs_dict[k].flatten() for k in keep_keys])

#-----------------------------------Main Function-------------------------------------
if __name__ == "__main__":

#----------------------------Argument & Configuration---------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", type=str, default="Lift") 
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Used Robot") 
    parser.add_argument("--config", type=str, default="default", help="Environment Configuration (if needed)") 
    parser.add_argument("--arm", type=str, default="right", help="Controlled Arm (e.g. 'right' or 'left')")
    parser.add_argument("--switch-on-grasp", action="store_true", help="Switch Gripper Control on Grasp")
    parser.add_argument("--toggle-camera-on-grasp", action="store_true", help="Switch Camera Angle on Grasp")
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Controller Selection. Can be a generic name (e.g. 'BASIC' or 'WHOLE_BODY_MINK_IK') or a json file (see robosuite/controllers/config example) or None (use robot's default controller if exists)",
    ) 
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=0.05, help="Position Input Scaling")
    parser.add_argument("--rot-sensitivity", type=float, default=0.1, help="Rotation Input Scaling")
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Pause when simulation speed exceeds specified frame rate; 20 fps is real-time.",
    )

    args = parser.parse_args()
    controller_config = load_composite_controller_config(controller=args.controller, robot=args.robots[0],)
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None
    # Generate the environment, where the detailed environment and arms are selected by argparser.
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=180,
        hard_reset=False,
        z_rotation=0 # I set the z_rotation as 0 to simplify the task settelment.
        
    )
    # Package the generated basic env as a VisualizationWrapper class, which is used to display the inference process.
    env = VisualizationWrapper(env, indicator_configs=None)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    
    # K can be used to stack K frames, When using MLP modeling, the process is estimated as Markov, K = 1 renders better I think.
    K = 1
    obs_raw = env.reset()
    obs = get_needed_obs(obs_raw)
    bread_pos = obs[7:10]
    bread_z = obs[9:10]
    eef_pos = obs[0:3]
    goal_to_bread = goal_pos - bread_pos
    goal_to_eef = goal_pos - eef_pos
    obs = np.concatenate([obs, goal_pos])
    obs = np.concatenate([obs, goal_to_bread])
    # obs = np.concatenate([obs, goal_to_eef])
    # obs = np.concatenate([obs, goal_pos])
    # obs = np.concatenate([obs, bread_z])
    # obs = np.concatenate([obs, eef_z])
    # queue for K != 1 & the first round
    obs_queue = deque([obs] * K, maxlen=K)
    
    env.render()
    done = False
    ret = 0.
    t = 0

    # Inference
    while not done:
        with torch.no_grad():
            state_stacked = np.concatenate(list(obs_queue)) 
            # scaler here before feeding the features into the MLP.
            state_stacked = scaler.transform(state_stacked.reshape(1, -1)).flatten()
            state_tensor = torch.tensor(state_stacked, dtype=torch.float32).unsqueeze(0)
            action = model(state_tensor).detach().cpu().numpy()[0]
            # action = model.get_action(state_tensor).detach().cpu().numpy()[0]
            # action, _ = model.sample(state_tensor)
            # action = action.squeeze(0).cpu().numpy()
            
            print(f"Current action is {action}")
        # Main function to step forward and concat some other states
        obs_raw, reward, done, _ = env.step(action)
        obs = get_needed_obs(obs_raw)
        bread_pos = obs[7:10] 
        bread_z = obs[9:10] 
        eef_z = obs[2:3]
        eef_pos = obs[0:3]
        goal_to_bread = goal_pos - bread_pos
        goal_to_eef = goal_pos - eef_pos
        obs = np.concatenate([obs, goal_pos])
        obs = np.concatenate([obs, goal_to_bread])
        # obs = np.concatenate([obs, goal_to_eef])
        # obs = np.concatenate([obs, bread_z])
        # obs = np.concatenate([obs, eef_z])

        obs_queue.append(obs)

        ret += reward
        t += 1
        if env._check_success():
            print("Fabulous Lando, your PandaPickPlaceBread task SUCCEED!!!")
            break
        # Not to fast
        # time.sleep(0.02)
    # print("rollout completed with return {}".format(ret))
    print(f"Totally {t} steps")
    