##########################################
# Title: sac_rollout
# Declaration: execute the trained SAC policy
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
from rl_sac import Actor

#-------------------------------------Config---------------------------------------
# Load scaler
scaler = joblib.load("../model/sac_20260105_010504/scaler.pkl")  # Change to your model dir
goal_pos = np.array((0.1975, 0.1575, 0.875))

body_names = [
    "gripper0_right_eef",
    "Bread_main",
    "robot0_right_hand",
    "gripper0_right_leftfinger",
    "gripper0_right_finger_joint1_tip",
    "gripper0_right_rightfinger",
    "gripper0_right_finger_joint2_tip",
]

keep_keys = [
    # "robot0_joint_pos", 
    # "robot0_joint_vel", 
    "robot0_eef_pos", 
    "robot0_eef_quat", 
    # "robot0_gripper_qpos", # important to avoid the action being guided by qpos(smallll variation in qpos --> smallll addition in gripper action), ⬇︎
    "Bread_pos", 
    "Bread_quat", 
    "Bread_to_robot0_eef_pos"
]

# Load SAC Actor model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Get state dimension from scaler
state_dim = scaler.n_features_in_
action_dim = 7  # Panda robot action dimension

# Initialize Actor network
model = Actor(state_dim=state_dim, action_dim=action_dim, hidden_dim=128).to(device)

# Load trained weights
load_path = Path("../model/sac_20260105_010504") / "actor.pt"  # Change to your model dir
checkpoint = torch.load(str(load_path), map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'actor' in checkpoint:
    model.load_state_dict(checkpoint['actor'])
else:
    model.load_state_dict(checkpoint)

model.eval()
print(f"Loaded SAC policy from {load_path}")
print(f"State dim: {state_dim}, Action dim: {action_dim}")

def get_needed_obs(obs_dict):
    """Extract needed observations from obs dict"""
    return np.concatenate([obs_dict[k].flatten() for k in keep_keys])

def process_observation(obs_raw, goal_pos):
    """Process observation same as training"""
    obs = get_needed_obs(obs_raw)
    bread_pos = obs[7:10]
    bread_z = obs[9:10]
    goal_to_bread = goal_pos - bread_pos
    
    # Concatenate features (match training preprocessing)
    obs = np.concatenate([obs, goal_to_bread])
    obs = np.concatenate([obs, goal_pos])
    # obs = np.concatenate([obs, bread_z])
    
    return obs

def get_action(model, obs, scaler, deterministic=True):
    """Get action from SAC policy"""
    with torch.no_grad():
        # Normalize observation
        obs_normalized = scaler.transform(obs.reshape(1, -1)).flatten()
        state_tensor = torch.tensor(obs_normalized, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get action from policy
        if deterministic:
            # Use mean action (deterministic)
            action = model.get_action(state_tensor)
        else:
            # Sample action (stochastic)
            action, _ = model.sample(state_tensor)
        
        action = action.cpu().numpy().flatten()
    
    return action

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
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (no exploration)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to run")

    args = parser.parse_args()
    
    # Load controller config
    controller_config = load_composite_controller_config(
        controller=args.controller, 
        robot=args.robots[0],
    )
    
    config = {
        "env_name": args.environment,
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config
    else:
        args.config = None
    
    # Generate the environment
    env = suite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
        z_rotation=0
    )
    
    # Wrap with visualization
    env = VisualizationWrapper(env, indicator_configs=None)
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
    
    # Run multiple episodes for evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"\n{'='*60}")
    print(f"Running SAC Policy Evaluation")
    print(f"Deterministic: {args.deterministic}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"{'='*60}\n")
    
    for episode in range(args.num_episodes):
        # Reset environment
        obs_raw = env.reset()
        obs = process_observation(obs_raw, goal_pos)
        
        # Use deque for observation stacking (K=1 for Markov assumption)
        K = 1
        obs_queue = deque([obs] * K, maxlen=K)
        
        env.render()
        done = False
        episode_reward = 0.
        t = 0
        success = False
        
        print(f"Episode {episode + 1}/{args.num_episodes}")
        
        # Inference loop
        while not done:
            # Get observation stack
            state_stacked = np.concatenate(list(obs_queue))
            
            # Get action from SAC policy
            action = get_action(model, state_stacked, scaler, deterministic=args.deterministic)
            
            print(f"Step {t:3d} | Action: {action}")
            
            # Step environment
            obs_raw, reward, done, _ = env.step(action)
            
            # Process next observation
            obs = process_observation(obs_raw, goal_pos)
            obs_queue.append(obs)
            
            episode_reward += reward
            t += 1
            
            # Check success
            if env._check_success():
                print("✓ SUCCESS! PandaPickPlaceBread task completed!")
                success = True
                success_count += 1
                break
            
            # Control simulation speed
            time.sleep(0.02)
        
        # Episode summary
        episode_rewards.append(episode_reward)
        episode_lengths.append(t)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Total steps: {t}")
        print(f"  Success: {'✓' if success else '✗'}")
        print()
    
    # Overall summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary")
    print(f"{'='*60}")
    print(f"Episodes completed: {args.num_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{args.num_episodes} ({100*success_count/args.num_episodes:.1f}%)")
    print(f"{'='*60}\n")