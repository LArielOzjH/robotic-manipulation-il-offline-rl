import numpy as np
import os

def augment_trajectory_data(data_root, goal_pos, output_root):

    augmented_count = 0
    
    for ep_name in os.listdir(data_root):
        ep_dir = os.path.join(data_root, ep_name)
        if not os.path.isdir(ep_dir):
            continue
            
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files:
            continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)
        obs_raw = np.asarray(data["obs"])
        # rewards = np.asarray(data["reward"])
        action_infos = data["action_infos"]

        num_augmentations = 3 
        
        for aug_id in range(num_augmentations):
            
            obs_noise = np.random.normal(0, 0.01, obs_raw.shape)
            obs_augmented = obs_raw + obs_noise

            actions_augmented = []
            for ai in action_infos:
                action = np.asarray(ai["actions"])
                action_noise = np.random.normal(0, 0.02, action.shape)
                action_aug = np.clip(action + action_noise, -1, 1)
                actions_augmented.append({"actions": action_aug})

            aug_ep_name = f"{ep_name}_aug_{aug_id}"
            aug_ep_dir = os.path.join(output_root, aug_ep_name)
            os.makedirs(aug_ep_dir, exist_ok=True)
            
            np.savez(
                os.path.join(aug_ep_dir, "trajectory.npz"),
                obs=obs_augmented,
                # reward=rewards,
                action_infos=np.array(actions_augmented, dtype=object)
            )
            augmented_count += 1
    
    print(f"生成了 {augmented_count} 条增强轨迹")
    return augmented_count

goal_pos = np.array((0.1975, 0.1575, 0.875))
original_data = "PoS0_05"
augmented_data = "PoS0_05_augmented"
augment_trajectory_data(original_data, goal_pos, augmented_data)

