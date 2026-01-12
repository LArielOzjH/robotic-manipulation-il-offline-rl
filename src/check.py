import os
import numpy as np

DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05RL"


num = 0

for ep_name in sorted(os.listdir(DATA_ROOT)):
    if not ep_name.startswith("ep_"):
        continue

    ep_dir = os.path.join(DATA_ROOT, ep_name)
    if not os.path.isdir(ep_dir):
        continue

    npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
    if len(npz_files) == 0:
        print(f"[WARN] {ep_name}: no npz file found")
        continue

    npz_path = os.path.join(ep_dir, npz_files[0])
    data = np.load(npz_path, allow_pickle=True)

    # successful = data["successful"]
    # action_infos = data["action_infos"]
    # states = data["states"]
    # obs = data["obs"]
    reward = data["reward"]
    num = num + 1


    
    
# print(f"---------------------Totally {num} trajectories---------------------")
