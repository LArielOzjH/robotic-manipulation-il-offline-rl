import os
import numpy as np

DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05RL"

ep_name = "ep_1767624960_524107"
ep_dir = os.path.join(DATA_ROOT, ep_name)
npz_dir = "state_1767625007_980675.npz"
npz_path = os.path.join(ep_dir, npz_dir)

data = np.load(npz_path, allow_pickle=True)
successful = data["successful"]
action_infos = data["action_infos"]
states = data["states"]
reward = data["reward"]
obs = data["obs"]
print(len(obs))
# print(len(states[0]))
print(len(reward))
print(len(action_infos))
print(successful)
print(reward)
# print(
#     f"successful={successful} | "
#     f"T={len(states)} | "
#     f"actions={len(action_infos)}"
# )
# print(action_infos)
# print(action_infos)

import numpy as np
import matplotlib.pyplot as plt

r = np.array(reward)  
steps = np.arange(len(r))

plt.figure()
plt.plot(steps, r)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.yscale("symlog", linthresh=1e-3)   
plt.title("Reward vs Steps (symlog scale)")
plt.tight_layout()
plt.show()
