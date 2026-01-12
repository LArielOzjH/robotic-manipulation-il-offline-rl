import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05RL"

rewards_list = []
lengths = []

for ep_name in sorted(os.listdir(DATA_ROOT)):
    if not ep_name.startswith("ep_"):
        continue

    ep_dir = os.path.join(DATA_ROOT, ep_name)
    if not os.path.isdir(ep_dir):
        continue

    npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
    if len(npz_files) == 0:
        continue

    data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)
    if "reward" not in data:
        continue

    r = np.asarray(data["reward"], dtype=np.float64).reshape(-1)
    if r.size == 0:
        continue

    rewards_list.append(r)
    lengths.append(len(r))

num_eps = len(rewards_list)
max_len = max(lengths)

# ---- NaN padding ----
R = np.full((num_eps, max_len), np.nan)
for i, r in enumerate(rewards_list):
    R[i, : len(r)] = r

steps = np.arange(max_len)

median = np.nanmedian(R, axis=0)
q25 = np.nanpercentile(R, 25, axis=0)
q75 = np.nanpercentile(R, 75, axis=0)

def moving_average_nan(x, w=15):
    x = np.asarray(x, dtype=np.float64)
    mask = ~np.isnan(x)
    x0 = np.where(mask, x, 0.0)
    k = np.ones(w)
    num = np.convolve(x0, k, mode="same")
    den = np.convolve(mask.astype(float), k, mode="same")
    return np.where(den > 0, num / den, np.nan)

W = 15
median_s = moving_average_nan(median, W)
q25_s = moving_average_nan(q25, W)
q75_s = moving_average_nan(q75, W)

# ---- Plot ----
plt.figure(figsize=(10, 6))

plt.plot(
    steps,
    median_s,
    linewidth=2.5,          # 主线加粗
    label="Median reward"
)

plt.fill_between(
    steps,
    q25_s,
    q75_s,
    alpha=0.30,             # IQR 更清晰一点
    label="25–75% range"
)

# 可选：叠加少量轨迹作为参考
for r in rewards_list[:60]:
    rs = np.convolve(r, np.ones(W) / W, mode="same") if len(r) >= W else r
    plt.plot(np.arange(len(rs)), rs, alpha=0.25, linewidth=1.2)

plt.xlabel("Step")
plt.ylabel("Reward")
plt.yscale("symlog", linthresh=1e-3)
ax = plt.gca()
ax.xaxis.set_minor_locator(AutoMinorLocator(2))

plt.grid(
    True,
    which="both",
    # linestyle="dashed",
    linewidth=0.6,
    alpha=0.6
)

plt.title(f"Reward vs Steps ({num_eps} episodes)")
plt.legend()
plt.tight_layout()
plt.savefig("./asset/Template_for_ICLR_2025_Conference_Submission/images/reward.png")
plt.savefig("./asset/Template_for_ICLR_2025_Conference_Submission/images/reward.pdf")
plt.show()
