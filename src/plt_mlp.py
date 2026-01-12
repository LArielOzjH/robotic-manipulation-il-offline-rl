import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 1. 读 CSV =====
csv_path = "./model/mlp_policy/mlp_policy_logs_20260112_122132.csv"   # ← 改成你的路径
df = pd.read_csv(csv_path)

epochs = df["epoch"].to_numpy()
train_loss = df["train_loss"].to_numpy()
val_loss = df["val_loss"].to_numpy()

# ===== 2. 平滑 =====
def moving_average(x, w=7):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")

train_s = moving_average(train_loss, 7)
val_s = moving_average(val_loss, 7)

# ===== 3. 最优点 =====
best_idx = np.argmin(val_loss)
best_epoch = epochs[best_idx]
best_val = val_loss[best_idx]

# ===== 4. 画图 =====
plt.figure(figsize=(8.5, 5.2), dpi=130)
ax = plt.gca()

# 配色：tab10
c_train = plt.get_cmap("tab10")(0)
c_val = plt.get_cmap("tab10")(1)

# 原始曲线（淡）
ax.plot(epochs, train_loss, color=c_train, linewidth=2.2)
ax.plot(epochs, val_loss,   color=c_val,   linewidth=2.2)

# 平滑曲线（主视觉）
ax.plot(epochs, train_s, color=c_train, alpha=0.65, linewidth=1.4, label="Train loss (MA-7)")
ax.plot(epochs, val_s,   color=c_val,   alpha=0.65, linewidth=1.4, label="Val loss (MA-7)")

# # 最优点
# ax.scatter(best_epoch, best_val, color=c_val, s=40, zorder=5)




# 坐标 & 样式
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("MSE Loss", fontsize=12)
ax.set_title("MLP Policy Training Convergence", fontsize=14, pad=10)

ax.grid(True, alpha=0.28, linewidth=0.8)
ax.legend(frameon=False, fontsize=11)

# for spine in ["top", "right"]:
#     ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("mlp_convergence.png", dpi=200)
plt.savefig("mlp_convergence.pdf")
plt.show()
