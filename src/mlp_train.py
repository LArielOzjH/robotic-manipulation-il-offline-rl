##########################################
# Title: mlp_train
# Declaration: preprocess the collected trajectories & train the policy
# Powered by LArielO
##########################################

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from pathlib import Path

'''
Observations Examples
(OrderedDict([
    ('robot0_joint_pos', array([0.042, 0.198, -0.013, -2.628, 0.008, 2.909, 0.795])), 
    ('robot0_joint_pos_cos', array([0.999, 0.981, 1.000, -0.871, 1.000, -0.973, 0.701])), 
    ('robot0_joint_pos_sin', array([0.042, 0.196, -0.013, -0.491, 0.008, 0.231, 0.714])), 
    ('robot0_joint_vel', array([-0.038, -0.018, -0.082, 0.083, -0.016, 0.131, -0.002])), 
    ('robot0_eef_pos', array([-0.055, -0.088, 1.004])), 
    ('robot0_eef_quat', array([0.999, 0.007, 0.041, -0.001])), 
    ('robot0_eef_quat_site', array([0.702, 0.711, 0.030, 0.029], dtype=float32)), 
    ('robot0_gripper_qpos', array([0.022, -0.022])), 
    ('robot0_gripper_qvel', array([0.016, -0.016])), 
    ('Bread_to_robot0_eef_pos', array([-0.027, -0.086, 1.005])), 
    ('Bread_to_robot0_eef_quat', array([0.999, 0.007, 0.041, 0.001], dtype=float32)), 
    ('Bread_pos', array([0.104, -0.373, 0.842])), 
    ('Bread_quat', array([-0.000, 0.002, 0.000, 1.000])), 
    ('robot0_proprio-state', array([0.042, 0.198, -0.013, -2.628, 0.008, 2.909, 0.795, 0.999, 0.981, 1.000, -0.871, 1.000, -0.973, 0.701, 0.042, 0.196, -0.013, -0.491, 0.008, 0.231, 0.714, -0.038, -0.018, -0.082, 0.083, -0.016, 0.131, -0.002, -0.055, -0.088, 1.004, 0.999, 0.007, 0.041, -0.001, 0.702, 0.711, 0.030, 0.029, 0.022, -0.022, 0.016, -0.016])), 
    ('object-state', array([-0.027, -0.086, 1.005, 0.999, 0.007, 0.041, 0.001, 0.104, -0.373, 0.842, -0.000, 0.002, 0.000, 1.000]))
]), np.float64(0.00013661406772260686), False, {})
'''

# ---------------------------- Model Class----------------------------
# the linear dimension less than 128 makes great progress, I set them as at least 256 at first but all fail.
class ScaledTanh(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return torch.tanh(self.alpha * x)

class MLPPolicy(nn.Module):
    def __init__(self, state_dim=23, action_dim=7):
        super().__init__()
        self.hidden_dim = 32
        self.net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, action_dim),
            nn.Tanh() # used to generate bounded actions
        )
        
    def forward(self, x):
        return self.net(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu" 

def moving_average(x, w=7):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w:
        return x
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, k, mode="same")


def save_logs_and_plot(out_dir: str, tag: str, train_losses, val_losses):
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_path = os.path.join(out_dir, f"{tag}_logs_{stamp}.npz")
    csv_path = os.path.join(out_dir, f"{tag}_logs_{stamp}.csv")
    fig_png = os.path.join(out_dir, f"{tag}_convergence_{stamp}.png")
    fig_pdf = os.path.join(out_dir, f"{tag}_convergence_{stamp}.pdf")

    epochs = np.arange(1, len(train_losses) + 1, dtype=np.int32)
    train_losses = np.asarray(train_losses, dtype=np.float32)
    val_losses = np.asarray(val_losses, dtype=np.float32)

    # ---- save npz ----
    np.savez(npz_path, epochs=epochs, train_loss=train_losses, val_loss=val_losses)

    # ---- save csv ----
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, va in zip(epochs, train_losses, val_losses):
            writer.writerow([int(e), float(tr), float(va)])

    # ---- plot ----
    # 配色：tab10 经典蓝/橙
    c_train = plt.get_cmap("tab10")(0)
    c_val = plt.get_cmap("tab10")(1)

    tr_s = moving_average(train_losses, w=7)
    va_s = moving_average(val_losses, w=7)

    best_ep = int(np.argmin(val_losses) + 1)
    best_val = float(np.min(val_losses))

    plt.figure(figsize=(9.2, 5.6), dpi=130)
    ax = plt.gca()

    # 原始曲线（淡一点）
    ax.plot(epochs, train_losses, color=c_train, alpha=0.22, linewidth=1.2)
    ax.plot(epochs, val_losses, color=c_val, alpha=0.22, linewidth=1.2)

    # 平滑曲线（主视觉）
    ax.plot(epochs, tr_s, color=c_train, linewidth=2.4, label="Train loss (MA-7)")
    ax.plot(epochs, va_s, color=c_val, linewidth=2.4, label="Val loss (MA-7)")

    # 最佳点标注
    ax.scatter([best_ep], [best_val], color=c_val, s=45, zorder=5)
    ax.annotate(
        f"best val={best_val:.4g} @ epoch {best_ep}",
        xy=(best_ep, best_val),
        xytext=(best_ep, best_val * 1.12),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=1.0, color=c_val, alpha=0.9),
        color=c_val,
    )

    ax.set_title("MLP Policy Training Convergence", fontsize=14, pad=10)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)

    # 让收敛更清晰：可选用 log y（loss跨度大时很有用）
    # ax.set_yscale("log")

    ax.grid(True, which="both", alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_png)
    plt.savefig(fig_pdf)
    plt.close()

    print(f"[Log] saved: {npz_path}")
    print(f"[Log] saved: {csv_path}")
    print(f"[Plot] saved: {fig_png}")
    print(f"[Plot] saved: {fig_pdf}")


def main():
    
    goal_pos = np.array((0.197, 0.159, 0.938))
    # -------------------------- Load data ----------------------------
    DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05"

    K = 1
    episodes_obs = [] 
    episodes_actions = [] 

    ep_names = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith("ep_")]

    for ep_name in ep_names:
        ep_dir = os.path.join(DATA_ROOT, ep_name)
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files: continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)
        # if not data["successful"]: continue

        # Features engineering
        obs_raw = np.asarray(data["obs"])
        goal_pos_fixed = np.asarray(goal_pos).reshape(1, 3)
        goal_block = np.repeat(goal_pos_fixed, obs_raw.shape[0], axis=0)
        bread_pos = obs_raw[:, 23:26]
        bread_z = obs_raw[:, 25:26]
        eef_z = obs_raw[:, 16:17]
        eef_pos = obs_raw[:, 14:17]
        goal_to_bread = goal_block - bread_pos
        goal_to_eef = goal_block - eef_pos
        obs_enhanced = obs_raw[:, 14:]
        obs_enhanced = np.concatenate([obs_enhanced[:, :7], obs_enhanced[:, 9:]], axis=1) # gripper qpos out
        obs_enhanced = np.concatenate([obs_enhanced, goal_block, goal_to_bread], axis=1)
        
        action_infos = data["action_infos"]
        
        T_total = min(len(action_infos), len(obs_enhanced) - 1)

        current_ep_obs = []
        current_ep_actions = []

        for t in range(K - 1, T_total):
            window = obs_enhanced[t - (K - 1) : t + 1] 
            s_stacked = window.flatten()
            a = np.asarray(action_infos[t]["actions"], dtype=np.float32).copy()

            current_ep_obs.append(s_stacked)
            current_ep_actions.append(a)

        if len(current_ep_obs) > 0:
            episodes_obs.append(np.array(current_ep_obs))
            episodes_actions.append(np.array(current_ep_actions))
            
# ------------------------ Split by Trajectory ------------------------
    num_episodes = len(episodes_obs)
    
    indices = np.arange(num_episodes)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42, shuffle=True
    )

    X_train = np.concatenate([episodes_obs[i] for i in train_idx], axis=0).astype(np.float32)
    y_train = np.concatenate([episodes_actions[i] for i in train_idx], axis=0).astype(np.float32)

    X_val = np.concatenate([episodes_obs[i] for i in val_idx], axis=0).astype(np.float32)
    y_val = np.concatenate([episodes_actions[i] for i in val_idx], axis=0).astype(np.float32)

    print("Data ready:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val  : {X_val.shape}")

    # ------------------------------ Scale ------------------------------
    state_scaler = StandardScaler()
    X_train = state_scaler.fit_transform(X_train)
    X_val = state_scaler.transform(X_val)

    os.makedirs("model/mlp_policy", exist_ok=True)
    joblib.dump(state_scaler, "model/mlp_policy/state_scaler.pkl")

    # --------------------------- Torch dataset --------------------------
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64, # 64 + 3e-4
        shuffle=True,
        drop_last=False,
    )

    # -------------------------- Model --------------------------
    state_dim = X_train.shape[1]
    model = MLPPolicy(state_dim=state_dim, action_dim=7).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    # -------------------------- Train --------------------------
    best_val = float("inf")
    best_state = None
    max_epochs = 150 
    train_losses = []
    val_losses = []


    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:03d} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f}"
            )

    # -------------------------- Save --------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), "model/mlp_policy/mlp_policy.pt")
    print("[DONE] saved mlp_policy.pt, best val =", best_val)
    save_logs_and_plot(
        out_dir="model/mlp_policy",
        tag="mlp_policy",
        train_losses=train_losses,
        val_losses=val_losses
    )

    
    
if __name__ == "__main__":
    main()

# ##########################################
# # Title: mlp_train_hybrid_loss
# # Declaration: preprocess the collected trajectories & train the policy with hybrid loss
# # Powered by LArielO
# ##########################################

# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import joblib
# from pathlib import Path

# # ---------------------------- Model Class ----------------------------
# class MLPPolicy(nn.Module):
#     def __init__(self, state_dim=23, action_dim=7):
#         super().__init__()
#         self.hidden_dim = 32

#         self.backbone = nn.Sequential(
#             nn.Linear(state_dim, self.hidden_dim),
#             nn.BatchNorm1d(self.hidden_dim),
#             nn.ReLU(),
#             nn.Linear(self.hidden_dim, self.hidden_dim),
#             nn.ReLU(),
#         )

#         self.continuous_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, 6),
#             nn.Tanh()
#         )

#         self.gripper_head = nn.Linear(self.hidden_dim, 1)
        
#     def forward(self, x):
#         """
#         Returns:
#             continuous: shape (batch, 6) 
#             gripper_logit: shape (batch, 1) 
#         """
#         features = self.backbone(x)
#         continuous = self.continuous_head(features)
#         gripper_logit = self.gripper_head(features)
#         return continuous, gripper_logit
    
#     def get_action(self, x):

#         continuous, gripper_logit = self.forward(x)

#         gripper = torch.where(gripper_logit > 0, 
#                              torch.ones_like(gripper_logit), 
#                              -torch.ones_like(gripper_logit))
#         return torch.cat([continuous, gripper], dim=-1)


# device = "cuda" if torch.cuda.is_available() else "cpu" 

# def main():
    
#     goal_pos = np.array((0.1975, 0.1575, 0.875))
#     # -------------------------- Load data ----------------------------
#     DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05RL"

#     K = 1
#     episodes_obs = [] 
#     episodes_actions = [] 

#     ep_names = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith("ep_")]

#     for ep_name in ep_names:
#         ep_dir = os.path.join(DATA_ROOT, ep_name)
#         npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
#         if not npz_files: continue

#         data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)

#         # Features engineering
#         obs_raw = np.asarray(data["obs"])   
#         goal_pos_fixed = np.asarray(goal_pos).reshape(1, 3) 
#         goal_block = np.repeat(goal_pos_fixed, obs_raw.shape[0], axis=0)
#         bread_pos = obs_raw[:, 23:26]
#         bread_z = obs_raw[:, 25:26]
#         eef_z = obs_raw[:, 16:17]
#         goal_to_bread = goal_block - bread_pos
#         obs_enhanced = obs_raw[:, 14:]
#         obs_enhanced = np.concatenate([obs_enhanced[:, :7], obs_enhanced[:, 9:]], axis=1) # gripper qpos out
#         obs_enhanced = np.concatenate([obs_enhanced, goal_to_bread, goal_block], axis=1)
        
#         action_infos = data["action_infos"]
        
#         T_total = min(len(action_infos), len(obs_enhanced) - 1)

#         current_ep_obs = []
#         current_ep_actions = []

#         for t in range(K - 1, T_total):
#             window = obs_enhanced[t - (K - 1) : t + 1] 
#             s_stacked = window.flatten() 
#             a = np.asarray(action_infos[t]["actions"], dtype=np.float32).copy()

#             current_ep_obs.append(s_stacked)
#             current_ep_actions.append(a)

#         if len(current_ep_obs) > 0:
#             episodes_obs.append(np.array(current_ep_obs))
#             episodes_actions.append(np.array(current_ep_actions))
            
#     # ------------------------ Split by Trajectory ------------------------
#     num_episodes = len(episodes_obs)
    
#     indices = np.arange(num_episodes)
#     train_idx, val_idx = train_test_split(
#         indices, test_size=0.1, random_state=42, shuffle=True
#     )

#     X_train = np.concatenate([episodes_obs[i] for i in train_idx], axis=0).astype(np.float32)
#     y_train = np.concatenate([episodes_actions[i] for i in train_idx], axis=0).astype(np.float32)

#     X_val = np.concatenate([episodes_obs[i] for i in val_idx], axis=0).astype(np.float32)
#     y_val = np.concatenate([episodes_actions[i] for i in val_idx], axis=0).astype(np.float32)

#     y_train_continuous = y_train[:, :6]
#     y_train_gripper = y_train[:, 6:7]
    
#     y_val_continuous = y_val[:, :6]
#     y_val_gripper = y_val[:, 6:7]
    
#     # Binary
#     y_train_gripper_binary = (y_train_gripper + 1) / 2
#     y_val_gripper_binary = (y_val_gripper + 1) / 2
    
#     print("Data ready:")
#     print(f"  Train: X={X_train.shape}, y_cont={y_train_continuous.shape}, y_grip={y_train_gripper_binary.shape}")
#     print(f"  Val  : X={X_val.shape}, y_cont={y_val_continuous.shape}, y_grip={y_val_gripper_binary.shape}")
#     print(f"  Gripper distribution (train): {np.bincount(y_train_gripper_binary.astype(int).flatten())}")

#     # ------------------------------ Scale ------------------------------
#     state_scaler = StandardScaler()
#     X_train = state_scaler.fit_transform(X_train)
#     X_val = state_scaler.transform(X_val)

#     os.makedirs("model/mlp_policy", exist_ok=True)
#     joblib.dump(state_scaler, "model/mlp_policy/state_scaler.pkl")

#     # --------------------------- Torch dataset --------------------------
#     X_train_t = torch.from_numpy(X_train)
#     y_train_cont_t = torch.from_numpy(y_train_continuous)
#     y_train_grip_t = torch.from_numpy(y_train_gripper_binary)
    
#     X_val_t = torch.from_numpy(X_val).to(device)
#     y_val_cont_t = torch.from_numpy(y_val_continuous).to(device)
#     y_val_grip_t = torch.from_numpy(y_val_gripper_binary).to(device)

#     train_loader = DataLoader(
#         TensorDataset(X_train_t, y_train_cont_t, y_train_grip_t),
#         batch_size=64,
#         shuffle=True,
#         drop_last=False,
#     )

#     # -------------------------- Model --------------------------
#     state_dim = X_train.shape[1]
#     model = MLPPolicy(state_dim=state_dim, action_dim=7).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=3e-4)

#     criterion_continuous = nn.MSELoss()
#     criterion_gripper = nn.BCEWithLogitsLoss() 

#     weight_continuous = 2
#     weight_gripper = 0.5  

#     # -------------------------- Train --------------------------
#     best_val = float("inf")
#     best_state = None
#     max_epochs = 100

#     for epoch in range(1, max_epochs + 1):
#         # ---- train ----
#         model.train()
#         train_loss_cont = 0.0
#         train_loss_grip = 0.0
#         train_loss_total = 0.0

#         for xb, yb_cont, yb_grip in train_loader:
#             xb = xb.to(device)
#             yb_cont = yb_cont.to(device)
#             yb_grip = yb_grip.to(device)

#             optimizer.zero_grad()

#             pred_cont, pred_grip_logit = model(xb)

#             loss_cont = criterion_continuous(pred_cont, yb_cont)
#             loss_grip = criterion_gripper(pred_grip_logit, yb_grip)

#             loss = weight_continuous * loss_cont + weight_gripper * loss_grip
            
#             loss.backward()
#             optimizer.step()

#             train_loss_cont += loss_cont.item() * xb.size(0)
#             train_loss_grip += loss_grip.item() * xb.size(0)
#             train_loss_total += loss.item() * xb.size(0)

#         train_loss_cont /= len(train_loader.dataset)
#         train_loss_grip /= len(train_loader.dataset)
#         train_loss_total /= len(train_loader.dataset)

#         # ---- val ----
#         model.eval()
#         with torch.no_grad():
#             val_pred_cont, val_pred_grip_logit = model(X_val_t)
#             val_loss_cont = criterion_continuous(val_pred_cont, y_val_cont_t).item()
#             val_loss_grip = criterion_gripper(val_pred_grip_logit, y_val_grip_t).item()
#             val_loss_total = weight_continuous * val_loss_cont + weight_gripper * val_loss_grip

#             val_pred_grip_binary = (torch.sigmoid(val_pred_grip_logit) > 0.5).float()
#             gripper_acc = (val_pred_grip_binary == y_val_grip_t).float().mean().item()

#         if val_loss_total < best_val:
#             best_val = val_loss_total
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

#         if epoch % 10 == 0 or epoch == 1:
#             print(
#                 f"Epoch {epoch:03d} | "
#                 f"train_total={train_loss_total:.6f} (cont={train_loss_cont:.6f}, grip={train_loss_grip:.6f}) | "
#                 f"val_total={val_loss_total:.6f} (cont={val_loss_cont:.6f}, grip={val_loss_grip:.6f}, acc={gripper_acc:.3f})"
#             )

#     # -------------------------- Save --------------------------
#     if best_state is not None:
#         model.load_state_dict(best_state)

#     torch.save(model.state_dict(), "model/mlp_policy/mlp_policy.pt")
#     print(f"[DONE] saved mlp_policy.pt, best val = {best_val:.6f}")
#     print(f"Note: Use model.get_action(state) for inference to get discrete gripper actions")
    
    
# if __name__ == "__main__":
#     main()