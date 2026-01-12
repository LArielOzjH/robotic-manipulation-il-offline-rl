##########################################
# Title: rnn_train
# Declaration: preprocess trajectories & train an RNN policy (GRU)
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
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from pathlib import Path


# ---------------------------- Model Class ----------------------------
class GRUPolicy(nn.Module):
    """
    Input:  (B, T, D)
    Output: (B, action_dim)
    We use the last hidden state for action regression.
    """
    def __init__(self, state_dim: int, action_dim: int = 7, hidden_dim: int = 32, num_layers: int = 1):
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
            nn.Tanh(),  # bounded actions
        )

    def forward(self, x):  # x: (B, T, D)
        _, h_n = self.gru(x)          # h_n: (num_layers, B, hidden_dim)
        h_last = h_n[-1]              # (B, hidden_dim)
        return self.head(h_last)      # (B, action_dim)


device = "cuda" if torch.cuda.is_available() else "cpu"


def fit_scaler_on_sequences(X_seq: np.ndarray) -> StandardScaler:
    """
    X_seq: (N, T, D)
    Fit scaler on all timesteps pooled: (N*T, D)
    """
    N, T, D = X_seq.shape
    scaler = StandardScaler()
    scaler.fit(X_seq.reshape(N * T, D))
    return scaler


def transform_sequences(scaler: StandardScaler, X_seq: np.ndarray) -> np.ndarray:
    N, T, D = X_seq.shape
    X_flat = scaler.transform(X_seq.reshape(N * T, D))
    return X_flat.reshape(N, T, D)

def moving_average(x, w=7):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w:
        return x
    k = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, k, mode="same")


def save_logs_and_plot(save_dir: Path, tag: str, epochs, train_losses, val_losses):
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    npz_path = save_dir / f"{tag}_logs_{stamp}.npz"
    csv_path = save_dir / f"{tag}_logs_{stamp}.csv"
    fig_png = save_dir / f"{tag}_convergence_{stamp}.png"
    fig_pdf = save_dir / f"{tag}_convergence_{stamp}.pdf"

    epochs = np.asarray(epochs, dtype=np.int32)
    train_losses = np.asarray(train_losses, dtype=np.float32)
    val_losses = np.asarray(val_losses, dtype=np.float32)

    # ---- save npz ----
    np.savez(npz_path, epoch=epochs, train_loss=train_losses, val_loss=val_losses)

    # ---- save csv ----
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, va in zip(epochs, train_losses, val_losses):
            w.writerow([int(e), float(tr), float(va)])

    # ---- plot ----
    c_train = plt.get_cmap("tab10")(0)  # 蓝
    c_val = plt.get_cmap("tab10")(1)    # 橙

    tr_s = moving_average(train_losses, 7)
    va_s = moving_average(val_losses, 7)

    best_idx = int(np.argmin(val_losses))
    best_epoch = int(epochs[best_idx])
    best_val = float(val_losses[best_idx])

    plt.figure(figsize=(8.5, 5.2), dpi=130)
    ax = plt.gca()

    ax.plot(epochs, train_losses, color=c_train, alpha=0.45, linewidth=1.6)
    ax.plot(epochs, val_losses,   color=c_val,   alpha=0.45, linewidth=1.6)

    ax.plot(epochs, tr_s, color=c_train, linewidth=2.4, label="Train loss (MA-7)")
    ax.plot(epochs, va_s, color=c_val,   linewidth=2.4, label="Val loss (MA-7)")

    ax.scatter(best_epoch, best_val, color=c_val, s=40, zorder=5)

    ax.annotate(
        f"best val = {best_val:.4f} (epoch {best_epoch})",
        xy=(best_epoch, best_val),
        xycoords="data",
        xytext=(0.98, 0.92),
        textcoords="axes fraction",
        ha="right",
        va="top",
        arrowprops=dict(
            arrowstyle="->",
            lw=1.0,
            color=c_val,
            connectionstyle="arc3,rad=-0.15",
        ),
        fontsize=10,
        color=c_val,
    )

    ax.set_title("GRU Policy Training Convergence", fontsize=14, pad=10)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)

    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.legend(frameon=False, fontsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_png, dpi=200)
    plt.savefig(fig_pdf)
    plt.close()

    print(f"[Log] saved: {npz_path}")
    print(f"[Log] saved: {csv_path}")
    print(f"[Plot] saved: {fig_png}")
    print(f"[Plot] saved: {fig_pdf}")



def main():
    # -------------------------- Config --------------------------
    DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/PoS0_05"
    SAVE_DIR = Path("model/gru_policy")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    goal_pos = np.array((0.197, 0.159, 0.938), dtype=np.float32)

    seq_len = 7
    batch_size = 64
    lr = 3e-4
    weight_decay = 1e-6
    max_epochs = 150
    grad_clip = 1.0

    # -------------------------- Load data --------------------------
    episodes_obs = []
    episodes_actions = []

    ep_names = [d for d in sorted(os.listdir(DATA_ROOT)) if d.startswith("ep_")]

    for ep_name in ep_names:
        ep_dir = os.path.join(DATA_ROOT, ep_name)
        npz_files = [f for f in os.listdir(ep_dir) if f.endswith(".npz")]
        if not npz_files:
            continue

        data = np.load(os.path.join(ep_dir, npz_files[0]), allow_pickle=True)
        # if "successful" in data and (not bool(data["successful"])):
        #     continue

        obs_raw = np.asarray(data["obs"])  # (T, obs_dim)
        action_infos = data["action_infos"]

        # -------------------- Feature engineering (keep your logic) --------------------
        goal_block = np.repeat(goal_pos.reshape(1, 3), obs_raw.shape[0], axis=0)
        bread_pos = obs_raw[:, 23:26]
        bread_to_goal = bread_pos - goal_block
        goal_to_bread = goal_block - bread_pos

        obs_enhanced = np.concatenate([obs_raw, goal_block, goal_to_bread], axis=1)


        obs_enhanced = obs_enhanced[:, 14:]
        obs_enhanced = np.concatenate([obs_enhanced[:, :7], obs_enhanced[:, 9:]], axis=1)

        # -------------------- Build sequences --------------------
        T_total = min(len(action_infos), len(obs_enhanced) - 1)

        cur_obs_seq = []
        cur_act = []

        # t 对应要预测的 action；输入用 [t-seq_len+1 ... t] 这一段观测
        for t in range(seq_len - 1, T_total):
            window = obs_enhanced[t - (seq_len - 1) : t + 1]  # (seq_len, D)
            a = np.asarray(action_infos[t]["actions"], dtype=np.float32)

            cur_obs_seq.append(window.astype(np.float32))
            cur_act.append(a)

        if len(cur_obs_seq) > 0:
            episodes_obs.append(np.stack(cur_obs_seq, axis=0))      # (N_i, seq_len, D)
            episodes_actions.append(np.stack(cur_act, axis=0))      # (N_i, action_dim)

    num_episodes = len(episodes_obs)
    if num_episodes == 0:
        raise RuntimeError("No successful episodes found. Check DATA_ROOT / successful flag / files.")

    # ------------------------ Split by trajectory ------------------------
    indices = np.arange(num_episodes)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)

    X_train = np.concatenate([episodes_obs[i] for i in train_idx], axis=0)      # (N, T, D)
    y_train = np.concatenate([episodes_actions[i] for i in train_idx], axis=0)  # (N, A)

    X_val = np.concatenate([episodes_obs[i] for i in val_idx], axis=0)
    y_val = np.concatenate([episodes_actions[i] for i in val_idx], axis=0)

    print("Data Load Done:")
    print(f"Train: {len(train_idx)} trajectories, {X_train.shape[0]} samples")
    print(f"Valid: {len(val_idx)} trajectories, {X_val.shape[0]} samples")
    print(f"Input sequence: (T={X_train.shape[1]}, D={X_train.shape[2]}) | action_dim={y_train.shape[1]}")

    # ------------------------------ Scale (sequence-aware) ------------------------------
    scaler = fit_scaler_on_sequences(X_train)
    X_train = transform_sequences(scaler, X_train)
    X_val = transform_sequences(scaler, X_val)

    joblib.dump(scaler, str(SAVE_DIR / "state_scaler.pkl"))

    # --------------------------- Torch dataset --------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # -------------------------- Train setup --------------------------
    state_dim = X_train.shape[2]
    action_dim = y_train.shape[1]

    model = GRUPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=64, num_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_val = float("inf")
    best_state = None
    epochs_log = []
    train_losses = []
    val_losses = []


    for epoch in range(1, max_epochs + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)   # (B, T, D)
            yb = yb.to(device)   # (B, A)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        scheduler.step(val_loss)
        epochs_log.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), str(SAVE_DIR / "gru_policy.pt"))
    print("[DONE] saved gru_policy.pt (best val =", best_val, ")")
    print("Saved to:", SAVE_DIR)
    save_logs_and_plot(
        save_dir=SAVE_DIR,
        tag="gru_policy",
        epochs=epochs_log,
        train_losses=train_losses,
        val_losses=val_losses,
    )



if __name__ == "__main__":
    main()
