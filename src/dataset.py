import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression


DATA_ROOT = "/Users/hanzhuojun/WorkSpace/FrankaImitation/data"

all_obs = []
all_actions = []

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

    if not data["successful"]:
        continue

    obs = data["obs"]
    action_infos = data["action_infos"]

    T = min(len(action_infos), len(obs) - 1)

    for t in range(T):
        s = obs[t]
        info = action_infos[t]

        a = np.asarray(action_infos[t]["actions"], dtype=np.float32)
        assert a.shape == (7,), a.shape

        all_obs.append(s)
        all_actions.append(a)

all_obs = np.stack(all_obs)
all_actions = np.stack(all_actions)

# remove_idx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,23,24,25,26,27,28,29,30,31,32,33,34,35,36,46,47,48,49,50,51,58,59,60,61,62,63,64,65,66,67,68,69]

# all_obs = all_obs[:, 1:]    
# all_obs = np.delete(all_obs, remove_idx, axis=1)

X = all_obs
y = all_actions


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])


# model = Lasso(alpha=1e-3)
# model = ElasticNet(alpha=1e-3, l1_ratio=0.1)
# model = DecisionTreeRegressor(
#     max_depth=8,          
#     min_samples_leaf=20,  # 保证叶子里有“统计意义”
#     random_state=42,
# )

model.fit(X_train, y_train)

joblib.dump(model, "ridge_policy.pkl")
# print("Model saved to ridge_policy.pkl")


train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("R^2 train:", train_score)
print("R^2 test :", test_score)
