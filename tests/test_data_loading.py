import numpy as np
import pickle
import os

# === Adjust these paths to match your directory layout ===
paths = {
    "X_full_reproduce": "data/new_reproduced/X_full_reproduce.npy",
    "X_original" : "data/processed/X.npy",
    "y_full_reproduce": "data/new_reproduced/y_full_reproduce.pkl",
    "y_original" : "data/processed/y.pkl",
}

# 1) Load and print shapes for X
if os.path.exists(paths["X_full_reproduce"]):
    X_rep = np.load(paths["X_full_reproduce"])
    print(f"X_full_reproduce shape: {X_rep.shape}")
else:
    print(f"Missing file: {paths['X_full_reproduce']}")

if os.path.exists(paths["X_original"]):
    X_org = np.load(paths["X_original"])
    print(f"X_original  shape: {X_org.shape}")
else:
    print(f"Missing file: {paths['X_original']}")

# 2) Load and print shapes for y
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

if os.path.exists(paths["y_full_reproduce"]):
    y_rep = load_pickle(paths["y_full_reproduce"])
    # ensure it’s a NumPy array or list
    y_rep_arr = np.array(y_rep)
    print(f"y_full_reproduce shape: {y_rep_arr.shape}")
else:
    print(f"Missing file: {paths['y_full_reproduce']}")

if os.path.exists(paths["y_original"]):
    y_org = load_pickle(paths["y_original"])
    y_org_arr = np.array(y_org)
    print(f"y_original  shape: {y_org_arr.shape}")
else:
    print(f"Missing file: {paths['y_original']}")

# Optionally, print a few sample values to verify content
print("\nSample X_full_reproduce[0, :, :5]:", X_rep[0, :, :5] if 'X_rep' in locals() else None)
print("Sample X_original [0, :, :5]:", X_org[0, :, :5] if 'X_org' in locals() else None)

print("\nSample y_full_reproduce[:5]:", y_rep_arr[:5] if 'y_rep_arr' in locals() else None)
print("Sample y_original  [:5]:", y_org_arr[:5]  if 'y_org_arr' in locals() else None)
