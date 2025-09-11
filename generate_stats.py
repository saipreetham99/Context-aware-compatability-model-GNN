# file: generate_stats.py
import numpy as np
from dataloaders import DataLoaderPolyvore
import scipy.sparse as sp  # Import scipy sparse

# This script will load the training data once, calculate the
# mean and standard deviation of the features, and save them to a file.

print("Loading Polyvore data to calculate feature statistics...")

# Initialize the same dataloader used in training
dl = DataLoaderPolyvore()
train_features_raw, _, _, _, _ = dl.get_phase("train")

# --- THIS IS THE FIX ---
# Check if the loaded features are in a sparse format. If so, convert to a dense numpy array.
if sp.issparse(train_features_raw):
    print("Features are in sparse format. Converting to dense array...")
    train_features_dense = train_features_raw.toarray()
else:
    train_features_dense = train_features_raw
# ---------------------

print(f"Features loaded and converted. Shape: {train_features_dense.shape}")

# Calculate the mean and std across all features using the dense array
mean = np.mean(train_features_dense, axis=0)
std = np.std(train_features_dense, axis=0)

# Avoid division by zero if a feature has no variance
std[std == 0] = 1

# Save these crucial statistics to a file
stats_file = "normalization_stats.npz"
np.savez(stats_file, mean=mean, std=std)

print(f"Mean (shape: {mean.shape}) and Std (shape: {std.shape}) saved to {stats_file}")
