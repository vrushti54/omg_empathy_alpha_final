# compute_norm_stats.py
# Create per-feature mean/std (shape: 79,) from runs/features/train_X.npy

import os
import numpy as np
import loadconfig
import configparser

def main():
    config_path = loadconfig.load()
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    train_X_path = cfg.get('model', 'training_predictors_load')   # runs/features/train_X.npy
    out_dir      = os.path.dirname(train_X_path)                   # runs/features

    X = np.load(train_X_path)   # (N, T, 79)
    X2 = X.reshape(-1, X.shape[-1])  # (N*T, 79)
    mean = X2.mean(axis=0).astype('float32')
    std  = X2.std(axis=0).astype('float32') + 1e-8

    mean_path = os.path.join(out_dir, 'train_mean.npy')
    std_path  = os.path.join(out_dir, 'train_std.npy')
    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f"[stats] wrote {mean_path} and {std_path}")
    print(f"[stats] mean[:5]={mean[:5]}  std[:5]={std[:5]}")

if __name__ == "__main__":
    main()
