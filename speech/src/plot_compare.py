# plot_compare.py
# Make bar charts of mean CCC per variant for each model
# Input: speech/runs/compare/run_*/compare_models.csv

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

def latest_master_csv():
    base = os.path.join("speech","runs","compare")
    runs = sorted(glob.glob(os.path.join(base, "run_*")))
    if not runs:
        raise RuntimeError(f"No runs found in {base}. Did you run compare_runs.py?")
    latest = runs[-1]
    csv_path = os.path.join(latest, "compare_models.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"compare_models.csv not found in {latest}")
    return csv_path, latest

def main(csv_arg=None):
    if csv_arg is None:
        csv_path, out_dir = latest_master_csv()
    else:
        csv_path = csv_arg
        out_dir = os.path.dirname(csv_path)
    print("[load]", csv_path)

    df = pd.read_csv(csv_path)
    # keep only rows with data
    df = df[df["n_files"] > 0].copy()
    if df.empty:
        raise RuntimeError("Master CSV has no rows with n_files > 0.")

    # sort for consistent plotting
    df = df.sort_values(["model","variant"])

    # one figure per model
    for model, sub in df.groupby("model"):
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
        ax.bar(sub["variant"], sub["mean_CCC"])
        ax.set_title(f"Mean CCC by Variant\n{model}")
        ax.set_ylabel("Mean CCC")
        ax.set_xlabel("Variant")
        ax.set_ylim(0, max(0.0001, sub["mean_CCC"].max()*1.15))
        plt.tight_layout()
        png = os.path.join(out_dir, f"{model}_meanCCC.png")
        plt.savefig(png, dpi=150)
        print("[save]", png)
        # Optionally show the plot window for the first model only:
        # plt.show()
        plt.close(fig)

    # also a combined chart (all models)
    pivot = df.pivot(index="model", columns="variant", values="mean_CCC").fillna(0.0)
    fig = plt.figure(figsize=(8,4))
    ax = fig.gca()
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Mean CCC by Variant (All Models)")
    ax.set_ylabel("Mean CCC")
    plt.tight_layout()
    all_png = os.path.join(out_dir, "ALL_MODELS_meanCCC.png")
    plt.savefig(all_png, dpi=150)
    print("[save]", all_png)
    plt.close(fig)

if __name__ == "__main__":
    # optional: python plot_compare.py path\to\compare_models.csv
    csv_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_arg)
