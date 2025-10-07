# show_compare_plot.py
# Create a summary.png bar chart for a compare run, saved quietly (no GUI).

import os
import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure no GUI popups
import matplotlib.pyplot as plt

def find_latest_compare_dir(base_dir):
    pattern = os.path.join(base_dir, "run_*")
    runs = glob.glob(pattern)
    if not runs:
        return None
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return runs[0]

def pick_mean_col(df):
    # be tolerant to different column names/cases
    lower_map = {c.lower(): c for c in df.columns}
    for key in ["mean_ccc", "mean ccc", "mean"]:
        if key in lower_map:
            return lower_map[key]
    # sometimes the CSV may store means per variant row, else compute from per-file CSVs
    return None

def main():
    # 1) resolve run directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    compare_root = os.path.join(project_root, "speech", "runs", "compare")

    # optional arg: path to a specific compare run dir
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = find_latest_compare_dir(compare_root)

    if not run_dir or not os.path.isdir(run_dir):
        print(f"[error] Could not find compare run directory under: {compare_root}")
        sys.exit(1)

    master_csv = os.path.join(run_dir, "compare_models.csv")
    if not os.path.exists(master_csv):
        print(f"[error] No compare_models.csv in: {run_dir}")
        sys.exit(1)

    print(f"[info] Loading: {master_csv}")
    df = pd.read_csv(master_csv)

    # 2) Normalize column names we’ll use
    # Expecting at least: model, variant, and some mean column
    # Try to locate model/variant robustly
    colmap = {c.lower(): c for c in df.columns}
    model_col = colmap.get("model") or colmap.get("checkpoint") or list(df.columns)[0]
    variant_col = colmap.get("variant") if "variant" in colmap else None

    mean_col = pick_mean_col(df)
    if mean_col is None:
        # Fallback: try to compute mean from a column named 'ccc' if already aggregated;
        # otherwise bail out with a helpful message.
        possible_ccc = [c for c in df.columns if c.lower() in ("ccc", "avg_ccc", "ccc_mean")]
        if possible_ccc:
            mean_col = possible_ccc[0]
        else:
            # As a last resort, try to compute mean from per-variant summary CSVs
            # if the master CSV has a 'summary_csv' column with file paths.
            if "summary_csv" in df.columns:
                means = []
                for _, row in df.iterrows():
                    s = row["summary_csv"]
                    if os.path.exists(s):
                        sdf = pd.read_csv(s)
                        # look for per-file CCC column
                        ccc_col = None
                        for candidate in sdf.columns:
                            if candidate.lower() in ("ccc", "ccc_val", "ccc_value"):
                                ccc_col = candidate
                                break
                        if ccc_col is None:
                            means.append(float("nan"))
                        else:
                            means.append(float(sdf[ccc_col].mean()))
                    else:
                        means.append(float("nan"))
                df["mean_ccc"] = means
                mean_col = "mean_ccc"
            else:
                print("[error] Could not find a mean CCC column in compare_models.csv "
                      "and no summary_csv to compute from.")
                sys.exit(1)

    # 3) Build a nice x-label = model (and variant if present)
    if variant_col:
        df["_label"] = df[model_col].astype(str) + "\n" + df[variant_col].astype(str)
    else:
        df["_label"] = df[model_col].astype(str)

    # Sort bars by mean (descending) so the best is first
    plot_df = df[["_label", mean_col]].copy()
    plot_df = plot_df.sort_values(mean_col, ascending=False).reset_index(drop=True)

    # 4) Plot quietly and save
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(plot_df)), plot_df[mean_col].values)
    plt.xticks(range(len(plot_df)), plot_df["_label"].values, rotation=30, ha="right")
    plt.ylabel("Mean CCC")
    plt.title("Model × Variant Comparison (higher is better)")
    plt.tight_layout()

    out_png = os.path.join(run_dir, "summary.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

    # Also save a short text summary
    top = plot_df.iloc[0]
    with open(os.path.join(run_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best: {top['_label']}  |  Mean CCC = {top[mean_col]:.4f}\n")
        f.write(f"Total variants compared: {len(plot_df)}\n")

    print(f"[done] Wrote: {out_png}")
    print(f"[done] Wrote: {os.path.join(run_dir, 'summary.txt')}")

if __name__ == "__main__":
    main()
