# export_prediction.py
# Save per-frame predictions for each file in the validation split.
# Outputs:
#   speech/runs/exports/<run_timestamp>/
#       ├── csv/<stem>.csv          (frame_idx, time_s, target, pred_*)
#       ├── plots/<stem>.png        (GT vs prediction)
#       └── summary.csv             (per-file CCC for each variant)

import os
import sys
import time
import glob
import argparse
import configparser
import numpy as np
import pandas as pd

# Use non-interactive backend so plots are saved quietly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import filtfilt, butter

# Keras / TF1.x style (your current env)
from keras.models import load_model
from keras import backend as K

# Project utils
import loadconfig
import utilities_func as uf
import feat_analysis2 as fa
from calculateCCC import ccc2

# ---------------------------
# Config / constants
# ---------------------------
cfg = configparser.ConfigParser()
cfg.read(loadconfig.load())

SR         = cfg.getint('sampling', 'sr')                              # 16000
HOP_SIZE   = cfg.getint('stft', 'hop_size')                            # 640
SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length')            # 250
FPS        = 25.0

MODEL_PATH   = cfg.get('model', 'load_model')                          # e.g., speech/runs/checkpoints/best_model.h5
TRAIN_X_PATH = cfg.get('model', 'training_predictors_load')

VAL_AUDIO = cfg.get('preprocessing', 'input_audio_folder_v')
VAL_ANN   = cfg.get('preprocessing', 'input_annotation_folder_v')
ANN_TRAIN = cfg.get('preprocessing', 'input_annotation_folder_t')      # for f_trick stats

frames_per_annotation = (SR / FPS) / float(HOP_SIZE)
print(f"[config] SR={SR}, HOP_SIZE={HOP_SIZE}, FPS={FPS:.0f}, frames_per_annotation={frames_per_annotation:.3f}")
print(f"[config] SEQ_LENGTH={SEQ_LENGTH}")

# ---------------------------
# Normalization stats
# ---------------------------
feat_dir = os.path.dirname(TRAIN_X_PATH)
mean_path = os.path.join(feat_dir, 'train_mean.npy')
std_path  = os.path.join(feat_dir, 'train_std.npy')

if os.path.exists(mean_path) and os.path.exists(std_path):
    ref_mean = np.load(mean_path).astype('float32')   # (79,)
    ref_std  = np.load(std_path).astype('float32') + 1e-8
    print(f"[norm] per-feature stats loaded: {mean_path}, {std_path}")
else:
    ref = np.load(TRAIN_X_PATH)
    ref_mean = np.float32(ref.mean())
    ref_std  = np.float32(ref.std() + 1e-8)
    print(f"[norm] fallback to scalar stats from {TRAIN_X_PATH}: "
          f"mean={ref_mean:.6f} std={ref_std:.6f} -> broadcast to per-feature at runtime")

def normalize_feats(feats: np.ndarray) -> np.ndarray:
    if np.ndim(ref_mean) == 1 and ref_mean.shape[0] == feats.shape[1]:
        return (feats - ref_mean) / ref_std
    return (feats - ref_mean) / ref_std

# ---------------------------
# Model loader (safe even if saved with custom loss)
# ---------------------------
def batch_CCC(y_true, y_pred):  # for loader only (same alias as other scripts)
    return uf.CCC(y_true, y_pred)

def load_keras_model_safe(path: str):
    try:
        # Works when the model was saved with metric names only
        return load_model(path, custom_objects={'CCC': uf.CCC, 'batch_CCC': batch_CCC})
    except Exception as e:
        print(f"[warn] Standard load failed ({e}); retrying with compile=False")
        return load_model(path, custom_objects={'CCC': uf.CCC, 'batch_CCC': batch_CCC}, compile=False)

model = load_keras_model_safe(MODEL_PATH)
print("[model] loaded:", MODEL_PATH)
try:
    print(model.summary())
except Exception:
    pass

# Optional latent extractor (kept for compatibility)
_ = K.function(inputs=[model.input], outputs=[model.layers[-2].output])

# Global target stats for f_trick mapping
t_mean_global, t_std_global = uf.find_mean_std(ANN_TRAIN)
print(f"[f_trick] global target stats from train annotations: mean={t_mean_global:.6f}, std={t_std_global:.6f}")

# ---------------------------
# Helpers
# ---------------------------
def find_audio(audio_dir: str, stem: str):
    c1 = os.path.join(audio_dir, stem + ".mp4.wav")
    c2 = os.path.join(audio_dir, stem + ".wav")
    if os.path.exists(c1): return c1
    if os.path.exists(c2): return c2
    return None

def sliding_predict(feats: np.ndarray, seq_len: int) -> np.ndarray:
    """Run sliding-window prediction over features, return sequence len(target)-long raw preds."""
    preds = []
    return np.array(preds)

def predict_one(wav_path: str, csv_path: str, seq_len: int):
    """
    Returns:
      target              (T,)
      pred_baseline       (T,)
      pred_ftrick         (T,)
      pred_smooth         (T,)
      pred_ftrick_smooth  (T,)
      best_ccc_variant    dict: {variant: (ccc, best_lag)}
    """
    # 1) Audio -> features with SAME hop as training
    sr, samples = uf.wavread(wav_path)
    e_samples   = uf.preemphasis(samples, sr)
    feats       = fa.extract_features(e_samples, sr=sr, hop_samples=HOP_SIZE)  # (F, 79)
    feats       = normalize_feats(feats)

    # 2) Target (ground-truth)
    target = pd.read_csv(csv_path).values.reshape(-1).astype('float32')
    T = len(target)

    # 3) Sliding-window predictions aligned to GT frames
    preds = []
    start = 0
    while start < (T - seq_len):
        s_feat = int(start * frames_per_annotation)
        e_feat = int((start + seq_len) * frames_per_annotation)
        window = feats[s_feat:e_feat].reshape(1, -1, feats.shape[1])
        pred = model.predict(window, verbose=0)[0]  # (seq_len,)
        preds.extend(pred.tolist())
        start += seq_len

    # tail to cover last chunk
    tail_len = int(seq_len * frames_per_annotation)
    tail = feats[-tail_len:].reshape(1, -1, feats.shape[1])
    tail_pred = model.predict(tail, verbose=0)[0]
    missing = T - len(preds)
    if missing > 0:
        preds.extend(tail_pred[-missing:].tolist())

    preds = np.asarray(preds, dtype='float32')

    # 4) Post-processing variants
    def _lowpass(x):
        b, a = butter(3, 0.01, 'low')
        return filtfilt(b, a, x)

    # raw network output assumed ~[0,1] -> back to [-1, 1]
    pred_baseline = (preds * 2.0) - 1.0
    pred_ftrick   = uf.f_trick(pred_baseline.copy(), t_mean_global, t_std_global)
    pred_smooth   = _lowpass(pred_baseline.copy())
    pred_ftrick_smooth = _lowpass(uf.f_trick(pred_baseline.copy(), t_mean_global, t_std_global))

    # 5) Per-variant best lag (±10 frames) and CCC
    def best_ccc_and_lag(p):
        best_ccc, best_k = -1.0, 0
        for k in range(-10, 11):
            if k < 0:
                t = target[-k:]
                q = p[:len(t)]
            elif k > 0:
                t = target[:-k]
                q = p[k:len(target)]
            else:
                t = target
                q = p
            c = float(ccc2(q, t))
            if c > best_ccc:
                best_ccc, best_k = c, k
        return best_ccc, best_k

    bests = {
        'baseline':        best_ccc_and_lag(pred_baseline),
        'ftrick':          best_ccc_and_lag(pred_ftrick),
        'smooth':          best_ccc_and_lag(pred_smooth),
        'ftrick_smooth':   best_ccc_and_lag(pred_ftrick_smooth),
    }

    return (target,
            pred_baseline, pred_ftrick, pred_smooth, pred_ftrick_smooth,
            bests)

# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Export per-frame predictions and plots.")
    parser.add_argument("--audio_dir", default=VAL_AUDIO, help="Folder with validation audio")
    parser.add_argument("--ann_dir",   default=VAL_ANN,   help="Folder with validation annotation CSVs")
    parser.add_argument("--model",     default=MODEL_PATH, help="Keras .h5 model path (if you want to override)")
    parser.add_argument("--outdir",    default=None, help="Output root (default: speech/runs/exports/<timestamp>)")
    parser.add_argument("--no_plots",  action="store_true", help="Skip PNG plot generation")
    args = parser.parse_args()

    # Output structure
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    out_root = args.outdir or os.path.join("speech", "runs", "exports", f"run_{run_tag}")
    out_csv_dir   = os.path.join(out_root, "csv")
    out_plot_dir  = os.path.join(out_root, "plots")
    os.makedirs(out_csv_dir, exist_ok=True)
    if not args.no_plots:
        os.makedirs(out_plot_dir, exist_ok=True)

    # Iterate over CSV list
    csv_files = sorted([f for f in os.listdir(args.ann_dir) if f.lower().endswith(".csv")])
    if not csv_files:
        print("[warn] No CSV files found for export.")
        return

    summary_rows = []
    print(f"[info] exporting to: {out_root}")

    for idx, csv_name in enumerate(csv_files, 1):
        stem = os.path.splitext(csv_name)[0]
        wav_path = find_audio(args.audio_dir, stem)
        csv_path = os.path.join(args.ann_dir, csv_name)
        if wav_path is None:
            print(f"{idx:3d}/{len(csv_files):3d}  {stem:>18s}  SKIP: missing audio")
            continue

        try:
            (target,
             pred_b, pred_f, pred_s, pred_fs,
             bests) = predict_one(wav_path, csv_path, SEQ_LENGTH)

            # Save per-frame CSV
            T = len(target)
            frame_idx = np.arange(T, dtype=np.int32)
            time_s = frame_idx / FPS
            df = pd.DataFrame({
                "frame_idx": frame_idx,
                "time_s": time_s,
                "target": target.astype("float32"),
                "pred_baseline":      pred_b.astype("float32"),
                "pred_ftrick":        pred_f.astype("float32"),
                "pred_smooth":        pred_s.astype("float32"),
                "pred_ftrick_smooth": pred_fs.astype("float32"),
            })
            out_csv = os.path.join(out_csv_dir, f"{stem}.csv")
            df.to_csv(out_csv, index=False)

            # Quiet plot (GT vs ftrick_smooth)
            if not args.no_plots:
                plt.figure(figsize=(11, 3.2))
                plt.plot(target, label="target", linewidth=1.0, alpha=0.95)
                plt.plot(pred_fs, label="prediction (ftrick+smooth)", linewidth=1.0, alpha=0.85)
                plt.title(f"{stem} | CCC(fs)={bests['ftrick_smooth'][0]:.4f}, lag={bests['ftrick_smooth'][1]:+d}")
                plt.legend()
                plt.tight_layout()
                out_png = os.path.join(out_plot_dir, f"{stem}.png")
                plt.savefig(out_png, dpi=130)
                plt.close()

            # Add to summary
            summary_rows.append({
                "stem": stem,
                "ccc_baseline":      bests["baseline"][0],
                "lag_baseline":      bests["baseline"][1],
                "ccc_ftrick":        bests["ftrick"][0],
                "lag_ftrick":        bests["ftrick"][1],
                "ccc_smooth":        bests["smooth"][0],
                "lag_smooth":        bests["smooth"][1],
                "ccc_ftrick_smooth": bests["ftrick_smooth"][0],
                "lag_ftrick_smooth": bests["ftrick_smooth"][1],
            })

            print(f"{idx:3d}/{len(csv_files):3d}  {stem:>18s}  saved -> {os.path.basename(out_csv)}")

        except Exception as e:
            print(f"{idx:3d}/{len(csv_files):3d}  {stem:>18s}  ERROR: {e}")

    # Save summary.csv
    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.loc["mean"] = sdf.mean(numeric_only=True)
        out_sum = os.path.join(out_root, "summary.csv")
        sdf.to_csv(out_sum, index=True)
        print(f"[done] summary.csv -> {out_sum}")
    else:
        print("[warn] No files processed; no summary written.")

if __name__ == "__main__":
    main()
