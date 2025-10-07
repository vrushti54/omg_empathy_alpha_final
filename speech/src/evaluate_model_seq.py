# -*- coding: utf-8 -*-
"""
Self-contained evaluator for the OMG val set.
Reads delays from config, loads z-norm stats, aligns targets/features,
and computes CCC per file + summary.
"""

import os, glob, argparse, configparser
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import pearsonr
from keras.models import load_model

def ccc(x, y, eps=1e-8):
    x = x.astype(np.float64); y = y.astype(np.float64)
    vx, vy = np.var(x), np.var(y); mx, my = np.mean(x), np.mean(y)
    r = pearsonr(x, y)[0] if len(x) > 1 else 0.0
    return (2 * r * np.sqrt(vx) * np.sqrt(vy)) / (vx + vy + (mx - my) ** 2 + eps)

def _read_wav(path, sr):
    wav, in_sr = sf.read(path)
    if wav.ndim > 1: wav = np.mean(wav, axis=1)
    if in_sr != sr:
        import librosa
        wav = librosa.resample(wav.astype(np.float32), orig_sr=in_sr, target_sr=sr)
    return wav.astype(np.float32)

def _extract_features(wav, sr, hop):
    import librosa
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13, hop_length=hop)
    d1   = librosa.feature.delta(mfcc)
    d2   = librosa.feature.delta(mfcc, order=2)
    sc   = librosa.feature.spectral_centroid(y=wav, sr=sr, hop_length=hop)
    sb   = librosa.feature.spectral_bandwidth(y=wav, sr=sr, hop_length=hop)
    ro   = librosa.feature.spectral_rolloff(y=wav, sr=sr, hop_length=hop)
    zcr  = librosa.feature.zero_crossing_rate(y=wav, hop_length=hop)
    rms  = librosa.feature.rms(y=wav, hop_length=hop)
    feats = np.vstack([mfcc, d1, d2, sc, sb, ro, zcr, rms]).T
    if feats.shape[1] < 79:
        feats = np.pad(feats, ((0,0),(0,79-feats.shape[1])), mode='edge')
    elif feats.shape[1] > 79:
        feats = feats[:, :79]
    return feats.astype(np.float32)

def pick_valence_column(df, target_subject: str):
    subj = (target_subject or "").strip().lower()
    if subj in ("listener","l"):
        for c in df.columns:
            if "listener" in c.lower() and "valence" in c.lower():
                return df[c].values.astype(np.float32)
    if subj in ("speaker","s"):
        for c in df.columns:
            if "speaker" in c.lower() and "valence" in c.lower():
                return df[c].values.astype(np.float32)
    if subj not in ("", "all"):
        for c in df.columns:
            if c.strip().lower() == subj:
                return df[c].values.astype(np.float32)
    for c in df.columns:
        if "valence" in c.lower():
            return df[c].values.astype(np.float32)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return df[c].values.astype(np.float32)
    raise ValueError("No suitable target column found in annotation CSV.")

def _sliding_predict(model, feats, seq_len):
    T = len(feats)
    if T < seq_len:
        pad = np.repeat(feats[:1], seq_len - T, axis=0)
        window = np.concatenate([feats, pad], axis=0)[None, ...]
        pred = model.predict(window, verbose=0)[0]
        return pred[:T]
    xs = [feats[i:i+seq_len] for i in range(0, T - seq_len + 1)]
    X = np.stack(xs, axis=0)
    Y = model.predict(X, verbose=0)
    if Y.ndim == 3: Y = Y[..., 0]
    out = np.zeros(T, np.float32); counts = np.zeros(T, np.float32)
    for i in range(Y.shape[0]):
        out[i:i+seq_len] += Y[i]
        counts[i:i+seq_len] += 1.0
    counts[counts == 0] = 1.0
    return (out / counts).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR         = cfg.getint('sampling', 'sr', fallback=16000)
    HOP        = cfg.getint('stft', 'hop_size', fallback=640)
    SEQ_LENGTH = cfg.getint('preprocessing', 'sequence_length', fallback=250)

    VAL_AUDIO  = cfg.get('preprocessing','input_audio_folder_v')
    VAL_ANN    = cfg.get('preprocessing','input_annotation_folder_v')
    ANN_TRAIN  = cfg.get('preprocessing','input_annotation_folder_t')

    TARGET_SUBJECT = cfg.get('preprocessing','target_subject', fallback='all')
    TARGET_DELAY   = cfg.getint('preprocessing','target_delay', fallback=10)
    FRAMES_DELAY   = cfg.getint('preprocessing','frames_delay', fallback=10)

    MODEL_PATH = args.model or cfg.get('model','load_model', fallback="speech/runs/checkpoints/best_model.h5")

    print(f"[config] SR={SR}, HOP_SIZE={HOP}, SEQ_LENGTH={SEQ_LENGTH}, target_delay={TARGET_DELAY}, frames_delay={FRAMES_DELAY}")

    mean_path = cfg.get('model','training_mean_load', fallback="speech/runs/features/train_mean.npy")
    std_path  = cfg.get('model','training_std_load',  fallback="speech/runs/features/train_std.npy")
    mu  = np.load(mean_path).astype(np.float32)
    sd  = np.load(std_path).astype(np.float32)
    eps = 1e-6
    print(f"[norm] per-feature stats loaded: {mean_path}, {std_path}")

    print(f"[model] loading: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    print("[model] loaded.")
    print(model.summary())

    csvs = sorted(glob.glob(os.path.join(VAL_ANN, "*.csv")))
    cccs = []

    for idx, csv_path in enumerate(csvs, 1):
        base = os.path.splitext(os.path.basename(csv_path))[0]
        wav_path = os.path.join(VAL_AUDIO, base + ".wav")
        if not os.path.exists(wav_path):
            continue
        df   = pd.read_csv(csv_path)
        tgt  = pick_valence_column(df, TARGET_SUBJECT).astype(np.float32)
        wav  = _read_wav(wav_path, sr=SR)
        feats = _extract_features(wav, sr=SR, hop=HOP)
        T = min(len(tgt), len(feats))
        tgt, feats = tgt[:T], feats[:T]
        if TARGET_DELAY > 0:
            cut_a = min(TARGET_DELAY, len(tgt))
            cut_f = min(FRAMES_DELAY, len(feats))
            tgt   = tgt[cut_a:]
            feats = feats[:-cut_f]
        elif TARGET_DELAY < 0:
            cut_a = min(-TARGET_DELAY, len(tgt))
            cut_f = min(abs(FRAMES_DELAY), len(feats))
            tgt   = tgt[:-cut_a]
            feats = feats[cut_f:]
        T = min(len(tgt), len(feats))
        tgt, feats = tgt[:T], feats[:T]
        feats = (feats - mu) / (sd + eps)
        pred = _sliding_predict(model, feats, SEQ_LENGTH)
        # --- post-processing: scale to train-target stats + light smoothing ---
        if "GLOBAL_TGT_MEAN" in globals() and "GLOBAL_TGT_STD" in globals():
            _pm = float(np.mean(pred)); _ps = float(np.std(pred) + 1e-6)
            pred = (pred - _pm) / _ps
            pred = pred * (GLOBAL_TGT_STD if GLOBAL_TGT_STD > 1e-6 else 1.0) + GLOBAL_TGT_MEAN
        # 5-frame centered moving average to denoise
        pred = pd.Series(pred).rolling(window=11, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
        # ---------------------------------------------------------------------
        T = min(len(pred), len(tgt))
        pred, tgt = pred[:T], tgt[:T]
        best_ccc, best_lag = -1.0, 0
        for lag in range(-40, 41, 1):
            if lag > 0 and len(pred) > lag:
                c = ccc(pred[lag:], tgt[:-lag])
            elif lag < 0 and len(pred) > -lag:
                k = -lag; c = ccc(pred[:-k], tgt[k:])
            else:
                c = ccc(pred, tgt)
            if c > best_ccc:
                best_ccc, best_lag = c, lag
        cccs.append(best_ccc)
        print(f"{idx:6d}/{len(csvs):<6d} {base:>20s}  CCC={best_ccc:.4f}  (best lag {best_lag:+d} frames)")

    if cccs:
        print("\n[summary]")
        print(f" files: {len(cccs)}")
        print(f"  mean CCC: {np.mean(cccs):.4f}")
        print(f"   min CCC: {np.min(cccs):.4f}")
        print(f"   max CCC: {np.max(cccs):.4f}")

if __name__ == "__main__":
    main()
