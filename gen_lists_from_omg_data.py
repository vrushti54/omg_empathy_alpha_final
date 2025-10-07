import os, glob

DATA = r"C:\omg_data"  # <- update if you ever move the dataset

def write_audio_list(src_dir, out_txt):
    wavs = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    with open(out_txt, "w", encoding="utf-8") as f:
        for p in wavs:
            f.write(p.replace("\\","/") + "\n")
    print(f"Wrote {len(wavs):4d} -> {out_txt}")

def write_pairs(wav_dir, ann_dir, out_txt):
    wavs = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    n_match = 0
    with open(out_txt, "w", encoding="utf-8") as f:
        for w in wavs:
            base = os.path.splitext(os.path.basename(w))[0]      # Subject_X_Story_Y
            csv  = os.path.join(ann_dir, base + ".csv")
            if os.path.isfile(csv):
                f.write(w.replace("\\","/") + "\t" + csv.replace("\\","/") + "\n")
                n_match += 1
            else:
                # fallback: write only wav path if annotation missing
                f.write(w.replace("\\","/") + "\n")
    print(f"Wrote pairs ({n_match} with labels) -> {out_txt}")

# paths in your structured dataset
train_wav = os.path.join(DATA, "audio", "train")
val_wav   = os.path.join(DATA, "audio", "val")
train_ann = os.path.join(DATA, "annotations", "train")
val_ann   = os.path.join(DATA, "annotations", "val")

# emit the lists into repo root (what many scripts expect)
write_audio_list(train_wav, "audio_train.txt")
write_audio_list(val_wav,   "audio_val.txt")

write_pairs(train_wav, train_ann, "ann_train.txt")
write_pairs(val_wav,   val_ann,   "ann_val.txt")
