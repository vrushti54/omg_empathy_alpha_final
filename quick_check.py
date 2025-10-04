import pathlib, pandas as pd, librosa, numpy as np

audio_dir = pathlib.Path(r"C:\omg_data\audio\train")
ann_dir   = pathlib.Path(r"C:\omg_data\annotations\train")

wavs = sorted([p.stem for p in audio_dir.glob("*.wav")])
csvs = sorted([p.stem for p in ann_dir.glob("*.csv")])

wset, cset = set(wavs), set(csvs)
both = sorted(wset & cset)
wa   = sorted(wset - cset)
ca   = sorted(cset - wset)

print(f"audio .wav: {len(wavs)} | csv .csv: {len(csvs)} | paired: {len(both)}")
if wa: print("audio-only (no csv):", wa[:10])
if ca: print("csv-only (no wav):", ca[:10])

if both:
    b = both[0]
    a = audio_dir / f"{b}.wav"
    c = ann_dir / f"{b}.csv"
    y, sr = librosa.load(a, sr=16000, mono=True)
    df = pd.read_csv(c)
    print("SAMPLE:", b)
    print("y len:", len(y), "sr:", sr)
    print("csv cols:", list(df.columns)[:5], "shape:", df.shape)
