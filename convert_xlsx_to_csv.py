import pandas as pd
import pathlib

train_ann = pathlib.Path(r"C:\omg_data\annotations\train")
val_ann = pathlib.Path(r"C:\omg_data\annotations\val")

def convert_all(folder):
    for f in folder.glob("*.xlsx"):
        csv_path = f.with_suffix(".csv")
        try:
            df = pd.read_excel(f)
            df.to_csv(csv_path, index=False)
            print(f"Converted {f.name} -> {csv_path.name}")
        except Exception as e:
            print(f"Failed {f.name}: {e}")

convert_all(train_ann)
convert_all(val_ann)
