# steps/ingest.py
import pandas as pd

def load_file_as_dataframe():
    df = pd.read_csv("data/raw/dataset.csv")
    df = df.dropna(how="all")
    df["target"] = df["target"].astype(int)
    print(f"[INGEST] Linhas={len(df)} | Colunas={df.shape[1]}")
    return df
