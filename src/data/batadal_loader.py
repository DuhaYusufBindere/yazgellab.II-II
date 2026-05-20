import pandas as pd
import os
from src.config.config_loader import load_config


def load_batadal_training(config=None):
    """
    Returns:
        pd.DataFrame: Yüklenmiş BATADAL eğitim veri seti.
    """
    if config is None:
        config = load_config()

    batadal_path = config["data"]["batadal_path"]

    df = pd.read_csv(batadal_path)

    df.columns = df.columns.str.strip()

    print(f"BATADAL Training Dataset 2 yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"Kaynak dosya: {batadal_path}")
    print(f"Sütunlar: {df.columns.tolist()}")

    return df

def drop_time_columns(df):
    """
    Returns:
        pd.DataFrame: Zaman kolonları çıkarılmış dataframe.
    """
    time_cols = ["DATETIME"]
    existing = [c for c in time_cols if c in df.columns]
    if existing:
        print(f"Zaman kolonları çıkarıldı: {existing}")
    return df.drop(columns=existing)
