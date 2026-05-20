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

def separate_target(df):
    """
    Returns:
        X (pd.DataFrame): Sensör değerlerini içeren özellik matrisi.
        y (pd.Series): 0 ve 1 olarak kodlanmış anomali etiketleri.
    """
    target_col = "ATT_FLAG"
    if target_col not in df.columns:
        raise KeyError(f"Hedef kolon '{target_col}' veri setinde bulunamadı.")
    
    # Hedef değişkeni kopyala ve parse et (-999 -> 0, 1 -> 1)
    y = df[target_col].copy()
    y = y.map({-999: 0, 1: 1})
    
    X = df.drop(columns=[target_col])
    return X, y

def ensure_temporal_order(df):
    """
    Returns:
        pd.DataFrame: Kronolojik sırası garanti edilmiş dataframe.
    """
    if "DATETIME" in df.columns:
        # Tarih formatı: '04/07/16 00' (Gün/Ay/Yıl Saat)
        temp_dt = pd.to_datetime(df["DATETIME"], format="%d/%m/%y %H")
        if not temp_dt.is_monotonic_increasing:
            print("⚠️ Uyarı: Veri kronolojik sırada değil! Sıralanıyor...")
            df = df.iloc[temp_dt.argsort()].reset_index(drop=True)
        else:
            print("✅ Veri sırası kontrol edildi: Kronolojik sıra (Temporal Order) korundu.")
    else:
        if not df.index.is_monotonic_increasing:
            print("⚠️ Uyarı: Veri indeks sırası bozuk! Sıralanıyor...")
            df = df.sort_index().reset_index(drop=True)
        else:
            print("✅ Veri sırası kontrol edildi: İndeks sıralaması korundu.")
            
    return df
