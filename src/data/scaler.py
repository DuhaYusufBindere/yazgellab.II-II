from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.config.config_loader import load_config


def get_scaler(config=None):

    if config is None:
        config = load_config()
    
    # Konfigürasyonda özel olarak belirtilmemişse varsayılan olarak MinMaxScaler seçilir
    scaler_type = config.get("data", {}).get("scaler_type", "minmax")
    
    if scaler_type == "standard":
        print("[INFO] StandardScaler secildi.")
        return StandardScaler()
    else:
        print("[INFO] MinMaxScaler secildi.")
        return MinMaxScaler()
    
def fit_scaler_on_train(scaler, X_train):
    """
    Veri sızıntısını (Data Leakage) önlemek amacıyla scaler nesnesini 
    sadece Train (Eğitim) verisi üzerinde fit eden fonksiyon.
    """
    print("[INFO] Scaler sadece Train verisi uzerinde fit ediliyor...")
    
    # Sadece sayısal (numeric) özellikleri fit edelim (source_file, source_group vs. varsa fit etmemek için)
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    scaler.fit(X_train[numeric_cols])
    return scaler

def transform_with_scaler(scaler, df):

    if df is None:
        return None
        
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_transformed = df.copy()
    
    # Sayısal kolonları ölçeklendir
    df_transformed[numeric_cols] = scaler.transform(df[numeric_cols])
    return df_transformed
