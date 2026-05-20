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
