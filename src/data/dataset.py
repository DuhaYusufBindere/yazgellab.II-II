import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.config.config_loader import load_config


class TimeSeriesDataset(Dataset):

    def __init__(self, X, y, sequence_length):

        # Verileri numpy dizisine dönüştürelim
        if hasattr(X, "values"):
            self.X = X.values.astype(np.float32)
        else:
            self.X = np.array(X, dtype=np.float32)
            
        if hasattr(y, "values"):
            self.y = y.values.astype(np.float32)
        else:
            self.y = np.array(y, dtype=np.float32)
            
        self.sequence_length = sequence_length
        
        # Geçerli veri uzunluğu kontrolü
        if len(self.X) < self.sequence_length:
            raise ValueError(
                f"Veri uzunluğu ({len(self.X)}), sequence_length değerinden ({self.sequence_length}) küçük olamaz."
            )

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        # idx'ten başlayarak sequence_length kadar zaman adımını kayan pencereyle al
        X_seq = self.X[idx : idx + self.sequence_length]
        # Hedef etiket olarak pencerenin en sonundaki zaman adımının anomali durumunu al
        y_label = self.y[idx + self.sequence_length - 1]
        
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_label, dtype=torch.float32)


def get_dataloader(X, y, sequence_length=None, batch_size=None, shuffle=False, config=None):
    """  
    Returns:
        DataLoader: Hazırlanmış PyTorch DataLoader nesnesi.
    """
    if config is None:
        config = load_config()
        
    if sequence_length is None:
        sequence_length = config.get("deep_learning", {}).get("sequence_length", 10)
        
    if batch_size is None:
        batch_size = config.get("deep_learning", {}).get("batch_size", 32)
        
    dataset = TimeSeriesDataset(X, y, sequence_length=sequence_length)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False
    )
    
    return dataloader
