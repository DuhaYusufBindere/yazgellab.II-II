import torch
import torch.nn as nn
from src.config.config_loader import load_config


class CNN1DClassifier(nn.Module):

    def __init__(self, input_size, cnn_filters=None, kernel_size=3, dropout=0.2, config=None):

        super(CNN1DClassifier, self).__init__()
        
        if config is None:
            config = load_config()
            
        if cnn_filters is None:
            cnn_filters = config.get("deep_learning", {}).get("cnn_filters", 64)
            
        # 1D Evrişim Katmanı
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Giriş boyutunu korumak için padding ayarı
        )
        self.relu = nn.ReLU()
        
        # Global Adaptive Average Pooling ile zamansal boyutu 1'e indirgeyerek esneklik sağlıyoruz
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        
        # Sınıflandırma Katmanı
        self.fc = nn.Linear(cnn_filters, 1)
        
    def forward(self, x):
        """
        Returns:
            torch.Tensor: [batch_size] boyutunda anomali skorları (logits).
        """
        # PyTorch Conv1d girdisi [batch_size, in_channels, sequence_length] formatında olmalıdır.
        # Bu yüzden girdi tensörünü [batch_size, input_size, sequence_length] olarak transpose edelim.
        x = x.transpose(1, 2)
        
        # Evrişim + Aktivasyon
        x = self.conv1(x)
        x = self.relu(x)
        
        # Global Pooling [batch_size, cnn_filters, 1]
        x = self.pool(x)
        
        # Boyutu sıkıştıralım [batch_size, cnn_filters]
        x = x.squeeze(-1)
        x = self.dropout(x)
        
        # Lineer katmandan geçirip [batch_size, 1] boyutunu [batch_size] boyutuna sıkıştıralım
        logits = self.fc(x).squeeze(-1)
        
        return logits
