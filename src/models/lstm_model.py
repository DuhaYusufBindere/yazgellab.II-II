import torch
import torch.nn as nn
from src.config.config_loader import load_config


class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=None, num_layers=2, dropout=0.2, config=None):

        super(LSTMClassifier, self).__init__()
        
        if config is None:
            config = load_config()
            
        if hidden_size is None:
            hidden_size = config.get("deep_learning", {}).get("lstm_hidden_size", 64)
            
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Katmanı (Giriş boyutu: [batch_size, sequence_length, input_size])
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Sınıflandırma Katmanı
        # Son zaman adımındaki LSTM çıkışını ikili sınıflandırma skoru (logit) olarak eşler.
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """
        Returns:
            torch.Tensor: [batch_size] boyutunda anomali skorları (logits).
        """
        # lstm_out boyutu: [batch_size, sequence_length, hidden_size]
        # h_n boyutu: [num_layers, batch_size, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Kayan pencerenin son zaman adımına ait gizli durumu alalım (sequence_length adımının sonuncusu)
        last_out = lstm_out[:, -1, :]
        
        # Lineer katmandan geçirip [batch_size, 1] boyutunu [batch_size] boyutuna sıkıştıralım
        logits = self.fc(last_out).squeeze(-1)
        
        return logits
