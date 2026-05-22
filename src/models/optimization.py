import torch.nn as nn
import torch.optim as optim
from src.config.config_loader import load_config


def get_loss_and_optimizer(model, learning_rate=None, config=None):
    """
    Returns:
        criterion (nn.Module): BCEWithLogitsLoss kayıp fonksiyonu.
        optimizer (optim.Optimizer): Adam optimize edici nesnesi.
    """
    if config is None:
        config = load_config()
        
    if learning_rate is None:
        learning_rate = config.get("deep_learning", {}).get("learning_rate", 0.001)
        
    # Sayısal kararlılık (numerical stability) açısından sigmoid işlemini loss ile 
    # bütünleştiren BCEWithLogitsLoss (Binary Cross Entropy) tercih edilmiştir.
    criterion = nn.BCEWithLogitsLoss()
    
    # En yaygın ve kararlı optimizasyon algoritmalarından biri olan Adam optimize edicisi
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return criterion, optimizer
