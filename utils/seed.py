import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    Tüm kütüphaneler için (Python random, NumPy, PyTorch) rastgelelik
    üreticilerini (random seed) sabitler, böylece deneyler tamamen
    tekrarlanabilir (reproducible) olur.
    
    Args:
        seed (int): Sabitlenecek rastgele tohum değeri. Varsayılan: 42
    """
    # 1. Python standart kütüphanesi
    random.seed(seed)
    
    # 2. NumPy
    np.random.seed(seed)
    
    # 3. PyTorch (CPU)
    torch.manual_seed(seed)
    
    # 4. PyTorch (GPU / CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # PyTorch'un evrişim (convolution) işlemlerinde deterministik davranması için
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
