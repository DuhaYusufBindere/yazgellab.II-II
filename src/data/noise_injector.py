import numpy as np
import pandas as pd

def add_gaussian_noise(data, mean=0.0, std_dev=0.1, seed=None):
    """
    Veri seti üzerine dışarıdan varyanslı bir Gauss Dağılımı (Gaussian noise) ekler.
    Sistemin zor şartlar altındaki dayanımını (robustness) test etmek için kullanılır.
    
    Args:
        data (numpy.ndarray veya pandas.DataFrame): Gürültü eklenecek orijinal veri.
        mean (float): Gauss dağılımının ortalaması (varsayılan: 0.0).
        std_dev (float): Gauss dağılımının standart sapması (varsayılan: 0.1).
        seed (int, optional): Tekrarlanabilirlik için rastgelelik tohumu.
        
    Returns:
        numpy.ndarray veya pandas.DataFrame: Gürültü eklenmiş yeni veri.
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Gürültü matrisinin oluşturulması
    noise = np.random.normal(loc=mean, scale=std_dev, size=data.shape)
    
    # Veri tipini koruyarak gürültüyü ekle
    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        return data + noise
    else:
        return data + noise
