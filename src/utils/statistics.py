import numpy as np
from scipy.stats import wilcoxon
import warnings

def perform_wilcoxon_test(dl_scores, automata_scores):
    """
    Deep Learning ve Otomata modelinin başarı skorları (örneğin F1-score dizileri)
    arasında istatistiksel olarak anlamlı bir fark olup olmadığını 
    Wilcoxon Signed-Rank Test ile (Eşleştirilmiş İki Örneklem) ölçer.
    
    Parametreler:
        dl_scores (list/np.array): DL modelinin deney skorları (örn: 50 deneylik F1 listesi)
        automata_scores (list/np.array): Otomata modelinin aynı deneylerdeki skorları.
        
    Dönüş:
        p_value (float): İstatistiksel anlamlılık değeri.
    """
    if len(dl_scores) != len(automata_scores):
        raise ValueError("[HATA] Wilcoxon testi eşleştirilmiş bir testtir. İki modelin de aynı sayıda deney skoruna sahip olması gerekir!")
        
    if len(dl_scores) < 5:
        warnings.warn("[BİLGİ] Örneklem sayısı (deney sayısı) çok az. P-Value güvenilir olmayabilir.")
        
    # Her iki dizi de birebir aynıysa Wilcoxon zero-diff hatası verir, bunu yakalayalım:
    diff = np.array(dl_scores) - np.array(automata_scores)
    if np.all(diff == 0):
        print("\n[İSTATİSTİK] İki modelin tüm skorları tamamen aynı. İstatistiksel fark yoktur.")
        return 1.0
        
    stat, p_value = wilcoxon(dl_scores, automata_scores)
    
    print("\n==================================================")
    print("      İSTATİSTİKSEL ANLAMLILIK (WILCOXON) TESTİ   ")
    print("==================================================")
    print(f"Deep Learning (Ortalama Başarı): {np.mean(dl_scores):.4f}")
    print(f"Otomata Modeli (Ortalama Başarı): {np.mean(automata_scores):.4f}")
    print(f"P-Value Değeri: {p_value:.6f}")
    print("--------------------------------------------------")
    
    if p_value < 0.05:
        print("-> KARAR: p < 0.05")
        print("-> İki modelin başarı oranları arasındaki fark İSTATİSTİKSEL OLARAK ANLAMLIDIR.")
        if np.mean(dl_scores) > np.mean(automata_scores):
            print("-> Sonuç: Deep Learning modeli Otomata modeline kıyasla tutarlı bir şekilde daha başarılıdır.")
        else:
            print("-> Sonuç: Otomata modeli Deep Learning modeline kıyasla tutarlı bir şekilde daha başarılıdır.")
    else:
        print("-> KARAR: p >= 0.05")
        print("-> İki modelin başarı oranları arasındaki fark ŞANSA BAĞLIDIR.")
        print("-> Sonuç: İstatistiksel olarak modellerden biri diğerinden 'kesinlikle daha iyidir' denilemez.")
        
    return p_value
