from src.explain.path_probability import calculate_path_probability
from src.explain.decision import make_decision

def analyze_counterfactuals(test_patterns, alternatives, transition_probs, threshold, use_log=True, epsilon=1e-8):
    """
    Karşıt Durum Analizi (Counterfactual Analysis):
    Gerçek pattern yerine en yakın/alternatif pattern'ler gelmiş olsaydı 
    modelin path probability ve kararının nasıl değişeceğini analiz eder.
    
    Args:
        test_patterns (list): Mevcut (gerçekleşen) pattern dizisi.
        alternatives (list): Son pattern yerine denenecek 2-3 alternatif pattern listesi.
        transition_probs (dict): Eğitimden elde edilen geçiş olasılıkları matrisi.
        threshold (float): Anomali sınırını belirleyen eşik değer.
        use_log (bool): Log olasılık kullanımı (underflow koruması).
        epsilon (float): Sıfır olasılıklar için smoothing değeri.
        
    Returns:
        list: Alternatif analiz sonuçlarını içeren sözlüklerin listesi.
    """
    results = []
    
    if not test_patterns:
        return results
        
    for alt_pattern in alternatives:
        # Son gerçekleşen pattern'i alternatif olanla değiştiriyoruz
        alt_sequence = test_patterns[:-1] + [alt_pattern]
        
        # Yeni dizi için olasılık hesaplama
        alt_prob = calculate_path_probability(
            alt_sequence, 
            transition_probs, 
            use_log=use_log, 
            epsilon=epsilon
        )
        
        # Yeni dizi için karar mekanizması
        alt_decision = make_decision(alt_prob, threshold)
        
        results.append({
            "alternative_pattern": alt_pattern,
            "path_probability": alt_prob,
            "decision": alt_decision
        })
        
    return results
