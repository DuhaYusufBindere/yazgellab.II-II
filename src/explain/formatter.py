import json

def generate_explanation_json(time_step, state, pattern, unseen, mapped_to, probability, decision, confidence_score=None, counterfactual_analysis=None):
    """
    Açıklanabilirlik adımlarından çıkan sonuçları standart bir JSON formatında döndürür.
    
    Args:
        time_step (int): Zaman adımı (index).
        state (str): Sistemdeki anlık durum (örn. "State-4").
        pattern (str): O anki sembolik örüntü (örn. "abc").
        unseen (bool): Örüntünün eğitim setinde bilinmeyen (unseen) olup olmadığı.
        mapped_to (str): Unseen ise Levenshtein ile eşleşen en yakın örüntü (değilse None).
        probability (float): Yol olasılığı (path probability).
        decision (str): Karar mekanizmasının çıktısı ("Normal" veya "Anomali...").
        confidence_score (str, optional): Güven skoru (Madde 10.4).
        counterfactual_analysis (list, optional): Karşıt durum analizi (Madde 10.6).
        
    Returns:
        str: JSON formatında, okunabilir (indented) çıktı.
    """
    output_dict = {
        "time_step": time_step,
        "state": state,
        "pattern": pattern,
        "unseen": unseen,
        "mapped_to": mapped_to,
        "probability": probability,
        "decision": decision
    }
    
    if confidence_score is not None:
        output_dict["confidence_score"] = confidence_score
        
    if counterfactual_analysis is not None:
        output_dict["counterfactual_analysis"] = counterfactual_analysis
        
    return json.dumps(output_dict, indent=4, ensure_ascii=False)
