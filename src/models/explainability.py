import json

def generate_explanation(time_step, previous_state, incoming_pattern, is_unseen, mapped_to, distance, transitions, path_probability, decision, confidence_score, counterfactual, save_path="explainability_sample.json"):
    """
    Model kararlarının olasılıksal gerekçelerini JSON formatında üreten modül.
    """
    
    explanation = {
        "time_step": time_step,
        "state": previous_state,
        "pattern": incoming_pattern,
        "status": "unseen" if is_unseen else "known",
        "mapped_to": mapped_to if is_unseen else incoming_pattern,
        "levenshtein_distance": distance if is_unseen else 0,
        "probability": round(path_probability, 4),
        "decision": decision,
        "confidence_score": round(confidence_score, 4),
        "transitions": transitions,
        "counterfactual_analysis": counterfactual
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=4)
        
    return explanation

def run_counterfactual_analysis(previous_state, incoming_pattern, transition_probs):
    """
    Karşıt Durum (Counterfactual) Analizi: Alternatif örüntüler gelseydi modelin tahmini ne olurdu?
    """
    analysis_results = []
    
    if previous_state in transition_probs:
        possible_next = transition_probs[previous_state]
        for alt_pattern, prob in possible_next.items():
            if alt_pattern != incoming_pattern:
                status = "anomaly" if prob < 0.01 else "normal"
                analysis_results.append(f"Eğer '{incoming_pattern}' yerine '{alt_pattern}' gelseydi, olasılık {prob:.4f} olacak ve sonuç '{status}' çıkacaktı.")
                
    if not analysis_results:
        return "Alternatif geçiş bulunamadı."
    
    return "\n".join(analysis_results[:3])
