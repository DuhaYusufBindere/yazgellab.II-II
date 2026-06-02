import math

def calculate_path_probability(test_patterns, transition_probs, use_log=True, epsilon=1e-8):
    """                
    Returns:
        float: Toplam yol olasılığı (use_log=True ise log olasılık değeri).
    """
    if not test_patterns or len(test_patterns) < 2:
        return 0.0 if use_log else 1.0
        
    total_prob = 0.0 if use_log else 1.0
    
    for i in range(len(test_patterns) - 1):
        curr_state = test_patterns[i]
        next_state = test_patterns[i + 1]
        
        # Eğer sözlükte geçiş varsa olasılığını al, yoksa 0.0 olarak kabul et
        prob = transition_probs.get(curr_state, {}).get(next_state, 0.0)
        
        # Smoothing (0 olasılıkları engelleme)
        if prob <= 0.0:
            prob = epsilon
            
        # Underflow Koruması (Log Olasılık)
        if use_log:
            total_prob += math.log(prob)
        else:
            total_prob *= prob
        
    return total_prob
