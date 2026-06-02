def calculate_path_probability(test_patterns, transition_probs):
    """                   
    Returns:
        float: Toplam yol olasılığı.
    """
    if not test_patterns or len(test_patterns) < 2:
        return 1.0
        
    total_prob = 1.0
    
    for i in range(len(test_patterns) - 1):
        curr_state = test_patterns[i]
        next_state = test_patterns[i + 1]
        
        # Eğer sözlükte geçiş varsa olasılığını al, yoksa 0.0 olarak kabul et
        prob = transition_probs.get(curr_state, {}).get(next_state, 0.0)
        total_prob *= prob
        
    return total_prob
