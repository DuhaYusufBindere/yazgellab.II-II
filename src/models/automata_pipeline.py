import numpy as np
from sklearn.decomposition import PCA
from src.models.paa import paa_transform
from src.models.sax import sax_transform
from src.models.automata import extract_patterns, count_transitions, compute_probabilities
from src.models.unseen import find_nearest_pattern
from src.models.explainability import generate_explanation, run_counterfactual_analysis

def run_automata_pipeline(X_train, y_train, X_test, y_test, window_size=4, alphabet_size=3, threshold=0.01):
    """
    Otomata modelinin PCA -> PAA -> SAX -> State Machine eğitim ve test sürecini
    tamamen uçtan uca (end-to-end) çalıştıran pipeline modülü.
    """
    print("[Automata] PCA ile 1 boyuta indirgeniyor...")
    pca = PCA(n_components=1)
    
    # X_train ve X_test numpy array formunda geliyor
    X_train_1d = pca.fit_transform(X_train).flatten()
    X_test_1d = pca.transform(X_test).flatten()
    
    print(f"[Automata] PAA ve SAX uygulanıyor... (Window={window_size}, Alphabet={alphabet_size})")
    # PAA
    paa_train = paa_transform(X_train_1d, window_size)
    paa_test = paa_transform(X_test_1d, window_size)
    
    # SAX
    sax_train = sax_transform(paa_train, alphabet_size)
    sax_test = sax_transform(paa_test, alphabet_size)
    
    print("[Automata] Eğitim: Geçiş matrisi (State Machine) oluşturuluyor...")
    # Eğitim aşaması
    # Patern uzunluğu olarak da window_size kullanıyoruz ki ardışık sembolleri birleştirsin
    patterns_train = extract_patterns(sax_train, window_size)
    counts = count_transitions(patterns_train)
    transition_probs = compute_probabilities(counts)
    
    print("[Automata] Çıkarım (Inference): Test verisi üzerinde anomali aranıyor...")
    # Test aşaması
    patterns_test = extract_patterns(sax_test, window_size)
    
    n_original = len(y_test)
    full_preds = np.zeros(n_original)
    
    explanation_generated = False
    
    if len(patterns_test) > 1:
        for i in range(len(patterns_test) - 1):
            curr_state = patterns_test[i]
            next_state = patterns_test[i+1]
            
            is_unseen = False
            mapped_to = next_state
            distance = 0
            
            if curr_state in transition_probs:
                if next_state in transition_probs[curr_state]:
                    prob = transition_probs[curr_state][next_state]
                else:
                    is_unseen = True
                    known_patterns = set(transition_probs[curr_state].keys())
                    if known_patterns:
                        match = find_nearest_pattern(next_state, known_patterns)
                        mapped_to = match["mapped_to"]
                        distance = match["nearest_distance"]
                        prob = transition_probs[curr_state][mapped_to]
                    else:
                        prob = 0.0
            else:
                is_unseen = True
                prob = 0.0
                
            if prob < threshold:
                start_idx = i * window_size
                end_idx = min((i + window_size) * window_size, n_original)
                full_preds[start_idx:end_idx] = 1
                
                if not explanation_generated:
                    confidence = 1.0 - prob
                    counterfactual_text = run_counterfactual_analysis(curr_state, next_state, transition_probs)
                    generate_explanation(
                        time_step=i,
                        previous_state=curr_state,
                        incoming_pattern=next_state,
                        is_unseen=is_unseen,
                        mapped_to=mapped_to,
                        distance=distance,
                        transitions={f"{curr_state} -> {mapped_to}": prob},
                        path_probability=prob,
                        decision="anomaly",
                        confidence_score=confidence,
                        counterfactual=counterfactual_text
                    )
                    explanation_generated = True
                
    # Hedef (target) etiketlerini numpy array'e çevir
    targets = y_test.values if hasattr(y_test, "values") else np.array(y_test)
    
    print("[SUCCESS] Automata modeli eğitimi ve tahmini başarıyla tamamlandı!")
    
    return {
        "transition_probs": transition_probs,
        "predictions": full_preds,
        "targets": targets
    }
