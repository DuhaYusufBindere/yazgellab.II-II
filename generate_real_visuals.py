import os
import pandas as pd
from src.config.config_loader import load_config
from src.data.batadal_loader import load_batadal_training, ensure_temporal_order, drop_time_columns, check_missing, separate_target
from src.data.scaler import get_scaler, fit_scaler_on_train, transform_with_scaler
from sklearn.decomposition import PCA
from src.models.paa import paa_transform
from src.models.sax import sax_transform
from src.models.automata import extract_patterns, count_transitions, compute_probabilities
from src.utils.visualization import plot_transition_heatmap, plot_automata_state_diagram, plot_parameter_sensitivity

def generate_real_visuals():
    print("[INFO] Gerçek veriler yükleniyor...")
    config = load_config()
    
    # 1. Veri Hazırlığı
    df = load_batadal_training(config)
    df = ensure_temporal_order(df)
    df = drop_time_columns(df)
    df = check_missing(df)
    X, y = separate_target(df)
    
    scaler = get_scaler(config)
    scaler = fit_scaler_on_train(scaler, X)
    X_scaled = transform_with_scaler(scaler, X)
    
    # PCA ile 1 Boyuta İndirgeme
    pca = PCA(n_components=1)
    X_1d = pca.fit_transform(X_scaled).flatten()
    
    os.makedirs('results/graphs', exist_ok=True)
    
    # ---------------------------------------------------------
    # A) Isı Haritası ve Durum Diyagramı (Heatmap & Network)
    # (Optimum parametre olarak Window=4, Alphabet=3 seçelim)
    # ---------------------------------------------------------
    print("[INFO] Gerçek Geçiş Matrisi (Heatmap & Network) oluşturuluyor...")
    opt_window = 4
    opt_alphabet = 3
    
    paa_vals = paa_transform(X_1d, opt_window)
    sax_vals = sax_transform(paa_vals, opt_alphabet)
    patterns = extract_patterns(sax_vals, opt_window)
    counts = count_transitions(patterns)
    transition_probs = compute_probabilities(counts)
    
    plot_transition_heatmap(transition_probs, title="Gerçek Geçiş Isı Haritası (BATADAL)", save_path='results/graphs/real_heatmap.png')
    plot_automata_state_diagram(transition_probs, title="Gerçek Durum Diyagramı (BATADAL)", save_path='results/graphs/real_network.png')
    
    # ---------------------------------------------------------
    # B) Parametre Duyarlılığı (Window Size vs State Count)
    # ---------------------------------------------------------
    print("[INFO] Parametre Duyarlılık Analizi yapılıyor...")
    window_sizes = [3, 4, 5, 6, 7, 8]
    state_counts = []
    
    for w in window_sizes:
        paa_v = paa_transform(X_1d, w)
        sax_v = sax_transform(paa_v, opt_alphabet)
        pats = extract_patterns(sax_v, w)
        c = count_transitions(pats)
        probs = compute_probabilities(c)
        # Unique state sayısı
        num_states = len(probs.keys())
        state_counts.append(num_states)
        
    plot_parameter_sensitivity(window_sizes, state_counts, title="Window Size vs Benzersiz Durum Sayısı", save_path='results/graphs/real_sensitivity.png')
    
    print("[SUCCESS] Tüm gerçek grafikler başarıyla oluşturuldu ve 'results/graphs/' klasörüne kaydedildi!")

if __name__ == "__main__":
    generate_real_visuals()
