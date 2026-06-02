import os
import json
from src.config.config_loader import load_config
from src.experiments.grid_search import run_grid_search
from src.experiments.baseline import run_baseline_comparison
from src.data.noise_injector import add_gaussian_noise

def run_single_experiment(seed, window_size=4, alphabet_size=3, scenario="normal"):
    """
    DL ve Otomata pipeline'ını tek bir kombinasyon için çalıştıran ana deney fonksiyonudur.
    """
    config = load_config()
    
    # NOT: Derin öğrenme eğitim döngüleri ve veri setleri buraya entegre edilecek.
    
    # 1. Veri Yükleme (Örnek)
    # X_train, y_train, X_test, y_test = load_data(...)
    
    # 2. Senaryo Yönetimi
    if scenario == "noisy":
        # X_test = add_gaussian_noise(X_test, mean=0.0, std_dev=0.1, seed=seed)
        pass
    elif scenario == "unseen":
        # Test verisine yapay bir "unseen" (bilinmeyen) anomali paterni enjekte edilebilir.
        pass
        
    # 3. DL Modeli (Eğitim ve Test)
    # dl_results = run_dl_pipeline(...)
    
    # 4. Otomata Modeli (PAA, SAX, Transitions, Explainability)
    # automata_results = run_automata_pipeline(...)
    
    # Simüle edilmiş sonuç (Faz 12'de gerçek metrikler hesaplanacak)
    result = {
        "scenario": scenario,
        "seed": seed,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "status": "Success",
        "mock_accuracy": 0.95 if scenario == "normal" else 0.85
    }
    return result

def main():
    print("==================================================")
    print("      YAZLAB II - PROJE ANA ÇALIŞTIRMA MODÜLÜ     ")
    print("==================================================")
    
    # 1. Sabit Parametre Karşılaştırması (Baseline) - Sadece Normal Senaryo
    print("\n[AŞAMA 1] Sabit Parametre Karşılaştırması (Baseline)")
    baseline_results = run_baseline_comparison(run_single_experiment, scenario="normal")
    print(f"Baseline Tamamlandı: {json.dumps(baseline_results, indent=2)}")
    
    # 2. Grid Search - Normal Senaryo
    print("\n[AŞAMA 2] Grid Search Tarama: Orijinal Veri Senaryosu")
    normal_results = run_grid_search(run_single_experiment, scenario="normal")
    
    # 3. Grid Search - Gürültülü Senaryo
    print("\n[AŞAMA 3] Grid Search Tarama: Gürültülü (Noisy) Veri Senaryosu")
    noisy_results = run_grid_search(run_single_experiment, scenario="noisy")
    
    # 4. Grid Search - Unseen Pattern Senaryosu
    print("\n[AŞAMA 4] Grid Search Tarama: Bilinmeyen (Unseen) Pattern Senaryosu")
    unseen_results = run_grid_search(run_single_experiment, scenario="unseen")
    
    print("\n[INFO] Tüm deney senaryoları (Normal, Noisy, Unseen) başarıyla tamamlandı.")
    print("[INFO] Model eğitim ve değerlendirme logları oluşturuluyor...")

if __name__ == "__main__":
    main()
