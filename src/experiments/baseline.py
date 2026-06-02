from utils.seed import set_seed


def run_baseline_comparison(experiment_fn, **kwargs):
    """
    Tüm modelleri (DL + Otomata) sabit parametrelerle (window_size=4, alphabet_size=3)
    çalıştırarak elma-elma karşılaştırması yapar. Grid search öncesi referans (baseline) noktasıdır.
    """
    seed = 42
    window_size = 4
    alphabet_size = 3
    
    set_seed(seed)
    print(f"[BASELINE] Deney başlatılıyor... Seed={seed}, window_size={window_size}, alphabet_size={alphabet_size}")
    
    result = experiment_fn(
        seed=seed,
        window_size=window_size,
        alphabet_size=alphabet_size,
        **kwargs
    )
    
    result["seed"] = seed
    result["window_size"] = window_size
    result["alphabet_size"] = alphabet_size
    result["is_baseline"] = True
    
    print("[BASELINE] Deney tamamlandı.")
    
    return result
