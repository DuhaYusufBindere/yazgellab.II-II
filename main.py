import os
import json
import pandas as pd
from src.config.config_loader import load_config
from src.experiments.grid_search import run_grid_search
from src.experiments.baseline import run_baseline_comparison
from src.data.noise_injector import add_gaussian_noise
from src.data.batadal_loader import load_batadal_training, drop_time_columns, separate_target, ensure_temporal_order, check_missing
from src.data.splitter import batadal_train_split, batadal_val_split, batadal_test_split
from src.data.scaler import get_scaler, fit_scaler_on_train, transform_with_scaler
from src.models.train import run_dl_pipeline
from src.models.automata_pipeline import run_automata_pipeline
from utils.metrics import calculate_metrics
from utils.logger import log_experiment_result
def prepare_batadal_data(config):
    df = load_batadal_training(config)
    df = ensure_temporal_order(df)
    df = drop_time_columns(df)
    df = check_missing(df)
    X, y = separate_target(df)
    
    X_train, y_train = batadal_train_split(X, y, config)
    X_val, y_val = batadal_val_split(X, y, config)
    X_test, y_test = batadal_test_split(X, y, config)
    
    scaler = get_scaler(config)
    scaler = fit_scaler_on_train(scaler, X_train)
    
    X_train = transform_with_scaler(scaler, X_train)
    X_val = transform_with_scaler(scaler, X_val)
    X_test = transform_with_scaler(scaler, X_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_single_experiment(seed, window_size=4, alphabet_size=3, scenario="normal", X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, dataset_name="BATADAL"):
    """
    DL ve Otomata pipeline'ını tek bir kombinasyon için çalıştıran ana deney fonksiyonudur.
    """
    config = load_config()
    
    print(f"\n[{dataset_name}] Deney Başlıyor: Seed={seed}, Window={window_size}, Alphabet={alphabet_size}, Scenario={scenario}")
    
    if scenario == "noisy":
        X_test = add_gaussian_noise(X_test, mean=config.get("experiment", {}).get("noise_mean", 0.0), std_dev=config.get("experiment", {}).get("noise_std", 0.1), seed=seed)
    elif scenario == "unseen":
        # Şimdilik unseen pattern simülasyonu boş geçiliyor
        pass
        
    # 3. DL Modeli (Eğitim ve Test)
    print(f"[AŞAMA] Derin Öğrenme (LSTM) Modeli Çalıştırılıyor...")
    input_size = X_train.shape[1]
    dl_results = run_dl_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, model_name="LSTM", input_size=input_size, config=config)
    
    # 4. Otomata Modeli (Eğitim ve Test)
    print(f"[AŞAMA] Probabilistic Automata Modeli Çalıştırılıyor...")
    automata_results = run_automata_pipeline(X_train, y_train, X_test, y_test, window_size=window_size, alphabet_size=alphabet_size)
    
    # DL Metrikleri
    dl_metrics = calculate_metrics(dl_results["targets"], dl_results["predictions"])
    dl_log = {
        "dataset": dataset_name,
        "model_name": "LSTM",
        "scenario": scenario,
        "seed": seed,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "status": "Success",
        "accuracy": dl_metrics["accuracy"],
        "precision": dl_metrics["precision"],
        "recall": dl_metrics["recall"],
        "f1_score": dl_metrics["f1_score"]
    }
    log_experiment_result(dl_log)
    
    # Automata Metrikleri
    aut_metrics = calculate_metrics(automata_results["targets"], automata_results["predictions"])
    aut_log = {
        "dataset": dataset_name,
        "model_name": "Automata",
        "scenario": scenario,
        "seed": seed,
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "status": "Success",
        "accuracy": aut_metrics["accuracy"],
        "precision": aut_metrics["precision"],
        "recall": aut_metrics["recall"],
        "f1_score": aut_metrics["f1_score"]
    }
    log_experiment_result(aut_log)
    
    return [dl_log, aut_log]

def main():
    print("==================================================")
    print("      YAZLAB II - PROJE ANA ÇALIŞTIRMA MODÜLÜ     ")
    print("==================================================")
    
    config = load_config()
    
    csv_path = "results/experiment_results.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"[INFO] Eski test verisi ({csv_path}) silindi, tamamen yeni eğitim başlıyor!")
        
    # Veriyi bir kez dışarıda yükleyip tüm deneylere parametre olarak geçiyoruz
    print("[INFO] BATADAL Veri seti hazırlanıyor...")
    X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b = prepare_batadal_data(config)
    
    # kwargs to pass to run_single_experiment
    data_kwargs = {
        "X_train": X_train_b, "y_train": y_train_b, 
        "X_val": X_val_b, "y_val": y_val_b, 
        "X_test": X_test_b, "y_test": y_test_b,
        "dataset_name": "BATADAL"
    }
    
    # 1. Sabit Parametre Karşılaştırması (Baseline) - Sadece Normal Senaryo
    print("\n[AŞAMA 1] Sabit Parametre Karşılaştırması (Baseline)")
    baseline_results = run_baseline_comparison(run_single_experiment, scenario="normal", **data_kwargs)
    
    # 2. Grid Search - Normal Senaryo
    print("\n[AŞAMA 2] Grid Search Tarama: Orijinal Veri Senaryosu")
    normal_results = run_grid_search(run_single_experiment, scenario="normal", **data_kwargs)
    
    # 3. Grid Search - Gürültülü Senaryo
    print("\n[AŞAMA 3] Grid Search Tarama: Gürültülü (Noisy) Veri Senaryosu")
    noisy_results = run_grid_search(run_single_experiment, scenario="noisy", **data_kwargs)
    
    # 4. Grid Search - Unseen Pattern Senaryosu
    print("\n[AŞAMA 4] Grid Search Tarama: Bilinmeyen (Unseen) Pattern Senaryosu")
    unseen_results = run_grid_search(run_single_experiment, scenario="unseen", **data_kwargs)
    
    print("\n[INFO] Tüm deney senaryoları başarıyla tamamlandı ve results/experiment_results.csv'ye kaydedildi.")

if __name__ == "__main__":
    main()
