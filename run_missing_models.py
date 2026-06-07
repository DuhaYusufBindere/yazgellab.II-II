import pandas as pd
from sklearn.decomposition import PCA
from src.config.config_loader import load_config
from main import prepare_batadal_data, prepare_skab_data_fold
from src.models.train import run_dl_pipeline
from utils.metrics import calculate_metrics
from utils.logger import log_experiment_result

def run_missing_dl_baseline(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name, scenario="normal", model_names=["CNN", "GRU"], is_1d=False):
    config = load_config()
    seeds = config.get("experiment", {}).get("random_seeds", [42, 123, 2026, 7, 999])
    
    input_size = 1 if is_1d else X_train.shape[1]
    
    for seed in seeds:
        for model_name in model_names:
            print(f"[{dataset_name}] Missing DL Baseline Running: Model={model_name}, Seed={seed}, Scenario={scenario}")
            
            # DL pipeline
            dl_results = run_dl_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, model_name=model_name, input_size=input_size, config=config)
            dl_metrics = calculate_metrics(dl_results["targets"], dl_results["predictions"])
            
            log = {
                "dataset": dataset_name,
                "model_name": model_name,
                "scenario": scenario,
                "seed": seed,
                "window_size": config.get("experiment", {}).get("window_size_default", 4),
                "alphabet_size": config.get("experiment", {}).get("alphabet_size_default", 3),
                "status": "Success",
                "accuracy": dl_metrics["accuracy"],
                "f1_score": dl_metrics["f1_score"],
                "precision": dl_metrics["precision"],
                "recall": dl_metrics["recall"]
            }
            log_experiment_result(log)
            print(f"[SUCCESS] {model_name} logged for {dataset_name}.")

def run_missing_models():
    config = load_config()
    
    print("[INFO] Loading data...")
    X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b = prepare_batadal_data(config)
    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = prepare_skab_data_fold(config, fold_idx=0)
    
    print("[INFO] Running BATADAL Baselines...")
    run_missing_dl_baseline(X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b, "BATADAL", "normal")
    run_missing_dl_baseline(X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b, "BATADAL", "noisy")
    run_missing_dl_baseline(X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b, "BATADAL", "unseen")
    
    print("[INFO] Running SKAB Baselines...")
    run_missing_dl_baseline(X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, "SKAB", "normal")
    
    print("[INFO] Running Cross-Dataset Baselines (1D PCA)...")
    pca_b = PCA(n_components=1)
    X_train_b_1d = pd.DataFrame(pca_b.fit_transform(X_train_b))
    X_val_b_1d = pd.DataFrame(pca_b.transform(X_val_b))
    X_test_b_1d = pd.DataFrame(pca_b.transform(X_test_b))
    
    pca_s = PCA(n_components=1)
    X_train_s_1d = pd.DataFrame(pca_s.fit_transform(X_train_s))
    X_val_s_1d = pd.DataFrame(pca_s.transform(X_val_s))
    X_test_s_1d = pd.DataFrame(pca_s.transform(X_test_s))
    
    run_missing_dl_baseline(X_train_b_1d, y_train_b, X_val_b_1d, y_val_b, X_test_s_1d, y_test_s, "CROSS_Train-BATADAL_Test-SKAB", "normal", is_1d=True)
    run_missing_dl_baseline(X_train_s_1d, y_train_s, X_val_s_1d, y_val_s, X_test_b_1d, y_test_b, "CROSS_Train-SKAB_Test-BATADAL", "normal", is_1d=True)
    
    print("[INFO] All missing models trained and logged!")

if __name__ == "__main__":
    import torch
    # GPU bellek şişmesini önlemek için seedleri sabitleyelim
    torch.manual_seed(42)
    run_missing_models()
