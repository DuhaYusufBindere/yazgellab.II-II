import pandas as pd
from sklearn.decomposition import PCA
from src.config.config_loader import load_config
from main import prepare_batadal_data, prepare_skab_data_fold, run_single_experiment
from src.experiments.baseline import run_baseline_comparison

def run_cross_only():
    config = load_config()
    
    print("[INFO] Loading BATADAL and SKAB data...")
    X_train_b, y_train_b, X_val_b, y_val_b, X_test_b, y_test_b = prepare_batadal_data(config)
    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s = prepare_skab_data_fold(config, fold_idx=0)
    
    print("[INFO] Applying PCA (1D) to solve shape mismatches for Cross-Dataset training...")
    # BATADAL için PCA
    pca_b = PCA(n_components=1)
    X_train_b_1d = pd.DataFrame(pca_b.fit_transform(X_train_b))
    X_val_b_1d = pd.DataFrame(pca_b.transform(X_val_b))
    X_test_b_1d = pd.DataFrame(pca_b.transform(X_test_b))
    
    # SKAB için PCA
    pca_s = PCA(n_components=1)
    X_train_s_1d = pd.DataFrame(pca_s.fit_transform(X_train_s))
    X_val_s_1d = pd.DataFrame(pca_s.transform(X_val_s))
    X_test_s_1d = pd.DataFrame(pca_s.transform(X_test_s))
    
    print("\n[AŞAMA 4] Cross-Dataset (Çapraz Veri Seti) Genellenebilirlik Testleri")
    
    cross_b_to_s_kwargs = {
        "X_train": X_train_b_1d, "y_train": y_train_b, 
        "X_val": X_val_b_1d, "y_val": y_val_b, 
        "X_test": X_test_s_1d, "y_test": y_test_s,
        "dataset_name": "CROSS_Train-BATADAL_Test-SKAB"
    }
    print("\n>>> Çapraz Test: Train(BATADAL) -> Test(SKAB)")
    cross_1 = run_baseline_comparison(run_single_experiment, scenario="normal", **cross_b_to_s_kwargs)
    
    cross_s_to_b_kwargs = {
        "X_train": X_train_s_1d, "y_train": y_train_s, 
        "X_val": X_val_s_1d, "y_val": y_val_s, 
        "X_test": X_test_b_1d, "y_test": y_test_b,
        "dataset_name": "CROSS_Train-SKAB_Test-BATADAL"
    }
    print("\n>>> Çapraz Test: Train(SKAB) -> Test(BATADAL)")
    cross_2 = run_baseline_comparison(run_single_experiment, scenario="normal", **cross_s_to_b_kwargs)
    
    print("[SUCCESS] Cross-Dataset testleri tamamlandı ve sonuçlar eklendi!")

if __name__ == "__main__":
    run_cross_only()
