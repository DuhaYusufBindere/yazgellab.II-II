import os
import pandas as pd

def log_experiment_result(result_dict, filepath="results/experiment_results.csv"):
    """
    Deney sonuçlarını (metrics, params) CSV dosyasına kaydeder.
    Eğer dosya yoksa sütun başlıklarıyla yeni oluşturur.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    df = pd.DataFrame([result_dict])
    
    if not os.path.isfile(filepath):
        df.to_csv(filepath, index=False, mode='w', encoding='utf-8-sig')
    else:
        df.to_csv(filepath, index=False, mode='a', header=False, encoding='utf-8-sig')
    
    print(f"[LOGGER] Sonuç başarıyla kaydedildi: {filepath}")
