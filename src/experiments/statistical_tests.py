import pandas as pd
from scipy.stats import wilcoxon

def run_wilcoxon_test(csv_path="results/experiment_results.csv"):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Hata: {e}"
        
    bat_norm = df[(df['dataset'] == 'BATADAL') & (df['scenario'] == 'normal')]
    
    lstm_scores = bat_norm[bat_norm['model_name'] == 'LSTM']['f1_score'].values
    auto_scores = bat_norm[bat_norm['model_name'] == 'Automata']['f1_score'].values
    
    if len(lstm_scores) == 0 or len(lstm_scores) != len(auto_scores):
        return "Yeterli veya eşleşen veri yok."
        
    stat, p_value = wilcoxon(lstm_scores, auto_scores)
    
    result = f"Wilcoxon Test Istatistigi: {stat:.4f}, p-degeri: {p_value:.4e}\n"
    if p_value < 0.05:
        result += "Sonuc: p < 0.05. Iki model arasindaki performans farki istatistiksel olarak ANLAMLIDIR."
    else:
        result += "Sonuc: p >= 0.05. Iki model arasindaki performans farki istatistiksel olarak ANLAMLI DEGILDIR."
        
    with open("results/wilcoxon_results.txt", "w", encoding="utf-8") as f:
        f.write(result)
        
    return result

if __name__ == "__main__":
    print(run_wilcoxon_test())
