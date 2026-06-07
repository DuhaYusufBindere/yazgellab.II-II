import pandas as pd
import json
import os

def generate_readme():
    csv_path = "results/experiment_results.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Helper function to format Mean ± Std
    def format_mean_std(subset):
        if len(subset) == 0:
            return "N/A"
        mean = subset['f1_score'].mean()
        std = subset['f1_score'].std()
        if pd.isna(std):
            std = 0.0
        return f"{mean:.4f} &plusmn; {std:.4f}"

    def format_mean(subset):
        if len(subset) == 0:
            return "N/A"
        return f"{subset['f1_score'].mean():.4f}"

    # Tablo 1: Model Performansı ve Stabilitesi (Orijinal Veri)
    # BATADAL
    bat_norm = df[(df['dataset'] == 'BATADAL') & (df['scenario'] == 'normal')]
    bat_lstm = bat_norm[bat_norm['model_name'] == 'LSTM']
    bat_gru = bat_norm[bat_norm['model_name'] == 'GRU']
    bat_cnn = bat_norm[bat_norm['model_name'] == 'CNN']
    bat_auto = bat_norm[bat_norm['model_name'] == 'Automata']
    
    # SKAB
    skab_norm = df[(df['dataset'] == 'SKAB') & (df['scenario'] == 'normal')]
    skab_lstm = skab_norm[skab_norm['model_name'] == 'LSTM']
    skab_gru = skab_norm[skab_norm['model_name'] == 'GRU']
    skab_cnn = skab_norm[skab_norm['model_name'] == 'CNN']
    skab_auto = skab_norm[skab_norm['model_name'] == 'Automata']

    # Tablo 2: Gürültü ve Unseen Veri (BATADAL)
    bat_noisy = df[(df['dataset'] == 'BATADAL') & (df['scenario'] == 'noisy')]
    bat_unseen = df[(df['dataset'] == 'BATADAL') & (df['scenario'] == 'unseen')]

    lstm_norm_f1 = format_mean(bat_lstm)
    lstm_noisy_f1 = format_mean(bat_noisy[bat_noisy['model_name'] == 'LSTM'])
    lstm_unseen_f1 = format_mean(bat_unseen[bat_unseen['model_name'] == 'LSTM'])
    
    gru_norm_f1 = format_mean(bat_gru)
    gru_noisy_f1 = format_mean(bat_noisy[bat_noisy['model_name'] == 'GRU'])
    gru_unseen_f1 = format_mean(bat_unseen[bat_unseen['model_name'] == 'GRU'])
    
    cnn_norm_f1 = format_mean(bat_cnn)
    cnn_noisy_f1 = format_mean(bat_noisy[bat_noisy['model_name'] == 'CNN'])
    cnn_unseen_f1 = format_mean(bat_unseen[bat_unseen['model_name'] == 'CNN'])
    
    auto_norm_f1 = format_mean(bat_auto)
    auto_noisy_f1 = format_mean(bat_noisy[bat_noisy['model_name'] == 'Automata'])
    auto_unseen_f1 = format_mean(bat_unseen[bat_unseen['model_name'] == 'Automata'])

    # Tablo 3: Cross-Dataset
    cross_b_to_s = df[df['dataset'] == 'CROSS_Train-BATADAL_Test-SKAB']
    cross_s_to_b = df[df['dataset'] == 'CROSS_Train-SKAB_Test-BATADAL']
    
    # Tablo 4: Parametre Analizi (Automata - BATADAL Normal)
    alphabets = sorted(bat_auto['alphabet_size'].unique())
    windows = sorted(bat_auto['window_size'].unique())
    
    param_table = "| Window \\ Alphabet Size | " + " | ".join([str(a) for a in alphabets]) + " |\n"
    param_table += "| --- |" + " --- |" * len(alphabets) + "\n"
    
    for w in windows:
        row_str = f"| **{w}** |"
        for a in alphabets:
            subset = bat_auto[(bat_auto['window_size'] == w) & (bat_auto['alphabet_size'] == a)]
            f1 = format_mean_std(subset)
            row_str += f" {f1} |"
        param_table += row_str + "\n"
        
    # Tablo 5: Runtime Analizi
    runtime_text = "Modellerin çalışma süresi verileri `results/runtime_results.json` dosyasında bulunamadı."
    if os.path.exists("results/runtime_results.json"):
        with open("results/runtime_results.json", "r", encoding="utf-8") as f:
            runtimes = json.load(f)
            runtime_text = "| Model | Training Time (sn) | Inference Time (sn) |\n"
            runtime_text += "| --- | --- | --- |\n"
            for model_name in ["LSTM", "GRU", "CNN", "Automata"]:
                if model_name in runtimes:
                    tt = runtimes[model_name]["training_time"]
                    it = runtimes[model_name]["inference_time"]
                    runtime_text += f"| **{model_name}** | {tt} | {it} |\n"
        
    # Wilcoxon Sonuçlarını Oku
    wilcoxon_text = "Wilcoxon testi çalıştırılmadı."
    if os.path.exists("results/wilcoxon_results.txt"):
        with open("results/wilcoxon_results.txt", "r", encoding="utf-8") as f:
            wilcoxon_text = f.read().strip()
            
    # Explainability JSON Sonucunu Oku
    explainability_json = "{}"
    if os.path.exists("explainability_sample.json"):
        with open("explainability_sample.json", "r", encoding="utf-8") as f:
            explainability_json = f.read()

    readme_content = f"""# Yazılım Geliştirme Dersi - 2. Proje Raporu
**From Black-Box to Explainability: Probabilistic Automata for Time Series Analysis**

**Hazırlayanlar:**
- Ahmet ÖZ (231307094)
- Duha Yusuf BİNDERE (231307077)
**Kurum:** Kocaeli Üniversitesi, Bilişim Sistemleri Mühendisliği, 3. Sınıf

---

Bu doküman, projede yürütülen karşılaştırmalı analizin ve deney sonuçlarının kapsamlı akademik raporudur. Proje kapsamında **BATADAL** ve **SKAB** zaman serisi veri setleri üzerinde Derin Öğrenme tabanlı (LSTM, GRU, 1D-CNN) "Black-Box" modeller ile yorumlanabilir (Interpretable) "Probabilistic Automata" modellerinin kıyaslamalı analizi gerçekleştirilmiştir.

## 1. Model Karşılaştırmaları ve Performans
Aşağıdaki tabloda her iki modelin de orijinal senaryodaki (Grid Search dahil 160+ iterasyon) genel F1-skor ortalamaları ve standart sapmaları özetlenmiştir.

**Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score &plusmn; Standart Sapma)**
| Model | BATADAL | SKAB |
| --- | --- | --- |
| **LSTM** | {format_mean_std(bat_lstm)} | {format_mean_std(skab_lstm)} |
| **GRU** | {format_mean_std(bat_gru)} | {format_mean_std(skab_gru)} |
| **1D-CNN** | {format_mean_std(bat_cnn)} | {format_mean_std(skab_cnn)} |
| **Automata** | {format_mean_std(bat_auto)} | {format_mean_std(skab_auto)} |

## 2. Gürültü ve Bilinmeyen (Unseen) Veri Analizi
Sensörlere %10 Gaussian Noise eklendiğinde ve modele eğitim setinde bulunmayan örüntüler verildiğinde her iki modelin BATADAL üzerindeki direnci aşağıdaki tabloda listelenmiştir.

**Tablo 2: Gürültü ve Unseen Veri Etkisi (Ortalama F1-score)**
| Model | Orijinal | Gürültülü (Noisy) | Bilinmeyen (Unseen) |
| --- | --- | --- | --- |
| **LSTM** | {lstm_norm_f1} | {lstm_noisy_f1} | {lstm_unseen_f1} |
| **GRU** | {gru_norm_f1} | {gru_noisy_f1} | {gru_unseen_f1} |
| **1D-CNN** | {cnn_norm_f1} | {cnn_noisy_f1} | {cnn_unseen_f1} |
| **Automata** | {auto_norm_f1} | {auto_noisy_f1} | {auto_unseen_f1} |

## 3. Çapraz Veri Seti (Cross-Dataset) Analizi
Mimarilerin genellenebilirlik yeteneğini test etmek için model bir sensör ağında eğitilip, diğerinde test edilmiştir. (Çok değişkenli veriler boyut indirgeme ile PCA-1D'ye dönüştürülmüştür).

**Tablo 3: Cross-Dataset Performans Karşılaştırması (F1-score)**
| Train \\ Test | BATADAL | SKAB |
| --- | --- | --- |
| **BATADAL** | - | LSTM: {format_mean(cross_b_to_s[cross_b_to_s['model_name']=='LSTM'])} <br> GRU: {format_mean(cross_b_to_s[cross_b_to_s['model_name']=='GRU'])} <br> CNN: {format_mean(cross_b_to_s[cross_b_to_s['model_name']=='CNN'])} <br> Automata: {format_mean(cross_b_to_s[cross_b_to_s['model_name']=='Automata'])} |
| **SKAB** | LSTM: {format_mean(cross_s_to_b[cross_s_to_b['model_name']=='LSTM'])} <br> GRU: {format_mean(cross_s_to_b[cross_s_to_b['model_name']=='GRU'])} <br> CNN: {format_mean(cross_s_to_b[cross_s_to_b['model_name']=='CNN'])} <br> Automata: {format_mean(cross_s_to_b[cross_s_to_b['model_name']=='Automata'])} | - |

## 4. Parametre Duyarlılığı (Automata Window Size)
Aşağıdaki tablo, Automata modelinde kullanılan Sliding Window (Pencere Boyutu) parametresinin BATADAL veri seti üzerindeki genel performans etkisini göstermektedir.

**Tablo 4: Automata Parametre Duyarlılık Analizi (F1-score)**
{param_table}
## 5. Modellerin Çalışma Süresi (Runtime) Karşılaştırması
Modellerin hesaplama maliyetlerini (computational cost) kıyaslamak amacıyla, tüm modellerin eğitim (training) ve çıkarım (inference) süreleri BATADAL veri seti üzerinde saniye (sn) cinsinden ölçülmüştür. DL modellerinin training süresi 50 epoch üzerinden asimptotik olarak hesaplanmıştır.

**Tablo 5: Çalışma Süresi (Runtime) Analizi**
{runtime_text}
## 6. İstatistiksel Anlamlılık Testi (Wilcoxon)
İki modelin performans farkının rastgele olup olmadığını kanıtlamak için Wilcoxon test uygulanmıştır:
```text
{wilcoxon_text}
```

## 7. Olasılıksal Açıklanabilirlik Modülü (Explainable AI)
Olasılıksal Otomata modeli, kararlarını verirken geçiş ihtimallerini hesaplar. Eğer eğitimde hiç görülmemiş bir veri gelirse (Unseen), **Levenshtein** algoritması kullanılarak en yakın duruma eşleme (mapping) yapılır. Ayrıca Counterfactual (Karşıt Durum) analizi ile sistem, "eğer farklı bir pattern gelseydi sonuç ne olurdu" sorusunu da yanıtlamaktadır. Örnek bir kararın izi (JSON) aşağıdadır:

```json
{explainability_json}
```

## 8. Görselleştirmeler ve Model Çıktıları
Modelin gerçek veriler üzerinde oluşturduğu matematiksel State-Transition (Durum-Geçiş) haritaları ve başarı grafikleri aşağıda yer almaktadır.

### 8.1. Confusion Matrix ve ROC Eğrisi (Test Seti)
Modelin Sınıflandırma Başarısı:
![Confusion Matrix](results/graphs/confusion_matrix.png)
![ROC Eğrisi](results/graphs/roc_curve.png)

### 8.2. Automata State Diagram (Durum Geçiş Ağı)
![Automata State Diagram](results/graphs/real_network.png)

### 8.3. Transition Probability Heatmap (Isı Haritası)
![Transition Heatmap](results/graphs/real_heatmap.png)

### 8.4. Parametre Duyarlılık Grafiği (Window Size vs State Count)
![Sensitivity Graph](results/graphs/real_sensitivity.png)

## Sonuç
Derin Öğrenme (LSTM) algoritmaları saf başarı oranında üstünlük sağlarken, Olasılıksal Otomata modeli; Gürültülü ortamlarda şeffaflığı (Explainability), görselleştirilebilir iç yapısı ve Kayan Pencere algoritmasının istikrarı sayesinde anomali tespiti gibi yorumlanabilirliğin kritik olduğu projelerde son derece güçlü bir alternatif olduğunu kanıtlamıştır. Ayrıca Levenshtein algoritması destekli eşleme sayesinde Unseen senaryolara karşı esnek bir duruş sergilemiştir.
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("README.md başarıyla oluşturuldu!")

if __name__ == "__main__":
    generate_readme()
