import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Modelin tahminleri üzerinden Karmaşıklık Matrisini (Confusion Matrix) hesaplar
    ve Seaborn/Matplotlib kullanarak görselleştirir.
    
    Parametreler:
        y_true (list/np.array): Gerçek etiketler
        y_pred (list/np.array): Modelin tahmin ettiği etiketler
        title (str): Grafiğin başlığı
        save_path (str): Grafiğin kaydedileceği dosya yolu. Eğer belirtilirse dosya olarak kaydeder.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    # Confusion matrix ısı haritası çizimi (annot=True ile renklendirilmiş)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal (0)', 'Anomali (1)'], 
                yticklabels=['Normal (0)', 'Anomali (1)'],
                annot_kws={"size": 14})
    
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=12)
    plt.ylabel('Gerçek Sınıf', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        # Kaydedilecek klasörün var olduğundan emin olalım
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] Confusion Matrix '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
        
    plt.close()
