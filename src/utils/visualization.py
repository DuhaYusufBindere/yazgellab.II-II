import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

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

def plot_roc_curve(y_true, y_prob, title="ROC Curve", save_path=None):
    """
    Modelin tahmin olasılıkları üzerinden ROC eğrisini çizer.
    
    Parametreler:
        y_true (list/np.array): Gerçek etiketler
        y_prob (list/np.array): Modelin 1 (Anomali) sınıfı için tahmin olasılıkları
        title (str): Grafiğin başlığı
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] ROC Eğrisi '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
        
    plt.close()

def plot_precision_recall_curve(y_true, y_prob, title="Precision-Recall Curve", save_path=None):
    """
    Modelin tahmin olasılıkları üzerinden Precision-Recall eğrisini çizer.
    
    Parametreler:
        y_true (list/np.array): Gerçek etiketler
        y_prob (list/np.array): Modelin 1 (Anomali) sınıfı için tahmin olasılıkları
        title (str): Grafiğin başlığı
        save_path (str): Grafiğin kaydedileceği dosya yolu.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] Precision-Recall Eğrisi '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
        
    plt.close()
