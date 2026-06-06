import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
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

def plot_transition_heatmap(transition_probs, title="Transition Probability Heatmap", save_path=None):
    """
    Otomata modelinin durum geçiş olasılıklarını bir ısı haritası (heatmap) olarak çizer.
    """
    states = list(transition_probs.keys())
    all_states = set(states)
    for s, trans in transition_probs.items():
        all_states.update(trans.keys())
    all_states = sorted(list(all_states))
    
    n = len(all_states)
    matrix = np.zeros((n, n))
    
    for i, s_from in enumerate(all_states):
        if s_from in transition_probs:
            for j, s_to in enumerate(all_states):
                matrix[i, j] = transition_probs[s_from].get(s_to, 0.0)
                
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='YlGnBu', xticklabels=all_states, yticklabels=all_states)
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Gidilen Durum (Next State)', fontsize=12)
    plt.ylabel('Mevcut Durum (Current State)', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] Geçiş Isı Haritası (Heatmap) '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
    plt.close()

def plot_automata_state_diagram(transition_probs, title="Automata State Diagram", save_path=None):
    """
    Otomata modelinin geçişlerini ağ grafiği (network graph) olarak çizer.
    Karmaşıklığı azaltmak için sadece olasılığı 0.1'den büyük geçişleri çizer.
    """
    G = nx.DiGraph()
    
    for s_from, transitions in transition_probs.items():
        for s_to, prob in transitions.items():
            if prob > 0.1:
                G.add_edge(s_from, s_to, weight=prob)
                
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='lightblue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    
    edges = G.edges(data=True)
    weights = [data['weight'] * 4 for _, _, data in edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray', 
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] State Diagram '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
    plt.close()

def plot_parameter_sensitivity(window_sizes, state_counts, title="Window Size vs State Count", save_path=None):
    """
    Otomata'da window_size değiştiğinde state sayısının nasıl değiştiğini çizen parametre duyarlılık grafiği.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(window_sizes, state_counts, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
    
    plt.title(title, pad=15, fontsize=14, fontweight='bold')
    plt.xlabel('Window Size (Pencere Boyutu)', fontsize=12)
    plt.ylabel('Oluşan Toplam Benzersiz Durum (State) Sayısı', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, txt in enumerate(state_counts):
        plt.annotate(str(txt), (window_sizes[i], state_counts[i]), textcoords="offset points", xytext=(0,10), ha='center')
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[BİLGİ] Parametre Duyarlılık Grafiği '{save_path}' konumuna kaydedildi.")
    else:
        plt.show()
    plt.close()
