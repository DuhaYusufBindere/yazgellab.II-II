from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred):
    """
    Modelin tahminleri üzerinden doğruluk ve diğer hata metriklerini hesaplar.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average='binary', zero_division=0)
    }
    return metrics
