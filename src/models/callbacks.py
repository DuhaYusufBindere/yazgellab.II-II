import numpy as np
import torch
from src.config.config_loader import load_config


class EarlyStopping:

    def __init__(self, patience=None, verbose=True, delta=0.0, checkpoint_path="checkpoint.pt", config=None):

        if config is None:
            config = load_config()
            
        if patience is None:
            patience = config.get("deep_learning", {}).get("early_stopping_patience", 5)
            
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] İyileşme yok. Sayaç: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("[EarlyStopping] Limit aşıldı. Eğitim sonlandırılıyor.")
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):

        if self.verbose and self.val_loss_min != np.inf:
            print(f"[EarlyStopping] Validation loss azaldı ({self.val_loss_min:.6f} --> {val_loss:.6f}). En iyi model kaydediliyor...")
        
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss
