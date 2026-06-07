import torch
import torch.nn as nn
import numpy as np
from src.config.config_loader import load_config
from src.data.dataset import get_dataloader
from src.models.lstm_model import LSTMClassifier
from src.models.cnn_model import CNN1DClassifier
from src.models.optimization import get_loss_and_optimizer
from src.models.callbacks import EarlyStopping
import uuid
import os


def train_one_epoch(model, dataloader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        
        # Backward Pass ve Ağırlık Güncelleme
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * X_batch.size(0)
        
    return running_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader, criterion, device):

    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            
            # Olasılıklara dönüştürüp (sigmoid) 0.5 eşiği ile sınıflandıralım
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, np.array(all_preds), np.array(all_targets)


def train_model(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience, device, checkpoint_path="checkpoint.pt"):

    early_stopping = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path, verbose=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _ = evaluate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}/{max_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
        
        # Early Stopping Kontrolü
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("-> Early stopping tetiklendi. Eğitim sonlandırılıyor.")
            break
            
    # En iyi model ağırlıklarını geri yükle
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        os.remove(checkpoint_path)
        
    return model, train_losses, val_losses


def run_dl_pipeline(X_train, y_train, X_val, y_val, X_test, y_test, model_name, input_size, config=None):

    if config is None:
        config = load_config()
        
    # Cihaz seçimi (GPU varsa CUDA, yoksa CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Model {model_name} '{device}' cihazı üzerinde çalıştırılacak.")
    
    # 1. DataLoader'ları oluştur (Zaman sırasını bozmamak için shuffle=False tutulur)
    sequence_length = config.get("deep_learning", {}).get("sequence_length", 10)
    batch_size = config.get("deep_learning", {}).get("batch_size", 32)
    
    train_loader = get_dataloader(X_train, y_train, sequence_length, batch_size, shuffle=False, config=config)
    val_loader = get_dataloader(X_val, y_val, sequence_length, batch_size, shuffle=False, config=config)
    test_loader = get_dataloader(X_test, y_test, sequence_length, batch_size, shuffle=False, config=config)
    
    # 2. Model nesnesini oluştur
    model_name = model_name.upper()
    if model_name == "LSTM":
        model = LSTMClassifier(input_size=input_size, config=config)
    elif model_name == "1D-CNN" or model_name == "CNN":
        model = CNN1DClassifier(input_size=input_size, config=config)
    elif model_name == "GRU":
        # 6.7 maddesi yapıldığında GRUClassifier import edilip kullanılacaktır.
        try:
            from src.models.gru_model import GRUClassifier
            model = GRUClassifier(input_size=input_size, config=config)
        except ImportError:
            raise NotImplementedError(
                "GRUClassifier henüz kodlanmadı (Madde 6.7). Lütfen önce onu tanımlayın veya model adını değiştirin."
            )
    else:
        raise ValueError(f"Bilinmeyen model tipi: {model_name}. Lütfen 'LSTM', 'CNN' veya 'GRU' seçin.")
        
    model = model.to(device)
    
    # 3. Loss ve Optimizer nesnelerini tanımla
    criterion, optimizer = get_loss_and_optimizer(model, config=config)
    
    # 4. Modeli eğit
    max_epochs = config.get("deep_learning", {}).get("max_epochs", 50)
    patience = config.get("deep_learning", {}).get("early_stopping_patience", 5)
    checkpoint_path = f"best_{model_name.lower()}_{uuid.uuid4().hex[:8]}.pt"
    
    model, train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    # 5. Test seti üzerinde tahmin üret
    test_loss, preds, targets = evaluate_model(model, test_loader, criterion, device)
    
    print(f"[SUCCESS] {model_name} eğitimi ve test değerlendirmesi tamamlandı! Test Loss: {test_loss:.6f}")
    
    return {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_loss,
        "predictions": preds,
        "targets": targets
    }
