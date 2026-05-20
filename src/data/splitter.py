from sklearn.model_selection import GroupKFold
from src.config.config_loader import load_config


def skab_group_kfold(X, y, groups, config=None):
    if config is None:
        config = load_config()
    n_splits = config["data"].get("skab_n_splits", 5)
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))
    return splits


def batadal_train_split(X, y, config=None):
    
    if config is None:
        config = load_config()
    
    train_split = config["data"].get("train_split", 0.6)
    
    n_samples = len(X)
    train_end = int(n_samples * train_split)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    return X_train, y_train

def batadal_val_split(X, y, config=None):

    if config is None:
        config = load_config()
    
    train_split = config["data"].get("train_split", 0.6)
    val_split = config["data"].get("val_split", 0.2)
    
    n_samples = len(X)
    val_start = int(n_samples * train_split)
    val_end = int(n_samples * (train_split + val_split))
    
    X_val = X.iloc[val_start:val_end]
    y_val = y.iloc[val_start:val_end]
    
    return X_val, y_val


def batadal_test_split(X, y, config=None):

    if config is None:
        config = load_config()
        
    train_split = config["data"].get("train_split", 0.6)
    val_split = config["data"].get("val_split", 0.2)
    
    n_samples = len(X)
    test_start = int(n_samples * (train_split + val_split))
    
    X_test = X.iloc[test_start:]
    y_test = y.iloc[test_start:]
    
    return X_test, y_test