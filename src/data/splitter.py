from sklearn.model_selection import GroupKFold
from src.config.config_loader import load_config


def skab_group_kfold(X, y, groups, config=None):
    if config is None:
        config = load_config()
    n_splits = config["data"].get("skab_n_splits", 5)
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))
    return splits
