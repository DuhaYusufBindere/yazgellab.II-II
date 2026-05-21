from sklearn.decomposition import PCA
from src.config.config_loader import load_config


def get_pca(config=None):
    if config is None:
        config = load_config()
    n_components = config.get("automata", {}).get("pca_n_components", 1)
    return PCA(n_components=n_components)

