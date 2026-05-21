from sklearn.decomposition import PCA
from src.config.config_loader import load_config
import numpy as np


def get_pca(config=None):
    if config is None:
        config = load_config()
    n_components = config.get("automata", {}).get("pca_n_components", 1)
    return PCA(n_components=n_components)


def fit_pca_on_train(pca, X_train):
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    pca.fit(X_train[numeric_cols])
    return pca


def transform_to_pc1(pca, df):
    if df is None:
        return None
    numeric_cols = df.select_dtypes(include=["number"]).columns
    transformed = pca.transform(df[numeric_cols])
    return transformed.flatten()
