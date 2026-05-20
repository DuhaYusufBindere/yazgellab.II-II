import pandas as pd
import os
from src.config.config_loader import load_config


def read_valve1(config=None):
    if config is None:
        config = load_config()
    skab_path = config["data"]["skab_path"]
    valve1_path = os.path.join(skab_path, "valve1")
    dfs = []
    for filename in sorted(os.listdir(valve1_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(valve1_path, filename)
            df = pd.read_csv(filepath, sep=";")
            dfs.append(df)
    return dfs


def read_valve2(config=None):
    if config is None:
        config = load_config()
    skab_path = config["data"]["skab_path"]
    valve2_path = os.path.join(skab_path, "valve2")
    dfs = []
    for filename in sorted(os.listdir(valve2_path)):
        if filename.endswith(".csv"):
            filepath = os.path.join(valve2_path, filename)
            df = pd.read_csv(filepath, sep=";")
            dfs.append(df)
    return dfs


def concat_skab(valve1_dfs, valve2_dfs):
    all_dfs = valve1_dfs + valve2_dfs
    return pd.concat(all_dfs, ignore_index=True)
