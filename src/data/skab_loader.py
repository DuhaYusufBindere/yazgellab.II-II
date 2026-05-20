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
