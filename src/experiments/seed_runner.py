from utils.seed import set_seed

SEEDS = [42, 123, 2026, 7, 999]


def run_with_seeds(experiment_fn, **kwargs):
    all_results = []

    for seed in SEEDS:
        set_seed(seed)
        print(f"[SEED {seed}] Deney başlatılıyor...")

        result = experiment_fn(seed=seed, **kwargs)
        result["seed"] = seed
        all_results.append(result)

        print(f"[SEED {seed}] Deney tamamlandı.")

    return all_results
