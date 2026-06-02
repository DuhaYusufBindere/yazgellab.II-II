from itertools import product
from utils.seed import set_seed

SEEDS = [42, 123, 2026, 7, 999]
WINDOW_SIZES = [3, 4, 5, 6]
ALPHABET_SIZES = [3, 4, 5, 6]


def run_grid_search(experiment_fn, **kwargs):
    all_results = []

    for seed in SEEDS:
        for window_size, alphabet_size in product(WINDOW_SIZES, ALPHABET_SIZES):
            set_seed(seed)
            print(f"[GRID] Seed={seed}, window_size={window_size}, alphabet_size={alphabet_size}")

            result = experiment_fn(
                seed=seed,
                window_size=window_size,
                alphabet_size=alphabet_size,
                **kwargs
            )
            result["seed"] = seed
            result["window_size"] = window_size
            result["alphabet_size"] = alphabet_size
            all_results.append(result)

    print(f"[GRID] Toplam {len(all_results)} kombinasyon tamamlandı.")
    return all_results
