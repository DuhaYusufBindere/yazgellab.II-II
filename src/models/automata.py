def extract_patterns(symbolic_sequence, window_size):
    patterns = []
    n = len(symbolic_sequence)
    for i in range(n - window_size + 1):
        pattern = symbolic_sequence[i : i + window_size]
        patterns.append(pattern)
    return patterns
