from collections import defaultdict

def extract_patterns(symbolic_sequence, window_size):
    patterns = []
    n = len(symbolic_sequence)
    for i in range(n - window_size + 1):
        pattern = symbolic_sequence[i : i + window_size]
        patterns.append(pattern)
    return patterns


def build_state_dictionary(patterns):
    return set(patterns)


def count_transitions(patterns):
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(patterns) - 1):
        curr_state = patterns[i]
        next_state = patterns[i + 1]
        counts[curr_state][next_state] += 1
    return counts

