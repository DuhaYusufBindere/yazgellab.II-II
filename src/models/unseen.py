import Levenshtein


def is_unseen_pattern(pattern, state_dictionary):
    return pattern not in state_dictionary


def compute_levenshtein_distance(pattern1, pattern2):
    return Levenshtein.distance(pattern1, pattern2)

