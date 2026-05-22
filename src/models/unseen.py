import Levenshtein


def is_unseen_pattern(pattern, state_dictionary):
    return pattern not in state_dictionary


def compute_levenshtein_distance(pattern1, pattern2):
    return Levenshtein.distance(pattern1, pattern2)


def find_nearest_pattern(unseen_pattern, state_dictionary):
    best_match = None
    min_distance = float('inf')
    
    for known_pattern in state_dictionary:
        dist = compute_levenshtein_distance(unseen_pattern, known_pattern)
        if dist < min_distance:
            min_distance = dist
            best_match = known_pattern
            
    return {
        "mapped_to": best_match,
        "nearest_distance": min_distance
    }


