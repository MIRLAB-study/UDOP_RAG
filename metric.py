def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def average_normalized_levenshtein_similarity(s1, s2):
    """
    Calculate the Average Normalized Levenshtein Similarity (ANLS) between two strings.
    
    Args:
    s1 (str): First string.
    s2 (str): Second string.
    
    Returns:
    float: ANLS score between 0.0 and 1.0, where 1.0 means identical.
    """
    distance = levenshtein_distance(s1, s2)
    average_length = (len(s1) + len(s2)) / 2
    
    if average_length == 0:  # Edge case for two empty strings
        return 1.0
    
    # Compute ANLS as 1 - (distance / average length of the two strings)
    return 1 - (distance / average_length)


def anls(s1, s2):
    """
    Calculate the Average Normalized Levenshtein Similarity (ANLS) between two strings.
    
    Args:
    s1 (str): First string.
    s2 (str): Second string.
    
    Returns:
    float: ANLS score between 0.0 and 1.0, where 1.0 means identical.
    """
    assert type(s1)==str or type(s1)==list, "s1 should be either a string or a list of strings"
    assert type(s2)==str, "s2 should be a string"
    
    if type(s1)==list:
        max_score = 0
        for i in range(len(s1)):
            max_score = max(max_score, average_normalized_levenshtein_similarity(s1[i], s2))
        return max_score
    elif type(s1)==str:
        return average_normalized_levenshtein_similarity(s1, s2)