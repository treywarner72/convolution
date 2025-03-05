from itertools import chain

def flatten(lst):
    """
    Converts nested lists into a single list
    """
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
