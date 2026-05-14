from similaritysearch.logger import logger

# L2 Distance
def euclidean_distance (vector1, vector2, verbose = False):
    """
    Calculate the Euclidean or (L2) distance between two vectors
    """
    cmp = zip(vector1, vector2)
    l2 = sum((x - y)**2 for x,y in cmp)
    if verbose:
        args = {
            "vector1": vector1,
            "vector2": vector2
        }
        logger.value("L2 Distance", str(l2), kwargs=args)
    return l2

