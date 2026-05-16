import math
from similaritysearch.logger import logger
from scipy.spatial import distance

# L2 Distance
def euclidean_distance (vector1, vector2, verbose = False):
    """
    Calculate the Euclidean or (L2) distance between two vectors
    """
    cmp = zip(vector1, vector2)
    l2 = sum((x - y)**2 for x,y in cmp)
    l2 = math.sqrt(l2)
    if verbose:
        args = {
            "vector1": vector1,
            "vector2": vector2
        }
        logger.value("L2 Distance", str(l2), kwargs=args)
    return l2

# Sci Py implementation of Euclidean Distance
def scipy_euclidean_distance(embeddings):
    return distance.cdist(embeddings, embeddings, metric='euclidean')

# Dot product similarity: Sum (qi.ri) where i = 0 to n where q and i represent the components of the two vectors
def dot_product (vector1, vector2, verbose = False):
    """
    Calculates the dot product of two vectors, without matrices
    Higher the dot product, the more similar they are
    """
    dot = 0
    assert len(vector1) == len(vector2)
    n = len(vector1)
    # Longer Expression
    # for q in range(n):
    #     for r in range(n):
    #         sum += vector1[q] * vector2[r]
    # Zipped Expression
    # Comprehension: (Expression Loop Condition)
    dot = sum(x*y for x, y in zip(vector1, vector2))
    return dot

def normalized (vector1)->list[float]:
    """
    Normalize a vector
    """
    normalized_vector = []
    # List Comprehension: Expression Loop Condition -> List
    magnitude = math.sqrt(sum(x ** 2 for x in vector1))
    normalized_vector = list(float(x) / magnitude for x in vector1)
    return normalized_vector


# Co-sine similarity
def cosine_similarity (vector1, vector2):
    """
    Cosine similarity is calculated as the dot product of two normalized vectors
    """
    #Finding the norm (A) and norm (B)
    normA = normalized(vector1)
    normB = normalized(vector2)
    #Cosine Similarity
    return dot_product(normA, normB)

