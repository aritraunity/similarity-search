from abc import ABC, abstractmethod
from similaritysearch.math import (
    euclidean_distance,
    dot_product,
    cosine_similarity,
    matrix_cosine_similarity,
    matrix_cosine_similarity_multi
)
from sentence_transformers import SentenceTransformer
import numpy as np
from similaritysearch.logger import logger

def initialize ():
    cache_path = '/home/aritra-mukherjee/projects/similarity-search/cache/'
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=cache_path)
    return model

# Abstract Factory Class
class AbstractSearchFactory (ABC):
    
    """
    
    """

    def __init__(self, query: str, documents: list) -> None:
        self.model = initialize()
        self.query = query
        self.documents = documents
        self.document_embeddings = self.model.encode(documents)
        self.query_embeddings = self.model.encode([query])
        logger.debug(f"Initialized Document and Query Embeddings")
        logger.debug(f"Document Embedding Shape: {self.document_embeddings.shape}")
        logger.debug(f"Query Embedding Shape: {self.query_embeddings.shape}")
        logger.debug(f"Document Embeddings Preview:\n {self.document_embeddings[:,:3]}")
        logger.debug(f"Query Embeddings Preview:\n {self.query_embeddings[:,:3]}")
    @abstractmethod
    def search (self):
        pass

# Concrete Implementation of Abstract Search Factory
class EuclideanSimilaritySearch (AbstractSearchFactory):
    """
    
    """

    def __init__(self, query:str, documents:list) -> None:
        super().__init__(query=query, documents=documents)

    def search(self):
        doc_shape_x = self.document_embeddings.shape[0]
 
        l2_distances = []
        
        for i in range (doc_shape_x):
            logger.debug(f"INDEX: {i} Performing L2 Dist between {self.document_embeddings[i, 0:3]} and {self.query_embeddings[0, 0:3]}")
            l2_distances.append (euclidean_distance(self.document_embeddings[i], self.query_embeddings[0]))
        
        logger.debug(msg=f"L2 Preview: \n{l2_distances}")
        min_l2 = min(l2_distances)
        min_l2_id = l2_distances.index(min_l2)
        logger.debug(f"Min: {min_l2} at {min_l2_id}")
        return self.documents[min_l2_id]
    
# Concrete 2
class DotProductSimilaritySearch (AbstractSearchFactory):
    """
    
    """

    def __init__(self, query:str, documents:list) -> None:
        super().__init__(query=query, documents=documents)
    
    def search(self):
        x_shape = self.document_embeddings.shape[0]
        dot_products = []
        for i in range(x_shape):
            dot_products.append(dot_product(self.document_embeddings[i], self.query_embeddings[0]))
        
        max_dot = max(dot_products)
        max_dot_id = dot_products.index(max_dot)
        logger.debug(f"Max Dot Product: {max_dot} at {max_dot_id}")
        return self.documents[max_dot_id]
    
# Concrete 3
class CosineSimilaritySearch (AbstractSearchFactory):
    """
    
    """

    def __init__(self, query:str, documents:list) -> None:
        super().__init__(query=query, documents=documents)
    
    def search(self):
        x_shape = self.document_embeddings.shape[0]
        cosine_similarities = []
        for i in range(x_shape):
            cosine_similarities.append(cosine_similarity(self.document_embeddings[i], self.query_embeddings[0]))
        logger.debug(f"Cosine Similarities Preview:\n{cosine_similarities}")
        max_cossm = max(cosine_similarities)
        max_cossm_id = cosine_similarities.index(max_cossm)
        logger.debug(f"Max Co-Sine Similarity: {max_cossm} at {max_cossm_id}")

        return self.documents[max_cossm_id]
     

def search_paragraph_manual (paragraph: str, query: str):
    if len(paragraph) == 0 or len(query) == 0:
        raise ValueError("String should not be empty")
    sentences = paragraph.split('.')
    model = initialize()

#Remove
documents = [
    'Bugs introduced by the intern had to be squashed by the lead developer.',
    'Bugs found by the quality assurance engineer were difficult to debug.',
    'Bugs are common throughout the warm summer months, according to the entomologist.',
    'Bugs, in particular spiders, are extensively studied by arachnologists.'
]
query = "Who is responsible for a coding project and fixing others' mistakes?"

ecdist = EuclideanSimilaritySearch(query=query, documents=documents)
ecdist.search()
dotdist = DotProductSimilaritySearch(query=query, documents=documents)
dotdist.search()
cossinesim = CosineSimilaritySearch(query=query, documents=documents)
cossinesim.search()

