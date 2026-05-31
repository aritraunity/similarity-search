from abc import ABC, abstractmethod
from similaritysearch.math import (
    euclidean_distance,
    dot_product,
    cosine_similarity,
)
from sentence_transformers import SentenceTransformer
import numpy as np
from similaritysearch.logger import logger


# Abstract Products

class AbstractEmbedder(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        pass


class AbstractSimilarityMetric(ABC):
    @abstractmethod
    def score(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> list[float]:
        pass

    @abstractmethod
    def best_match_index(self, scores: list[float]) -> int:
        pass


# Abstract Factory

class AbstractSearchFactory(ABC):
    @abstractmethod
    def create_embedder(self) -> AbstractEmbedder:
        pass

    @abstractmethod
    def create_metric(self) -> AbstractSimilarityMetric:
        pass


# Concrete Product — Embedder

class SentenceTransformerEmbedder(AbstractEmbedder):
    _cache_path = '/home/aritra-mukherjee/projects/similarity-search/cache/'

    def __init__(self):
        self._model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder=self._cache_path)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts))


# Concrete Products — Similarity Metrics

class EuclideanMetric(AbstractSimilarityMetric):
    def score(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> list[float]:
        scores = [euclidean_distance(document_embeddings[i], query_embedding)
                  for i in range(document_embeddings.shape[0])]
        logger.debug(f"L2 Distances: {scores}")
        return scores

    def best_match_index(self, scores: list[float]) -> int:
        return scores.index(min(scores))


class DotProductMetric(AbstractSimilarityMetric):
    def score(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> list[float]:
        scores = [float(dot_product(document_embeddings[i], query_embedding))
                  for i in range(document_embeddings.shape[0])]
        logger.debug(f"Dot Products: {scores}")
        return scores

    def best_match_index(self, scores: list[float]) -> int:
        return scores.index(max(scores))


class CosineMetric(AbstractSimilarityMetric):
    def score(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> list[float]:
        scores = [float(cosine_similarity(document_embeddings[i], query_embedding))
                  for i in range(document_embeddings.shape[0])]
        logger.debug(f"Cosine Similarities: {scores}")
        return scores

    def best_match_index(self, scores: list[float]) -> int:
        return scores.index(max(scores))


# Concrete Factories

class EuclideanSearchFactory(AbstractSearchFactory):
    def create_embedder(self) -> AbstractEmbedder:
        return SentenceTransformerEmbedder()

    def create_metric(self) -> AbstractSimilarityMetric:
        return EuclideanMetric()


class DotProductSearchFactory(AbstractSearchFactory):
    def create_embedder(self) -> AbstractEmbedder:
        return SentenceTransformerEmbedder()

    def create_metric(self) -> AbstractSimilarityMetric:
        return DotProductMetric()


class CosineSearchFactory(AbstractSearchFactory):
    def create_embedder(self) -> AbstractEmbedder:
        return SentenceTransformerEmbedder()

    def create_metric(self) -> AbstractSimilarityMetric:
        return CosineMetric()


# Search Client

class SimilaritySearch:
    def __init__(self, factory: AbstractSearchFactory):
        self._embedder = factory.create_embedder()
        self._metric = factory.create_metric()

    def search(self, query: str, documents: list[str]) -> str:
        document_embeddings = self._embedder.encode(documents)
        query_embedding = self._embedder.encode([query])[0]
        logger.debug(f"Document Embedding Shape: {document_embeddings.shape}")
        logger.debug(f"Query Embedding Shape: {query_embedding.shape}")
        scores = self._metric.score(query_embedding, document_embeddings)
        best_idx = self._metric.best_match_index(scores)
        logger.debug(f"Best match at index {best_idx}")
        return documents[best_idx]


# Implementation Example
#
# documents = [
#     'Bugs introduced by the intern had to be squashed by the lead developer.',
#     'Bugs found by the quality assurance engineer were difficult to debug.',
#     'Bugs are common throughout the warm summer months, according to the entomologist.',
#     'Bugs, in particular spiders, are extensively studied by arachnologists.'
# ]
# query = "Who is responsible for a coding project and fixing others' mistakes?"
#
# # Euclidean distance — smallest L2 distance wins
# euclidean_search = SimilaritySearch(factory=EuclideanSearchFactory())
# print(euclidean_search.search(query=query, documents=documents))
#
# # Dot product — highest dot product wins
# dot_search = SimilaritySearch(factory=DotProductSearchFactory())
# print(dot_search.search(query=query, documents=documents))
#
# # Cosine similarity — highest cosine similarity wins
# cosine_search = SimilaritySearch(factory=CosineSearchFactory())
# print(cosine_search.search(query=query, documents=documents))