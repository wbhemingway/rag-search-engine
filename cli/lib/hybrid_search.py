import os

from .keyword_search import InvertedIndex
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_LIMIT_MULTIPLIER,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    SearchResult,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents: list[Movie]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * DEFAULT_LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * DEFAULT_LIMIT_MULTIPLIER
        )
        combined_results = combine_search_results(bm25_results, semantic_results, alpha)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[Movie]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]


def normalize_search_result(results: list[SearchResult]) -> list[dict]:
    scores = [result["score"] for result in results]
    normalized_scores = normalize_scores(scores)
    return [
        {**result, "normalized_score": normalized_scores[i]}
        for i, result in enumerate(results)
    ]


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    alpha: float = DEFAULT_ALPHA,
) -> list[dict]:
    bm25_norm = normalize_search_result(bm25_results)
    semantic_norm = normalize_search_result(semantic_results)

    combined_scores = {}
    for result in bm25_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["bm25_score"]:
            combined_scores[doc_id]["bm25_score"] = result["normalized_score"]

    for result in semantic_norm:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0.0,
                "semantic_score": 0.0,
            }
        if result["normalized_score"] > combined_scores[doc_id]["semantic_score"]:
            combined_scores[doc_id]["semantic_score"] = result["normalized_score"]

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        hybrid_results.append(
            {
                "id": doc_id,
                "title": data["title"],
                "document": data["document"],
                "score": hybrid_score(
                    data["bm25_score"], data["semantic_score"], alpha
                ),
                "bm25_score": data["bm25_score"],
                "semantic_score": data["semantic_score"],
            }
        )

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    original_query = query

    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)

    return {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }
