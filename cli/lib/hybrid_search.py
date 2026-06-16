import logging
import os
from typing import Literal, Optional, TypedDict

from .keyword_search import InvertedIndex
from .query_enhancement import enhance_query
from .reranking import rerank
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_K,
    DEFAULT_LIMIT_MULTIPLIER,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_MULTIPLIER,
    Movie,
    SearchResult,
    load_movies,
)
from .semantic_search import ChunkedSemanticSearch

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.basicConfig(filename="rrf_search_command.log", level=logging.INFO)


class RRFSearchCommandResult(TypedDict):
    original_query: str
    enhanced_query: Optional[str]
    enhance_method: Optional[Literal["spell", "expand", "rewrite"]]
    reranked: bool
    reranking_method: Optional[Literal["individual", "batch", "cross_encoder"]]
    query: str
    k: int
    results: list[SearchResult]


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

    def weighted_search(
        self, query: str, alpha: float, limit: int = 5
    ) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * DEFAULT_LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * DEFAULT_LIMIT_MULTIPLIER
        )
        combined_results = combine_search_results(bm25_results, semantic_results, alpha)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[SearchResult]:
        bm25_results = self._bm25_search(query, limit * DEFAULT_LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * DEFAULT_LIMIT_MULTIPLIER
        )
        combined_results = combine_search_results_rrf(bm25_results, semantic_results, k)
        return combined_results[:limit]


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
) -> list[SearchResult]:
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


def rrf_score(rank: int | None, k: int = DEFAULT_K) -> float:
    if rank is None:
        return 0.0
    return 1 / (k + rank)


def combine_search_results_rrf(
    bm25_results: list[SearchResult],
    semantic_results: list[SearchResult],
    k: int = DEFAULT_K,
) -> list[SearchResult]:
    ranked_bm25 = sorted(bm25_results, key=lambda x: x["score"], reverse=True)
    ranked_semantic = sorted(semantic_results, key=lambda x: x["score"], reverse=True)
    rrf_scores = {}
    for rank, result in enumerate(ranked_bm25, 1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm2f_rank": None,
                "semantic_rank": None,
            }
        rrf_scores[doc_id]["bm2f_rank"] = (
            rank
            if rrf_scores[doc_id]["bm2f_rank"] is None
            else rrf_scores[doc_id]["bm2f_rank"]
        )
        rrf_scores[doc_id]["score"] = rrf_score(rank, k)

    for rank, result in enumerate(ranked_semantic, 1):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "bm2f_rank": None,
                "semantic_rank": None,
            }
        rrf_scores[doc_id]["semantic_rank"] = (
            rank
            if rrf_scores[doc_id]["semantic_rank"] is None
            else rrf_scores[doc_id]["semantic_rank"]
        )
        rrf_scores[doc_id]["score"] = rrf_score(
            rrf_scores[doc_id]["bm2f_rank"], k
        ) + rrf_score(rrf_scores[doc_id]["semantic_rank"], k)

    return sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)


def rrf_search_command(
    query: str,
    k: int = DEFAULT_K,
    enhance: Optional[Literal["spell", "expand", "rewrite"]] = None,
    rerank_method: Optional[Literal["individual", "batch", "cross_encoder"]] = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
) -> RRFSearchCommandResult:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    logger.info(f"Original query: {original_query}")
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query
        logger.info(f"Enhanced query: {enhanced_query}")

    search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit
    results = searcher.rrf_search(query, k, search_limit)
    logger.info(f"Search results: {[res['title'] for res in results]}")
    reranked = rerank_method is not None
    if reranked:
        results = rerank(query, results, method=rerank_method, limit=limit)
        logger.info(f"Reranked results: {[res['title'] for res in results]}")
    return {
        "original_query": original_query,
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "reranked": reranked,
        "reranking_method": rerank_method,
        "query": query,
        "k": k,
        "results": results,
    }
