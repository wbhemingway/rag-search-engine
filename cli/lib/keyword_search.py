from .inverted_index import InvertedIndex
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    tokenize,
)


def search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[tuple[int, str]]:
    index = InvertedIndex()
    index.load()
    results: list[tuple[int, str]] = []
    tokens = tokenize(query)
    for token in tokens:
        doc_set = index.get_documents(token)
        for id in doc_set:
            id = int(id)
            results.append((id, index.docmap[id]["title"]))
            if len(results) == limit:
                return results
    return results


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def tf_command(doc_id: int, term: str) -> int:
    index = InvertedIndex()
    index.load()
    return index.get_tf(doc_id, term)


def idf_command(term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_idf(term)


def tfidf_command(doc_id: int, term: str) -> float:
    index = InvertedIndex()
    index.load()
    return index.get_tfidf(doc_id, term)
