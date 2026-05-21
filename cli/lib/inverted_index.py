import os
from collections import Counter
from pickle import dump, load
from typing import Any

from .search_utils import (
    CACHE_DIR,
    DOCMAP_PATH,
    INDEX_PATH,
    TERMF_PATH,
    load_movies,
    tokenize,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict[str, Any]] = {}
        self.term_frequecies: dict[int, Counter[str]] = {}

    def __add_doc(self, doc_id: int, text: str) -> None:
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            if doc_id not in self.term_frequecies:
                self.term_frequecies[doc_id] = Counter()
            self.term_frequecies[doc_id][token] += 1
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_set = self.index.get(term, set())
        return sorted(list(doc_set))

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize(term)
        if len(token) != 1:
            raise Exception("Term must be a single token")
        return self.term_frequecies.get(doc_id, Counter()).get(token[0], 0)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_doc(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(INDEX_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            dump(self.docmap, f)
        with open(TERMF_PATH, "wb") as f:
            dump(self.term_frequecies, f)

    def load(self) -> None:
        if (
            not os.path.exists(INDEX_PATH)
            or not os.path.exists(DOCMAP_PATH)
            or not os.path.exists(TERMF_PATH)
        ):
            raise Exception("Files not found")
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)
        with open(TERMF_PATH, "rb") as f:
            self.term_frequecies = load(f)
