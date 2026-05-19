import os
from pickle import dump, load

from .search_utils import CACHE_DIR, DOCMAP_PATH, INDEX_PATH, load_movies, tokenize


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add_doc(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term):
        doc_set = self.index.get(term, set())
        return sorted(list(doc_set))

    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_doc(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        with open(INDEX_PATH, "wb") as f:
            dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            dump(self.docmap, f)

    def load(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCMAP_PATH):
            raise Exception("Index not found")
        with open(INDEX_PATH, "rb") as f:
            self.index = load(f)
        with open(DOCMAP_PATH, "rb") as f:
            self.docmap = load(f)
