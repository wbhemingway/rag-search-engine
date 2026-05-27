import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import MOVIE_EMBEDDINGS_PATH, Movie, load_movies


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Text to be embedded cannot be empty")
        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        doc_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(doc_strings, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                self.documents = documents
                self.document_map = {doc["id"]: doc for doc in documents}
                return self.embeddings
        self.embeddings = self.build_embeddings(documents)
        return self.embeddings


def verify_model() -> None:
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")


def embed_text(text: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    model = SemanticSearch()
    documents = load_movies()
    embeddings = model.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
