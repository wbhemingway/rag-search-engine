import os

import re

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    MOVIE_EMBEDDINGS_PATH,
    Movie,
    load_movies,
)


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

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


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


def embed_query_text(query: str) -> None:
    model = SemanticSearch()
    embeded_query = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embeded_query[:3]}")
    print(f"Shape: {embeded_query.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_sim = dot_product / (norm1 * norm2)
    return cos_sim


def semantic_search(query: str, limit: int = 5) -> None:
    model = SemanticSearch()
    movies = load_movies()
    model.load_or_create_embeddings(movies)

    results = model.search(query, limit)

    for i, result in enumerate(results):
        print(f"{i}. Title: {result['title']} (score: {result['score']:.4f})")
        print(
            f"Description: {result['description'][:200]}{'...' if len(result['description']) > 200 else ''}"
        )
        print()


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunk(
    text: str,
    chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    
    chunks = []
    n_sentences = len(sentences)
    i = 0
    while i < n_sentences:
        chunk_sentences = sentences[i:i + chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += chunk_size - overlap

    return chunks