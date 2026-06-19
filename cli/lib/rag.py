import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import HybridSearch
from .search_utils import (
    DEFAULT_K,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
)

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"


def rag_command(query: str) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    results = searcher.rrf_search(query, DEFAULT_K, DEFAULT_SEARCH_LIMIT)
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {"\n".join([f"{result.get('title', '')} - {result.get('document', '')[:200]}" for result in results])}

    Answer:"""
    response = client.models.generate_content(model=model, contents=prompt)
    rag = (response.text or "").strip()

    return {"query": query, "results": results, "rag": rag}


def summarize_command(query: str, limit: int) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)

    results = searcher.rrf_search(query, DEFAULT_K, limit)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {"\n".join([f"{result.get('title', '')} - {result.get('document', '')[:200]}" for result in results])}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(model=model, contents=prompt)
    summary = (response.text or "").strip()

    return {"query": query, "results": results, "summary": summary}
