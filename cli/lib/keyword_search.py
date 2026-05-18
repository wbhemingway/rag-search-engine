from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    preprocess_text,
    token_match,
    tokenize,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        if token_match(
            tokenize(preprocess_text(query)), tokenize(preprocess_text(movie["title"]))
        ):
            results.append(movie)
            if len(results) == limit:
                break
    return results
