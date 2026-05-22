from typing import Any, cast
import json
import os
import string

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT: int = 5

PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR: str = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH: str = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH: str = os.path.join(CACHE_DIR, "docmap.pkl")
TERMF_PATH: str = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[dict[str, Any]]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return cast(list[dict[str, Any]], data["movies"])


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize(text: str) -> list[str]:
    stop_words = load_stop_words()
    text = preprocess_text(text)
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def token_match(search_terms: list[str], targets: list[str]) -> bool:
    return any(term in target for target in targets for term in search_terms)


def load_stop_words() -> list[str]:
    with open(os.path.join(PROJECT_ROOT, "data", "stopwords.txt"), "r") as f:
        return list(f.read().splitlines())

def enforced_tokenize(text: str) -> list[str]:
    token = tokenize(text)
    if len(token) != 1:
        raise Exception("Term must be a single token")
    return token