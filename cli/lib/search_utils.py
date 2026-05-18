import json
import os
import string

from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize(text: str) -> list[str]:
    stop_words = load_stop_words()
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


def token_match(search_terms: list[str], targets: list[str]) -> bool:
    return any(term in target for target in targets for term in search_terms)


def load_stop_words() -> list[str]:
    with open(os.path.join(PROJECT_ROOT, "data", "stopwords.txt"), "r") as f:
        return list(f.read().splitlines())
