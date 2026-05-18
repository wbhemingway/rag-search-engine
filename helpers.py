import json

def load_movies() -> dict[str, str]:
    path = "data/movies.json"
    with open(path, "r") as f:
        return json.loads(f.read())