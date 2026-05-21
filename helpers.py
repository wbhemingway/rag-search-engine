from typing import Any, cast
import json

def load_movies() -> dict[str, Any]:
    path = "data/movies.json"
    with open(path, "r") as f:
        return cast(dict[str, Any], json.loads(f.read()))