#!/usr/bin/env python3
import argparse

from lib.keyword_search import search_command, build_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search Cli")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build invterted index")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query)
            for result in results:
                print(f"{result[0]}. {result[1]}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
