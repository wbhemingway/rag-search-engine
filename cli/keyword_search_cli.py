#!/usr/bin/env python3
import argparse

from lib.keyword_search import build_command, search_command, tf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search Cli")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build invterted index")  # noqa: F841

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query)
            for result in results:
                print(f"{result[0]}. {result[1]}")
        case "build":
            build_command()
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(tf)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
