#!/usr/bin/env python3
import argparse

from lib.semantic_search import (
    chunk_text,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    semantic_chunk,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the model")

    embed_parser = subparsers.add_parser("embed_text", help="Generate embedding")
    embed_parser.add_argument("text", help="Text to generate embedding for")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings of the movies")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query text")
    embed_query_parser.add_argument("query", help="Query to generate embedding for")

    search_parser = subparsers.add_parser(
        "search", help="Search semantically for the query within movies"
    )
    search_parser.add_argument("query", help="Query to search for")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a given text")
    chunk_parser.add_argument("text", help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Maximum chunk size"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk a text into sentences"
    )
    semantic_chunk_parser.add_argument("text", help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum chunk size"
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )

    subparsers.add_parser(
        "embed_chunks", help="Generate Embeddings for chunked documents"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "embed_chunks":
            embeddings = embed_chunks_command()
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
