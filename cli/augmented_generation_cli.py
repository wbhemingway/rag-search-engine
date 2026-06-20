import argparse

from lib.rag import citations_command, question_command, rag_command, summarize_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Get a RAG summary based off query"
    )
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results to summarize"
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Get a response with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results to cite"
    )

    question_parser = subparsers.add_parser(
        "question", help="Get a response with citations"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results to cite"
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            result = rag_command(query)
            results = result["results"]
            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")
            print()
            print("RAG Response:")
            print(result["rag"])
        case "summarize":
            query = args.query
            result = summarize_command(query, args.limit)
            results = result["results"]
            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")
            print()
            print("LLM Summary")
            print(result["summary"])
        case "citations":
            query = args.query
            result = citations_command(query, args.limit)
            results = result["results"]
            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")
            print()
            print("LLM Answer")
            print(result["citations"])
        case "question":
            query = args.query
            result = question_command(query, args.limit)
            results = result["results"]
            print("Search Results:")
            for res in results:
                print(f"- {res['title']}")
            print()
            print("LLM Answer")
            print(result["answer"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
