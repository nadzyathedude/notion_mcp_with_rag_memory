#!/usr/bin/env python3
"""CLI for the document indexing and RAG pipeline."""

import argparse
import sys
from pathlib import Path

from indexer.settings import PipelineConfig, ChunkingConfig, EmbeddingConfig, IndexConfig
from indexer.pipeline import IndexingPipeline
from indexer.rag import answer_question, answer_question_filtered, answer_question_cited, RAGResult


def _print_filtered_result(result: RAGResult) -> None:
    """Print stats and answer from a filtered RAG result."""
    print(f"Chunks retrieved: {result.chunks_retrieved}")
    print(f"Chunks after filter: {result.chunks_after_filter}")
    print(f"Avg similarity: {result.avg_similarity:.3f}")
    if result.avg_rerank_score is not None:
        print(f"Avg rerank score: {result.avg_rerank_score:.3f}")
    if result.used_fallback:
        print("Warning: No chunks met the threshold — used fallback (top-1)")
    if result.citation_validation is not None:
        cv = result.citation_validation
        status = "PASS" if cv.is_valid else "FAIL"
        print(f"Citation validation: {status}")
        if cv.citations_found:
            print(f"Citations found: {', '.join(cv.citations_found)}")
        if cv.invalid_citations:
            print(f"Invalid citations: {', '.join(cv.invalid_citations)}")
    print(f"\nAnswer:\n{result.answer}")


def _run_comparison(args, llm, reranker) -> None:
    """Run baseline vs enhanced pipeline and print side-by-side comparison."""
    from indexer.settings import FallbackStrategy

    print("=" * 60)
    print("BASELINE (no threshold, no reranker)")
    print("=" * 60)
    baseline = answer_question_filtered(
        question=args.question,
        index_path=args.index_file,
        top_k=args.top_k,
        threshold=0.0,
        reranker=None,
        fallback_strategy=FallbackStrategy.TOP_1,
        llm_provider=llm,
    )
    _print_filtered_result(baseline)

    print()
    print("=" * 60)
    print(f"ENHANCED (threshold={args.threshold}"
          f"{', reranker=on' if reranker else ''})")
    print("=" * 60)
    enhanced = answer_question_filtered(
        question=args.question,
        index_path=args.index_file,
        top_k=args.top_k,
        threshold=args.threshold,
        reranker=reranker,
        fallback_strategy=FallbackStrategy.TOP_1,
        llm_provider=llm,
    )
    _print_filtered_result(enhanced)

    # Summary
    print()
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"  Baseline chunks used:  {baseline.chunks_after_filter}")
    print(f"  Enhanced chunks used:  {enhanced.chunks_after_filter}")
    print(f"  Baseline avg sim:      {baseline.avg_similarity:.3f}")
    print(f"  Enhanced avg sim:      {enhanced.avg_similarity:.3f}")
    if enhanced.avg_rerank_score is not None:
        print(f"  Enhanced avg rerank:   {enhanced.avg_rerank_score:.3f}")
    if enhanced.used_fallback:
        print("  Note: Enhanced result used fallback (threshold too strict)")
    if enhanced.chunks_after_filter < baseline.chunks_after_filter:
        print("  Observation: Threshold filtered out low-relevance chunks")
    if enhanced.avg_rerank_score is not None and not enhanced.used_fallback:
        print("  Observation: Reranker re-scored chunks for better ordering")


def main():
    parser = argparse.ArgumentParser(
        description="Index documents and ask questions with RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a single file
  python main.py file document.txt

  # Index multiple files
  python main.py file doc1.txt doc2.txt doc3.txt

  # Index a directory (all .txt files)
  python main.py dir ./documents

  # Index directory recursively with custom pattern
  python main.py dir ./documents --pattern "*.md" --recursive

  # Custom chunk size and output
  python main.py file document.txt --chunk-size 1000 --overlap 200 --output my_index.json

  # Ask a question using RAG
  python main.py ask document_index.json "What is this document about?"

  # Ask with more context chunks
  python main.py ask document_index.json "Summarize the key points" --top-k 10

  # Ask with threshold filtering
  python main.py ask document_index.json "What is an Activity?" --threshold 0.75

  # Ask with reranker
  python main.py ask document_index.json "What is an Activity?" --use-reranker --threshold 0.6

  # Compare baseline vs enhanced
  python main.py ask document_index.json "What is an Activity?" --compare --threshold 0.7

  # Ask with enforced source citations
  python main.py ask document_index.json "What is an Intent?" --enforce-citations

  # Strict mode: fail if citations are missing
  python main.py ask document_index.json "What is an Intent?" --enforce-citations --strict
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # File command
    file_parser = subparsers.add_parser("file", help="Index one or more files")
    file_parser.add_argument("files", nargs="+", help="File paths to index")

    # Directory command
    dir_parser = subparsers.add_parser("dir", help="Index files in a directory")
    dir_parser.add_argument("directory", help="Directory path")
    dir_parser.add_argument(
        "--pattern", default="*.txt",
        help="Glob pattern for files (default: *.txt)"
    )
    dir_parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Search recursively"
    )

    # Text command (from stdin)
    text_parser = subparsers.add_parser("text", help="Index text from stdin")
    text_parser.add_argument(
        "--doc-id", default="stdin",
        help="Document ID (default: stdin)"
    )

    # Ask command (RAG)
    ask_parser = subparsers.add_parser("ask", help="Ask a question using RAG")
    ask_parser.add_argument("index_file", help="Path to the JSON index file")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    ask_parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model for generation (default: gpt-4o-mini)"
    )
    ask_parser.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity threshold for chunk filtering (0.0-1.0, default: 0.75)"
    )
    ask_parser.add_argument(
        "--use-reranker", action="store_true",
        help="Use LLM-based reranker for second-stage scoring"
    )
    ask_parser.add_argument(
        "--compare", action="store_true",
        help="Run baseline vs enhanced pipeline and show comparison"
    )
    ask_parser.add_argument(
        "--enforce-citations", action="store_true",
        help="Require the LLM to cite source chunk IDs in its answer"
    )
    ask_parser.add_argument(
        "--strict", action="store_true",
        help="Fail if citations are missing or invalid (use with --enforce-citations)"
    )

    # Common arguments for all commands
    for p in [file_parser, dir_parser, text_parser]:
        p.add_argument(
            "--output", "-o", default="document_index.json",
            help="Output JSON file path (default: document_index.json)"
        )
        p.add_argument(
            "--chunk-size", type=int, default=800,
            help="Chunk size in characters (default: 800)"
        )
        p.add_argument(
            "--overlap", type=int, default=150,
            help="Chunk overlap in characters (default: 150)"
        )
        p.add_argument(
            "--model", default="text-embedding-3-small",
            help="Embedding model (default: text-embedding-3-small)"
        )
        p.add_argument(
            "--batch-size", type=int, default=64,
            help="Batch size for embedding requests (default: 64)"
        )

    args = parser.parse_args()

    # Handle RAG ask command separately
    if args.command == "ask":
        try:
            from indexer.settings import LLMConfig, FallbackStrategy
            from indexer.llm import OpenAILLM
            from indexer.reranker import LLMReranker

            llm = OpenAILLM(LLMConfig(model=args.model))

            # Validate threshold
            if args.threshold is not None and not 0.0 <= args.threshold <= 1.0:
                print("Error: --threshold must be between 0.0 and 1.0",
                      file=sys.stderr)
                sys.exit(1)

            # Create reranker if requested
            reranker = LLMReranker(llm) if args.use_reranker else None

            use_filtered = (args.threshold is not None
                            or args.use_reranker
                            or args.compare)

            if args.compare:
                # Comparison mode
                threshold = args.threshold if args.threshold is not None else 0.75
                args.threshold = threshold
                _run_comparison(args, llm, reranker)
            elif args.enforce_citations:
                # Citation-enforced pipeline
                threshold = args.threshold if args.threshold is not None else 0.75
                result = answer_question_cited(
                    question=args.question,
                    index_path=args.index_file,
                    top_k=args.top_k,
                    threshold=threshold,
                    reranker=reranker,
                    fallback_strategy=FallbackStrategy.TOP_1,
                    llm_provider=llm,
                    strict=args.strict,
                )
                _print_filtered_result(result)
            elif use_filtered:
                # Filtered pipeline
                threshold = args.threshold if args.threshold is not None else 0.75
                result = answer_question_filtered(
                    question=args.question,
                    index_path=args.index_file,
                    top_k=args.top_k,
                    threshold=threshold,
                    reranker=reranker,
                    fallback_strategy=FallbackStrategy.TOP_1,
                    llm_provider=llm,
                )
                _print_filtered_result(result)
            else:
                # Original baseline pipeline
                result = answer_question(
                    question=args.question,
                    index_path=args.index_file,
                    top_k=args.top_k,
                    llm_provider=llm,
                )
                print(result)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error during RAG: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Build configuration for indexing commands
    config = PipelineConfig(
        chunking=ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap
        ),
        embedding=EmbeddingConfig(
            model=args.model,
            batch_size=args.batch_size
        ),
        index=IndexConfig(
            output_path=args.output
        )
    )

    try:
        pipeline = IndexingPipeline(config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    total_chunks = 0

    try:
        if args.command == "file":
            for file_path in args.files:
                if not Path(file_path).exists():
                    print(f"Warning: File not found: {file_path}", file=sys.stderr)
                    continue
                chunks = pipeline.add_file(file_path)
                print(f"Indexed {file_path}: {chunks} chunks")
                total_chunks += chunks

        elif args.command == "dir":
            if not Path(args.directory).is_dir():
                print(f"Error: Not a directory: {args.directory}", file=sys.stderr)
                sys.exit(1)
            total_chunks = pipeline.add_directory(
                args.directory,
                pattern=args.pattern,
                recursive=args.recursive
            )
            print(f"Indexed directory {args.directory}: {total_chunks} chunks")

        elif args.command == "text":
            text = sys.stdin.read()
            if not text.strip():
                print("Error: No text provided on stdin", file=sys.stderr)
                sys.exit(1)
            total_chunks = pipeline.add_text(text, args.doc_id)
            print(f"Indexed stdin: {total_chunks} chunks")

        if total_chunks > 0:
            output_path = pipeline.save()
            stats = pipeline.get_stats()
            print(f"\nIndex saved to: {output_path}")
            print(f"Total documents: {stats['total_documents']}")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Embedding model: {stats['embedding_model']}")
        else:
            print("No content indexed.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error during indexing: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
