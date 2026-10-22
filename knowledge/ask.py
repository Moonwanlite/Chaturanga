"""
Interactive CLI for asking Chaturanga rules/strategy questions, backed by
the RAG pipeline (knowledge/rag.py) over knowledge/rules_corpus/.

Run (either works):
    python knowledge/ask.py
    python -m knowledge.ask

Default backend is --backend extractive: zero setup, no LLM, no model
download, no server -- just the retrieved rule text itself. This is the
right default for handing the project to someone who shouldn't need to
install anything (e.g. submitting for grading). Other backends:
    --backend transformers  A small open-source model run in-process via
                             `pip install transformers` -- no separate app,
                             but downloads model weights on first use
                             (needs internet once). Real generated answers.
    --backend ollama         Best quality for its size, but needs Ollama
                             installed and running as a separate application,
                             plus a manual `ollama pull <model>` step.
    --backend anthropic       Hosted, needs ANTHROPIC_API_KEY set.
    --backend mock             Fixed placeholder text, no real answer at all --
                               only useful for testing the pipeline mechanically.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board.state import ChaturangaState
from search.search_agent import minimax_agent_move
from knowledge.rag import ChaturangaRAG
from knowledge.llm_client import (
    OllamaLLMClient, AnthropicLLMClient, MockLLMClient,
    TransformersLLMClient, ExtractiveLLMClient,
)


def main():
    parser = argparse.ArgumentParser(description="Ask questions about Chaturanga rules and strategy.")
    parser.add_argument("--backend", choices=["extractive", "transformers", "ollama", "anthropic", "mock"],
                         default="extractive",
                         help="which backend to use (default: extractive -- no setup needed at all)")
    parser.add_argument("--model", type=str, default=None,
                         help="model name override (e.g. 'llama3.1' for ollama, 'claude-sonnet-4-6' for "
                              "anthropic, an HF repo id for transformers)")
    args = parser.parse_args()

    try:
        if args.backend == "extractive":
            llm_client = ExtractiveLLMClient()
        elif args.backend == "transformers":
            kwargs = {"model_name": args.model} if args.model else {}
            print("Loading local model (first run downloads it -- this may take a while)...")
            llm_client = TransformersLLMClient(**kwargs)
        elif args.backend == "ollama":
            kwargs = {"model": args.model} if args.model else {}
            llm_client = OllamaLLMClient(**kwargs)
        elif args.backend == "anthropic":
            kwargs = {"model": args.model} if args.model else {}
            llm_client = AnthropicLLMClient(**kwargs)
        else:
            llm_client = MockLLMClient()
    except Exception as e:
        print(f"\n[Error] Could not set up backend '{args.backend}': {type(e).__name__}: {e}\n")
        print("Falling back to --backend extractive (no setup required) for this session.")
        llm_client = ExtractiveLLMClient()
        args.backend = "extractive"

    print(f"Building retrieval index over knowledge/rules_corpus/ ...")
    rag = ChaturangaRAG(llm_client=llm_client)
    print(f"Ready. Backend: {args.backend}. Type a question, or 'quit' to exit.")
    print("(Board context is the starting position unless you set one up in code -- "
          "this CLI doesn't track an ongoing game.)")
    print()

    state = ChaturangaState()

    while True:
        try:
            query = input("Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        try:
            answer = rag.answer(query, state=state,
                                move_fn=lambda s: minimax_agent_move(s, depth=3, verbose=False),
                                engine_label="Minimax (depth 3)")
        except ConnectionError as e:
            print(f"\n[Error] {e}\n")
            continue
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}\n")
            continue

        print()
        print(answer)
        print()


if __name__ == "__main__":
    main()