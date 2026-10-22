"""
RAG orchestration: retrieves relevant rules/strategy context for a query
and the current board, builds a prompt, and calls an LLM client.

Deliberately independent of gameplay logic -- this module never influences
move legality or search; it only explains/comments. board/encode.to_text
is used purely to give the LLM a human-readable snapshot of the position,
the same text representation a person would read.
"""

from knowledge.build_index import RulesIndex
from knowledge.llm_client import ExtractiveLLMClient
from knowledge.move_hints import is_move_request, build_engine_block


def build_prompt(query: str, retrieved_chunks, board_text: str, engine_block: str = "") -> str:
    # NOTE: the markers "Relevant rules/strategy context:\n" and "\n\nQuestion:"
    # are parsed by ExtractiveLLMClient -- keep them exactly as they are.
    engine_section = ""
    closing_extra = ""
    if engine_block:
        engine_section = f"\n{engine_block}\n"
        closing_extra = (" The engine recommendation above was computed by the game engine; "
                         "briefly explain the idea behind it using the context, and do NOT "
                         "suggest any different move.")
    context = "\n---\n".join(f"[{c['source']}] {c['text']}" for c in retrieved_chunks)
    return f"""You are a Chaturanga rules and strategy assistant. Chaturanga is \
a historical predecessor of chess with different pieces and rules -- do \
not assume modern chess rules apply unless the context below says so.

Current board position:
{board_text}
{engine_section}
Relevant rules/strategy context:
{context}

Question: {query}

Answer in 2-4 sentences, concisely and precisely, grounded ONLY in the \
context above. If the context doesn't cover something, say so rather \
than guessing. Do not invent additional questions or continue the \
conversation after your answer -- stop once you've answered.{closing_extra}"""


class ChaturangaRAG:
    def __init__(self, llm_client, index: RulesIndex = None):
        self.llm_client = llm_client
        self.index = index or RulesIndex.build()

    def answer(self, query: str, state=None, top_k: int = 3,
               move_fn=None, engine_label: str = "the game engine") -> str:
        """If the query asks for a next move / set of moves and `move_fn`
        (state -> move) is given, the moves come from the engine, are shown
        verbatim, and the LLM only explains them."""
        from board.encode import to_text

        engine_block = ""
        if move_fn is not None and state is not None and is_move_request(query):
            engine_block = build_engine_block(state, move_fn, query, engine_label)

        # Extractive backend can't explain anything -- show the engine's moves alone.
        if engine_block and isinstance(self.llm_client, ExtractiveLLMClient):
            return engine_block

        board_text = to_text(state) if state is not None else "(no board position provided)"
        retrieved = self.index.search(query, top_k=top_k)
        prompt = build_prompt(query, retrieved, board_text, engine_block)
        explanation = self.llm_client.complete(prompt)
        return f"{engine_block}\n\n{explanation}" if engine_block else explanation