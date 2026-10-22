"""
Simple click-to-move Chaturanga GUI, built with Tkinter (bundled with Python,
no extra install required).

Click a square to select a piece (legal destinations highlight), click a
highlighted square to move there. Rules/legality come entirely from
board/state.py -- this file only handles drawing and input.

Run from the project root:
    python gui/chaturanga_gui.py
"""

import argparse
import sys
import os
import random
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board.state import ChaturangaState, BOARD_SIZE
from search.search_agent import (
    minimax_agent_move, cnn_agent_move, rl_agent_move, mcts_agent_move, minimax_ppo_agent_move,
    DEFAULT_DEPTH, DEFAULT_CNN_DEPTH, DEFAULT_CHECKPOINT_PATH, DEFAULT_PPO_PATH,
    DEFAULT_MCTS_SIMULATIONS, DEFAULT_PPO_EVAL_DEPTH,
)
from knowledge.rag import ChaturangaRAG
from knowledge.llm_client import (
    ExtractiveLLMClient, TransformersLLMClient, OllamaLLMClient, AnthropicLLMClient, MockLLMClient,
)

SQUARE_SIZE = 80
BOARD_PIXELS = SQUARE_SIZE * BOARD_SIZE
LIGHT_COLOR = "#EEEED2"
DARK_COLOR = "#769656"
SELECT_COLOR = "#F6F669"
MOVE_HINT_COLOR = "#7FA650"
FILES = "abcdefgh"

# Reuse the same glyph mapping as the vendored visualiser (Mantri->Queen
# glyph, Gaja->Bishop glyph -- rendering only, doesn't affect rules).
GLYPHS = {
    "K": "\u2654", "M": "\u2655", "R": "\u2656", "G": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "m": "\u265B", "r": "\u265C", "g": "\u265D", "n": "\u265E", "p": "\u265F",
}


class ChaturangaGUI:
    def __init__(self, root, human_color="w", agent_move_fn=None, verbose=False, rag_client=None,
                 engine_label="the game engine"):
        self.root = root
        self.root.title("Chaturanga")
        self.human_color = human_color
        self.agent_move_fn = agent_move_fn or minimax_agent_move
        self.verbose = verbose
        self.engine_label = engine_label
        # Defaults to ExtractiveLLMClient: zero setup, always works -- see
        # knowledge/llm_client.py. Pass a different ChaturangaRAG (e.g.
        # built with TransformersLLMClient or OllamaLLMClient) for real
        # generated answers instead of retrieved rule text.
        self.rag = rag_client or ChaturangaRAG(llm_client=ExtractiveLLMClient())

        self.state = ChaturangaState()
        self.selected_square = None       # (row, col) or None
        self.legal_destinations = []      # list of (row, col) for the selected piece

        self.status_var = tk.StringVar()

        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        board_frame = tk.Frame(main_frame)
        board_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(board_frame, width=BOARD_PIXELS, height=BOARD_PIXELS)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.piece_font = self._pick_font(root, ["Segoe UI Symbol", "DejaVu Sans", "Arial Unicode MS"], 40)

        status_bar = tk.Label(board_frame, textvariable=self.status_var, font=("Segoe UI", 11))
        status_bar.pack(pady=8)

        new_game_btn = tk.Button(board_frame, text="New Game", command=self.new_game)
        new_game_btn.pack()

        side_panel = tk.Frame(main_frame, width=320)
        side_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        side_panel.pack_propagate(False)  # keep the fixed width even as content changes

        self._build_rag_panel(side_panel)

        self.draw_board()
        self.update_status()
        self._maybe_agent_move()

    def _build_rag_panel(self, parent):
        """Ask-a-question panel, backed by knowledge/rag.py. Uses the
        LIVE game state as context (unlike knowledge/ask.py's standalone
        CLI, which always uses a fresh starting position) -- so a question
        like 'can my king castle here' gets answered against the actual
        current board, not a generic one. Lives in a side column next to
        the board rather than stacked below it, so the window stays a
        reasonable height regardless of how long the answer log gets."""
        tk.Label(parent, text="Ask about rules or strategy:", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(0, 6)
        )

        input_row = tk.Frame(parent)
        input_row.pack(fill=tk.X, pady=(0, 6))

        self.question_entry = tk.Entry(input_row, font=("Segoe UI", 10))
        self.question_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.question_entry.bind("<Return>", lambda event: self.ask_question())

        self.ask_button = tk.Button(input_row, text="Ask", command=self.ask_question)
        self.ask_button.pack(side=tk.LEFT, padx=(6, 0))

        self.answer_box = ScrolledText(parent, wrap=tk.WORD, font=("Segoe UI", 9), state=tk.DISABLED)
        self.answer_box.pack(fill=tk.BOTH, expand=True)

    def ask_question(self):
        query = self.question_entry.get().strip()
        if not query:
            return

        self.question_entry.delete(0, tk.END)
        self.ask_button.config(state=tk.DISABLED)
        self._append_to_answer_box(f"Q: {query}\n")
        self._append_to_answer_box("(thinking...)\n")
        self.root.update()  # show "thinking..." before the (possibly slow) call below

        try:
            answer = self.rag.answer(
                query, state=self.state,
                move_fn=lambda s: self.agent_move_fn(s, verbose=False),
                engine_label=self.engine_label,
            )
        except ConnectionError as e:
            answer = f"[Error] {e}"
        except Exception as e:
            answer = f"[Error] {type(e).__name__}: {e}"

        self._remove_last_line_from_answer_box()  # replace "(thinking...)" with the real answer
        self._append_to_answer_box(f"{answer}\n\n")
        self.ask_button.config(state=tk.NORMAL)

    def _append_to_answer_box(self, text: str):
        self.answer_box.config(state=tk.NORMAL)
        self.answer_box.insert(tk.END, text)
        self.answer_box.see(tk.END)
        self.answer_box.config(state=tk.DISABLED)

    def _remove_last_line_from_answer_box(self):
        self.answer_box.config(state=tk.NORMAL)
        self.answer_box.delete("end-2l", "end-1l")
        self.answer_box.config(state=tk.DISABLED)

    @staticmethod
    def _pick_font(root, preferred, size):
        """Return the first available font family from `preferred`, else fall
        back to the platform default (still renders Unicode, just plainer)."""
        available = set(tkfont.families(root))
        for family in preferred:
            if family in available:
                return (family, size)
        return (tkfont.nametofont("TkDefaultFont").actual("family"), size)

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def draw_board(self):
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x0, y0, x1, y1 = self._square_bbox(r, c)
                color = self._square_color(r, c)
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

                piece = self.state.board[r][c]
                if piece:
                    color_code, kind = piece[0], piece[1]
                    glyph = GLYPHS[kind if color_code == "w" else kind.lower()]
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    # Unicode chess glyphs are already visually distinct:
                    # white pieces are hollow outlines, black pieces are
                    # solid silhouettes. A single dark fill renders both
                    # correctly without needing a fake outline effect.
                    self.canvas.create_text(cx, cy, text=glyph, font=self.piece_font, fill="black")

        # file/rank labels
        for c in range(BOARD_SIZE):
            x0, y0, x1, y1 = self._square_bbox(0, c)
            self.canvas.create_text(x1 - 8, y1 - 8, text=FILES[c], font=("Segoe UI", 9), fill="gray30")
        for r in range(BOARD_SIZE):
            x0, y0, x1, y1 = self._square_bbox(r, 0)
            self.canvas.create_text(x0 + 8, y0 + 8, text=str(r + 1), font=("Segoe UI", 9), fill="gray30")

    def _square_bbox(self, row, col):
        # row 0 (rank 1) drawn at the bottom, matching a normal board view
        x0 = col * SQUARE_SIZE
        y0 = (BOARD_SIZE - 1 - row) * SQUARE_SIZE
        return x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE

    def _square_color(self, row, col):
        if self.selected_square == (row, col):
            return SELECT_COLOR
        if (row, col) in self.legal_destinations:
            return MOVE_HINT_COLOR
        return LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR

    def _pixel_to_square(self, x, y):
        col = x // SQUARE_SIZE
        row = BOARD_SIZE - 1 - (y // SQUARE_SIZE)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return int(row), int(col)
        return None

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def on_click(self, event):
        if self.state.turn != self.human_color or self.state.is_terminal():
            return

        square = self._pixel_to_square(event.x, event.y)
        if square is None:
            return
        row, col = square

        if self.selected_square is None:
            piece = self.state.board[row][col]
            if piece and piece[0] == self.human_color:
                self.selected_square = (row, col)
                self.legal_destinations = [
                    (m[2], m[3]) for m in self.state.legal_moves()
                    if (m[0], m[1]) == (row, col)
                ]
                self.draw_board()
            return

        # a piece is already selected
        if (row, col) == self.selected_square:
            self.selected_square = None
            self.legal_destinations = []
            self.draw_board()
            return

        move = self._find_move(self.selected_square, (row, col))
        if move:
            self.state.push(move)
            self.selected_square = None
            self.legal_destinations = []
            self.draw_board()
            self.update_status()
            self._maybe_agent_move()
        else:
            # clicked an unrelated square -- reselect if it's our own piece
            piece = self.state.board[row][col]
            if piece and piece[0] == self.human_color:
                self.selected_square = (row, col)
                self.legal_destinations = [
                    (m[2], m[3]) for m in self.state.legal_moves()
                    if (m[0], m[1]) == (row, col)
                ]
            else:
                self.selected_square = None
                self.legal_destinations = []
            self.draw_board()

    def _find_move(self, from_sq, to_sq):
        for move in self.state.legal_moves():
            if (move[0], move[1]) == from_sq and (move[2], move[3]) == to_sq:
                return move
        return None

    def _maybe_agent_move(self):
        if self.state.is_terminal():
            self.update_status()
            return
        if self.state.turn != self.human_color:
            self.root.after(300, self._play_agent_move)

    def _play_agent_move(self):
        if self.verbose:
            turn_name = "White" if self.state.turn == "w" else "Black"
            print(f"\n--- {turn_name} (agent) to move ---")

        move = self.agent_move_fn(self.state, verbose=self.verbose)
        self.state.push(move)
        self.draw_board()
        self.update_status()
        if not self.state.is_terminal() and self.state.turn != self.human_color:
            self.root.after(300, self._play_agent_move)

    @staticmethod
    def _random_agent_move(state):
        return random.choice(state.legal_moves())

    # ------------------------------------------------------------------ #
    # Status / lifecycle
    # ------------------------------------------------------------------ #
    def update_status(self):
        if self.state.is_terminal():
            reason = self.state.end_reason()
            result = self.state.result()
            if result == 0.0:
                message = f"Game over -- draw by {reason}."
            else:
                winner = "White" if result > 0 else "Black"
                message = f"Game over -- {winner} wins by {reason}!"
            self.status_var.set(message)
            messagebox.showinfo("Game Over", message)
            return

        turn_name = "White" if self.state.turn == "w" else "Black"
        check_note = "  (in check)" if self.state.is_check() else ""
        self.status_var.set(f"{turn_name} to move{check_note}")

    def new_game(self):
        self.state = ChaturangaState()
        self.selected_square = None
        self.legal_destinations = []
        self.draw_board()
        self.update_status()
        self._maybe_agent_move()


def main():
    parser = argparse.ArgumentParser(description="Chaturanga GUI.")
    parser.add_argument("--agent", choices=["mcts", "minimax_ppo", "rl", "cnn", "material"], default="mcts",
                         help="mcts (default): PRIMARY engine -- MCTS guided by a trained PPO policy's "
                              "move priors + value estimate. minimax_ppo: alpha-beta using PPO's value "
                              "at leaf nodes. rl: bare PPO policy, no search. cnn/material: pre-RL "
                              "comparison baselines -- see README.md.")
    parser.add_argument("--simulations", type=int, default=DEFAULT_MCTS_SIMULATIONS,
                         help=f"MCTS simulation count (default: {DEFAULT_MCTS_SIMULATIONS}), only used with --agent mcts")
    parser.add_argument("--depth", type=int, default=None,
                         help=f"search depth (default: {DEFAULT_PPO_EVAL_DEPTH} for minimax_ppo, "
                              f"{DEFAULT_CNN_DEPTH} for cnn, {DEFAULT_DEPTH} for material). "
                              "Not used with --agent mcts (uses --simulations) or rl (no search).")
    parser.add_argument("--model-path", type=str, default=DEFAULT_PPO_PATH,
                         help=f"path to the trained PPO checkpoint (default: {DEFAULT_PPO_PATH}), "
                              "used with --agent mcts/minimax_ppo/rl")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH,
                         help=f"path to the bootstrap CNN checkpoint (default: {DEFAULT_CHECKPOINT_PATH}), only used with --agent cnn")
    parser.add_argument("--agent-color", choices=["w", "b"], default="w",
                         help="which color the loaded PPO checkpoint was trained to play (default: w). "
                              "Used with --agent mcts/minimax_ppo/rl to correctly interpret its value/policy output.")
    parser.add_argument("--verbose", action="store_true",
                         help="print the agent's reasoning to the terminal before each move")
    parser.add_argument("--rag-backend", choices=["extractive", "transformers", "ollama", "anthropic", "mock"],
                         default="extractive",
                         help="backend for the in-GUI 'ask about rules/strategy' panel (default: extractive -- "
                              "no setup needed at all; see knowledge/llm_client.py for the other options)")
    parser.add_argument("--rag-model", type=str, default=None,
                         help="model name override for --rag-backend (e.g. an HF repo id for transformers, "
                              "a model name for ollama/anthropic)")
    args = parser.parse_args()

    ENGINE_LABELS = {"mcts": "MCTS guided by the PPO network", "minimax_ppo": "Minimax guided by the PPO network",
                     "rl": "the PPO policy", "cnn": "Minimax with the CNN evaluator", "material": "Minimax (material evaluation)"}
    engine_label = ENGINE_LABELS.get(args.agent, args.agent)

    if args.agent == "mcts":
        agent_move_fn = lambda state, verbose=False: mcts_agent_move(
            state, n_simulations=args.simulations, model_path=args.model_path,
            agent_color=args.agent_color, verbose=verbose
        )
    elif args.agent == "minimax_ppo":
        depth = args.depth if args.depth is not None else DEFAULT_PPO_EVAL_DEPTH
        agent_move_fn = lambda state, verbose=False: minimax_ppo_agent_move(
            state, depth=depth, model_path=args.model_path, agent_color=args.agent_color, verbose=verbose
        )
    elif args.agent == "rl":
        agent_move_fn = lambda state, verbose=False: rl_agent_move(
            state, model_path=args.model_path, verbose=verbose
        )
    elif args.agent == "cnn":
        depth = args.depth if args.depth is not None else DEFAULT_CNN_DEPTH
        agent_move_fn = lambda state, verbose=False: cnn_agent_move(
            state, depth=depth, checkpoint_path=args.checkpoint, verbose=verbose
        )
    else:
        depth = args.depth if args.depth is not None else DEFAULT_DEPTH
        agent_move_fn = lambda state, verbose=False: minimax_agent_move(state, depth=depth, verbose=verbose)

    try:
        if args.rag_backend == "extractive":
            rag_llm_client = ExtractiveLLMClient()
        elif args.rag_backend == "transformers":
            kwargs = {"model_name": args.rag_model} if args.rag_model else {}
            print("Loading local RAG model (first run downloads it -- this may take a while)...")
            rag_llm_client = TransformersLLMClient(**kwargs)
        elif args.rag_backend == "ollama":
            kwargs = {"model": args.rag_model} if args.rag_model else {}
            rag_llm_client = OllamaLLMClient(**kwargs)
        elif args.rag_backend == "anthropic":
            kwargs = {"model": args.rag_model} if args.rag_model else {}
            rag_llm_client = AnthropicLLMClient(**kwargs)
        else:
            rag_llm_client = MockLLMClient()
    except Exception as e:
        print(f"[Warning] Could not set up RAG backend '{args.rag_backend}': {type(e).__name__}: {e}")
        print("Falling back to --rag-backend extractive for the ask panel.")
        rag_llm_client = ExtractiveLLMClient()

    rag_client = ChaturangaRAG(llm_client=rag_llm_client)

    root = tk.Tk()
    ChaturangaGUI(root, human_color="w", agent_move_fn=agent_move_fn, verbose=args.verbose, rag_client=rag_client,
               engine_label=engine_label)
    root.mainloop()


if __name__ == "__main__":
    main()