"""
Minimal LLM client for the RAG layer. Four options, all exposing the same
.complete(prompt) -> str method, so rag.py works unchanged with any of them:

    OllamaLLMClient        Local server (https://ollama.com). Best quality
                            for its size, but requires a SEPARATE
                            application installed and running, plus a
                            manual `ollama pull <model>` step -- real
                            friction if you're handing this project to
                            someone else (e.g. submitting for grading) who
                            won't want to install and run another program.

    TransformersLLMClient   A small open-source model (default:
                            HuggingFaceTB/SmolLM2-360M-Instruct) run
                            in-process via the `transformers` Python
                            library -- just `pip install transformers`,
                            same as every other dependency in this
                            project. The model weights download
                            automatically on first use (one-time,
                            needs internet), but there's no separate
                            application to install or keep running, and
                            no manual "pull a model" CLI step.

    AnthropicLLMClient      Hosted, needs an ANTHROPIC_API_KEY.

    MockLLMClient           No server, key, model download, or network
                            access at all -- a fixed placeholder string,
                            for testing retrieval + prompt construction
                            in isolation.

    ExtractiveLLMClient (in this file too) is the true zero-dependency
    fallback: no LLM at all, just the retrieved rule text formatted
    directly. See its docstring below.

For submitting/sharing this project with someone who shouldn't need to
install anything extra: use ExtractiveLLMClient (guaranteed to work,
no download) or TransformersLLMClient (real generated answers, only
needs `pip install transformers` + one-time automatic model download).
"""

import os
import requests

OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "llama3.2"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"

TRANSFORMERS_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


class TransformersLLMClient:
    """
    Runs a small open-source model in-process via HuggingFace `transformers`
    -- no separate application to install, no server to keep running.
    `pip install transformers` is the only setup step; the model weights
    download automatically the first time this class is instantiated
    (needs internet once, then works offline).

    Default model (SmolLM2-360M-Instruct, ~360M parameters) is small enough
    to run reasonably on CPU alone -- no GPU required, though it'll use one
    automatically if available.

    HONEST QUALITY NOTE, based on real testing (not just theoretical): at
    360M parameters, this model sometimes still gets facts wrong or invents
    plausible-sounding but incorrect answers even when retrieval found the
    right context -- it's a genuine generative LLM, not a placeholder, but
    it is a noticeably weaker one than something like Ollama's llama3.2
    (~10x larger). repetition_penalty/no_repeat_ngram_size below fix a
    specific observed failure mode (getting stuck in an infinite repeated
    phrase), but don't fix the model's limited reasoning capacity.

    If answer quality matters more than download size/speed, pass a larger
    instruction-tuned model instead, e.g.:
        TransformersLLMClient(model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct")
        TransformersLLMClient(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    both still install via plain `pip install transformers` with no other
    setup, just a larger one-time download and slower generation.

    For guaranteed-correct rule text with no hallucination risk at all
    (at the cost of not being a natural-language generated answer), use
    ExtractiveLLMClient instead.
    """

    def __init__(self, model_name: str = TRANSFORMERS_DEFAULT_MODEL, max_new_tokens: int = 300, device: int = None):
        from transformers import pipeline
        import torch

        if device is None:
            device = 0 if torch.cuda.is_available() else -1  # -1 = CPU in HF's pipeline convention

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = pipeline("text-generation", model=model_name, device=device)

    def complete(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = self._pipe.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fall back to the raw prompt if this model/tokenizer has no chat template.
            formatted_prompt = prompt

        outputs = self._pipe(
            formatted_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_full_text=False,
            # Fixes a real failure mode observed in testing: greedy decoding
            # (do_sample=False) on a small model can get stuck repeating the
            # same phrase indefinitely (e.g. "To respond to the question, I
            # would say:" x25). repetition_penalty discourages reusing
            # recent tokens; no_repeat_ngram_size hard-bans repeating any
            # exact 3-token sequence. Both are standard, well-established
            # fixes for this exact symptom, not a workaround specific to
            # this project.
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        return outputs[0]["generated_text"].strip()


class ExtractiveLLMClient:
    """
    The true zero-dependency fallback: no LLM, no model download, no
    server, no network access at all beyond what retrieval itself already
    needed (scikit-learn/NLTK, already required for knowledge/build_index.py).

    Instead of generating a natural-language answer, this returns the
    retrieved rule text directly, clearly labeled as such. It works by
    parsing the "Relevant rules/strategy context:" section back out of the
    prompt that knowledge.rag.build_prompt() constructs -- a bit of a
    layering hack (in a cleaner design ChaturangaRAG would hand chunks to
    the client directly instead of a single flattened string), but it
    means this class needs zero changes to rag.py or its interface.

    Use this when you need something that is GUARANTEED to work with no
    setup at all -- e.g. handing the project to someone (a professor, a
    grader) who shouldn't have to install anything.
    """

    _CONTEXT_START = "Relevant rules/strategy context:\n"
    _CONTEXT_END = "\n\nQuestion:"

    def complete(self, prompt: str) -> str:
        start = prompt.find(self._CONTEXT_START)
        end = prompt.find(self._CONTEXT_END)
        if start == -1 or end == -1:
            return "[ExtractiveLLMClient] Could not locate retrieved context in the prompt."

        context = prompt[start + len(self._CONTEXT_START):end].strip()
        if not context:
            return "[ExtractiveLLMClient] No relevant rules found for this question."

        return (
            "[ExtractiveLLMClient -- retrieved directly, not AI-generated]\n\n"
            f"{context}"
        )


class OllamaLLMClient:
    """Talks to a local Ollama server. Open-source, no API key, runs
    entirely on your own machine (CPU or GPU, whatever Ollama is using)."""

    def __init__(self, model: str = OLLAMA_DEFAULT_MODEL, host: str = OLLAMA_DEFAULT_HOST, max_tokens: int = 500):
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens

    def complete(self, prompt: str) -> str:
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": self.max_tokens},
                },
                timeout=120,  # local generation on CPU can be slow, especially for larger models
            )
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.host}. Make sure Ollama is installed and "
                f"running, and that you've pulled a model (e.g. `ollama pull {self.model}`). "
                "See https://ollama.com/download."
            ) from e

        response.raise_for_status()
        data = response.json()
        return data["response"]


class AnthropicLLMClient:
    """Hosted alternative -- needs an ANTHROPIC_API_KEY. Kept as an option
    in case you want a stronger model than what's practical to run locally,
    but OllamaLLMClient is the default for this project."""

    def __init__(self, model: str = ANTHROPIC_DEFAULT_MODEL, max_tokens: int = 500, api_key: str = None):
        self.model = model
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set the ANTHROPIC_API_KEY environment variable, "
                "or pass api_key= explicitly. For a fully open-source local option "
                "instead, use OllamaLLMClient."
            )

    def complete(self, prompt: str) -> str:
        response = requests.post(
            ANTHROPIC_API_URL,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return "".join(block["text"] for block in data["content"] if block["type"] == "text")


class MockLLMClient:
    """Stand-in that needs no server, key, or network access at all.
    Returns a fixed, clearly-labeled placeholder so you can test the
    retrieval + prompt-construction pipeline (rag.py) end-to-end before
    Ollama (or any real backend) is set up."""

    def complete(self, prompt: str) -> str:
        return (
            "[MockLLMClient placeholder response -- no real LLM was called]\n"
            f"(prompt was {len(prompt)} characters; "
            "swap in OllamaLLMClient for real local, open-source answers)"
        )