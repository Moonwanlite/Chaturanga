"""
Lightweight local retrieval index over knowledge/rules_corpus/*.md.

Uses TF-IDF + cosine similarity (scikit-learn) rather than a hosted
embedding API: the corpus is tiny (a handful of short documents), so a
classical sparse retrieval method is more than sufficient, requires no API
key, and is instant to rebuild. If you outgrow this (a much larger corpus,
or a need for semantic/paraphrase matching TF-IDF can't catch), swap this
module for an embedding-based index -- rag.py only depends on the
RulesIndex.search(query, top_k) interface below, so nothing else needs to
change.

Documents are chunked by paragraph (blank-line-separated) rather than
indexed as whole files, so retrieval can return a specific relevant
paragraph instead of an entire multi-topic document.
"""

import glob
import os
import pickle
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_WORD_RE = re.compile(r"[a-zA-Z]{2,}")


def _stemmed_analyzer(text: str):
    """Lowercase, tokenize, drop stopwords, and stem -- so 'move', 'moves',
    and 'moving' all collapse to the same token ('move'). Plain word-level
    TF-IDF without this collapsed differently-inflected forms of the same
    word into separate features, which measurably hurt retrieval quality
    in testing (e.g. a query for 'how does the elephant move' failed to
    retrieve the Gaja section, which says 'moves' rather than 'move')."""
    words = _WORD_RE.findall(text.lower())
    return [_STEMMER.stem(w) for w in words if w not in ENGLISH_STOP_WORDS]

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "rules_corpus")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "index.pkl")


def _load_chunks(corpus_dir: str = CORPUS_DIR):
    """Returns a list of {'text': str, 'source': str} chunks, one per
    paragraph, skipping markdown headers on their own (kept attached to
    the paragraph that follows them for context)."""
    chunks = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        source = os.path.basename(path)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            chunks.append({"text": para, "source": source})
    return chunks


class RulesIndex:
    def __init__(self, chunks, vectorizer, matrix):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def build(cls, corpus_dir: str = CORPUS_DIR):
        chunks = _load_chunks(corpus_dir)
        texts = [c["text"] for c in chunks]
        vectorizer = TfidfVectorizer(analyzer=_stemmed_analyzer)
        matrix = vectorizer.fit_transform(texts)
        return cls(chunks, vectorizer, matrix)

    def search(self, query: str, top_k: int = 3):
        """Returns up to top_k {'text', 'source', 'score'} dicts, best first."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            results.append({
                "text": self.chunks[i]["text"],
                "source": self.chunks[i]["source"],
                "score": float(scores[i]),
            })
        return results

    def save(self, path: str = INDEX_PATH):
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path: str = INDEX_PATH):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], data["vectorizer"], data["matrix"])


def main():
    index = RulesIndex.build()
    index.save()
    print(f"Indexed {len(index.chunks)} chunks from {CORPUS_DIR}")
    print(f"Saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
