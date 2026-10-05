"""Config, data loading, embeddings and the shared context budget."""
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"


def load_config():
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def load_jsonl(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if "_meta" not in r]


def load_data(limit=None):
    """Questions, plus the corpus pooled over exactly those questions."""
    questions = load_jsonl(ROOT / "data" / "questions.jsonl")
    if limit:
        questions = questions[:limit]
    wanted = {t for q in questions for t in q["context_titles"]}
    corpus = [p for p in load_jsonl(ROOT / "data" / "corpus.jsonl") if p["title"] in wanted]
    return questions, corpus


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def count_tokens(text):
    # Rough, model-agnostic estimate (~0.75 words per token). Applied identically
    # to both pipelines, which is all the budget needs.
    return int(len(text.split()) / 0.75) + 1


def paragraph_text(p):
    return f"{p['title']}: {p['text']}"


def fit_budget(paragraphs, k, max_tokens):
    """Keep paragraphs in rank order until k are taken or the token cap is hit."""
    kept, used = [], 0
    for p in paragraphs[:k]:
        n = count_tokens(paragraph_text(p))
        if kept and used + n > max_tokens:
            break
        kept.append(p)
        used += n
    return kept


class Embedder:
    """sentence-transformers model with an in-memory cache for the current run.

    Nothing is cached on disk, so index-build times are real embedding times.
    Questions are encoded fresh (memo=False) so query latency is honest too.
    """

    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.memo = {}

    def encode(self, texts, memo=True):
        """Unit-normalised vectors, shape (len(texts), dim)."""
        if not memo:
            return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)
        missing = list(dict.fromkeys(t for t in texts if t not in self.memo))
        if missing:
            vecs = self.model.encode(
                missing, batch_size=64, normalize_embeddings=True, show_progress_bar=len(missing) > 200
            )
            self.memo.update(zip(missing, vecs))
        return np.vstack([self.memo[t] for t in texts]).astype(np.float32)
