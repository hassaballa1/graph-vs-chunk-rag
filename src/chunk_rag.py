"""Chunk pipeline: one embedding per paragraph, cosine top-k."""
import time

import numpy as np

from src.common import paragraph_text


class ChunkIndex:
    def __init__(self, corpus, embedder):
        start = time.time()
        self.corpus = corpus
        self.embedder = embedder
        self.vectors = embedder.encode([paragraph_text(p) for p in corpus])
        self.build_seconds = time.time() - start
        self.build_llm_tokens = 0

    def scores(self, qvec):
        """Cosine similarity of a question vector to every paragraph (vectors are unit length)."""
        return self.vectors @ qvec

    def ranked(self, qvec):
        """Every paragraph, best first."""
        return [self.corpus[i] for i in np.argsort(-self.scores(qvec))]

    def retrieve(self, question, k):
        return self.ranked(self.embedder.encode([question], memo=False)[0])[:k]
