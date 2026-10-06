"""Hybrid pipelines over the chunk and graph indexes.

FusionHybrid merges the two rankings. RoutedHybrid sends each question to one of them.
"""
import re

COMPARISON_WORDS = re.compile(r"\b(or|both|same|in common)\b")
YES_NO_PAIR = re.compile(r"^(are|were|is|was|do|does|did|have|has|had)\b.* and ")


def is_comparison(question):
    """True for questions that weigh two or more entities against each other.

    Written from the HotpotQA results (graph wins comparison, chunk wins bridge)
    and frozen before any 2WikiMultiHopQA question was answered.
    """
    q = question.lower()
    return bool(COMPARISON_WORDS.search(q) or YES_NO_PAIR.search(q))


class RoutedHybrid:
    """Graph for comparison-style questions, chunk for everything else."""

    def __init__(self, chunk_index, graph_index):
        self.chunk = chunk_index
        self.graph = graph_index
        self.build_seconds = chunk_index.build_seconds + graph_index.build_seconds
        self.build_llm_tokens = chunk_index.build_llm_tokens + graph_index.build_llm_tokens

    def retrieve(self, question, k):
        return (self.graph if is_comparison(question) else self.chunk).retrieve(question, k)


class FusionHybrid:
    """score(p) = sum over rankings of 1 / (rrf_k + rank). A paragraph the graph walk
    never reaches gets nothing from the graph side, so it can still win on chunk rank alone.

    rrf_k = 60 is the constant from Cormack et al. (2009), not tuned on these questions.
    Both rankings get equal weight.
    """

    def __init__(self, chunk_index, graph_index, rrf_k):
        self.chunk = chunk_index
        self.graph = graph_index
        self.rrf_k = rrf_k
        self.embedder = chunk_index.embedder
        # Needs both indexes, so it pays for both.
        self.build_seconds = chunk_index.build_seconds + graph_index.build_seconds
        self.build_llm_tokens = chunk_index.build_llm_tokens + graph_index.build_llm_tokens

    def retrieve(self, question, k):
        qvec = self.embedder.encode([question], memo=False)[0]
        score, by_id = {}, {}
        for ranking in (self.chunk.ranked(qvec), self.graph.ranked(question, qvec)):
            for rank, p in enumerate(ranking, start=1):
                score[p["id"]] = score.get(p["id"], 0.0) + 1 / (self.rrf_k + rank)
                by_id[p["id"]] = p
        # Ties (rare) fall back to dict order, which is chunk order.
        top = sorted(score, key=lambda pid: -score[pid])[:k]
        return [by_id[pid] for pid in top]
