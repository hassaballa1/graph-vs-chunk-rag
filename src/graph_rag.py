"""Graph pipeline, query side: seed entities, hop expansion, ranking."""
import copy
import re
import time

import numpy as np

from src.graph_build import build_graph, norm


class GraphIndex:
    version = "v1"

    def __init__(self, corpus, embedder, cfg, chunk_index, hub_cap=None):
        # hub_cap: nodes attached to more paragraphs than this ("united states",
        # "2012") are ignored at query time. None keeps every node.
        self.hub_cap = hub_cap
        self.cfg = cfg
        self.embedder = embedder
        self.by_id = {p["id"]: p for p in corpus}
        # Similarity of question to paragraph reuses the chunk pipeline's vectors
        # (same embedding model), only as a tie-breaker inside a hop level.
        self.chunk_index = chunk_index
        self.pos = {p["id"]: i for i, p in enumerate(corpus)}
        # stats["build_seconds"] covers extraction and merging, even when the
        # graph comes from disk; only the node embedding below is timed here.
        self.graph, self.stats = build_graph(corpus, embedder, cfg, self.version)
        start = time.time()
        self.nodes = list(self.graph.nodes)
        self.node_vecs = embedder.encode(self.nodes) if self.nodes else np.zeros((0, 1))
        self.build_seconds = self.stats["build_seconds"] + (time.time() - start)
        self.build_llm_tokens = self.stats["build_llm_tokens"]

    def seeds(self, question, qvec):
        """Entity nodes closest to the question by embedding, plus any whose name the question contains."""
        sims = self.node_vecs @ qvec
        chosen = [self.nodes[i] for i in np.argsort(-sims)[: self.cfg["n_seeds"]]]
        qn = f" {norm(question)} "
        chosen += [n for n in self.nodes if len(n) >= 4 and f" {n} " in qn]
        return list(dict.fromkeys(chosen))

    def pruned(self, hub_cap):
        """The same index (same graph, same build cost) with hub pruning switched on."""
        other = copy.copy(self)
        other.hub_cap = hub_cap
        return other

    def is_hub(self, n):
        return self.hub_cap is not None and len(self.graph.nodes[n].get("paragraphs", ())) > self.hub_cap

    def retrieve(self, question, k):
        qvec = self.embedder.encode([question], memo=False)[0]
        return self.ranked(question, qvec)[:k]

    def ranked(self, question, qvec):
        """Every paragraph the walk reaches, best first. Unreached paragraphs are left out."""
        pdist = self.paragraph_hops(question, qvec)
        sims = self.chunk_index.scores(qvec)
        ranked = sorted(pdist, key=lambda pid: (pdist[pid], -sims[self.pos[pid]]))
        return [self.by_id[pid] for pid in ranked]

    def paragraph_hops(self, question, qvec):
        """Hop distance from the seeds to every paragraph the walk reaches."""
        seeds = [s for s in self.seeds(question, qvec) if not self.is_hub(s)]
        # BFS from all seeds at once: hop distance of every node within max_hops.
        # Hubs are never entered, so they neither spread the walk nor add paragraphs.
        dist = {s: 0 for s in seeds}
        frontier = list(seeds)
        for hop in range(1, self.cfg["max_hops"] + 1):
            nxt = []
            for n in frontier:
                for m in self.graph.neighbors(n):
                    if m not in dist and not self.is_hub(m):
                        dist[m] = hop
                        nxt.append(m)
            frontier = nxt

        # A paragraph's distance is the smallest distance of any node or edge
        # it is attached to. An edge counts at the distance of its nearer end.
        pdist = {}
        for n, d in dist.items():
            for pid in self.graph.nodes[n].get("paragraphs", ()):
                pdist[pid] = min(pdist.get(pid, d), d)
            for m in self.graph.neighbors(n):
                if m in dist:
                    d_edge = min(d, dist[m])
                    for pid in self.graph.edges[n, m]["paragraphs"]:
                        pdist[pid] = min(pdist.get(pid, d_edge), d_edge)
        return pdist


def words(text):
    """Lowercase word sequence with punctuation dropped, padded for whole-word matching."""
    return " " + " ".join(re.findall(r"\w+", text.lower())) + " "


class GraphIndexV2(GraphIndex):
    """The three fixes from the HotpotQA failure analysis, set before the 2Wiki run:

    1. Names merged after stripping disambiguation (graph built with norm_v2).
    2. Seeds are the entities the question names, longest match first; embedding seeds
       only when it names none. Hubs (> hub_cap paragraphs) are never seeds.
    3. Paragraphs ranked by similarity * hop_decay ** hop, not by hop first.
    """

    version = "v2"

    def __init__(self, corpus, embedder, cfg, chunk_index):
        super().__init__(corpus, embedder, cfg, chunk_index)
        self.seed_cap = cfg["hub_cap"]
        self.node_words = [(n, words(n)) for n in self.nodes if len(n) >= 4]

    def seeds(self, question, qvec):
        qw = words(question)
        hits = [(n, w) for n, w in self.node_words
                if w in qw and len(self.graph.nodes[n].get("paragraphs", ())) <= self.seed_cap]
        # Keep only maximal mentions: "war" is dropped when "polish russian war" matched.
        chosen = [n for n, w in hits if not any(w != w2 and w in w2 for _, w2 in hits)]
        if chosen:
            return chosen
        sims = self.node_vecs @ qvec
        return [self.nodes[i] for i in np.argsort(-sims)[: self.cfg["n_seeds"]]]

    def ranked(self, question, qvec):
        pdist = self.paragraph_hops(question, qvec)
        sims = self.chunk_index.scores(qvec)
        decay = self.cfg["hop_decay"]
        ranked = sorted(pdist, key=lambda pid: -sims[self.pos[pid]] * decay ** pdist[pid])
        return [self.by_id[pid] for pid in ranked]
