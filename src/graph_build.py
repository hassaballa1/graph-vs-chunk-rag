"""Graph pipeline, index side: LLM triple extraction, entity merging, NetworkX graph."""
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np

from src.common import CACHE, paragraph_text, sha
from src.generate import call_llm

EXTRACT_PROMPT = """Extract the named entities and the facts that connect them from the paragraph.

Return JSON with exactly this shape:
{{"entities": ["...", "..."],
  "triples": [{{"subject": "...", "relation": "...", "object": "..."}}]}}

Rules:
- Entities are specific people, places, organisations, works, events, dates and numbers.
- Always include the paragraph's title as an entity.
- Use the full name as written in the paragraph.
- relation is a short verb phrase, e.g. "directed", "born in", "member of".
- Every subject and object must also appear in entities.

Paragraph:
{paragraph}"""


def norm(name):
    return " ".join(str(name).lower().strip().strip(".,;:'\"").split())


def parse_extraction(text):
    """Tolerant JSON parse; small models occasionally return malformed output."""
    # Keep the outermost {...}; hosted models sometimes wrap JSON in a code fence.
    text = text[text.find("{"): text.rfind("}") + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [], []
    if not isinstance(data, dict):
        return [], []
    entities = [e for e in data.get("entities", []) if isinstance(e, str) and e.strip()]
    triples = []
    for t in data.get("triples", []):
        if isinstance(t, dict) and all(isinstance(t.get(x), str) for x in ("subject", "relation", "object")):
            if t["subject"].strip() and t["object"].strip():
                triples.append((t["subject"], t["relation"], t["object"]))
    return entities, triples


def extract_all(corpus, cfg):
    """One LLM call per paragraph, run in parallel.

    Returns per-paragraph results, total tokens, and the summed duration of the
    calls as originally made (stored in the cache, so it survives cached re-runs).
    """

    def one(p):
        out = call_llm(EXTRACT_PROMPT.format(paragraph=paragraph_text(p)), cfg, json_mode=True)
        entities, triples = parse_extraction(out["text"])
        return p["id"], entities, triples, out

    results, tokens, call_seconds, done = {}, 0, 0.0, 0
    with ThreadPoolExecutor(cfg["llm_workers"]) as pool:
        for pid, entities, triples, out in pool.map(one, corpus):
            results[pid] = (entities, triples)
            tokens += out["prompt_tokens"] + out["output_tokens"]
            call_seconds += out["seconds"]
            done += 1
            if done % 25 == 0 or done == len(corpus):
                print(f"  extracted {done}/{len(corpus)} paragraphs", flush=True)
    return results, tokens, call_seconds


def merge_entities(names, embedder, threshold):
    """Map each normalised name to a canonical one.

    Step 1 (done by the caller): lowercase and strip. Step 2: names whose
    embeddings have cosine >= threshold are merged into the most frequent one.
    """
    names = sorted(names)
    if not names:
        return {}
    vecs = embedder.encode(names)
    sims = vecs @ vecs.T
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    rows, cols = np.where(np.triu(sims, 1) >= threshold)
    for i, j in zip(rows, cols):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(names)):
        groups.setdefault(find(i), []).append(names[i])
    # Canonical name: the shortest member, a stable and readable choice.
    return {n: min(g, key=lambda s: (len(s), s)) for g in groups.values() for n in g}


def build_graph(corpus, embedder, cfg):
    key = sha(json.dumps([EXTRACT_PROMPT, cfg["llm_model"], cfg["merge_threshold"], sorted(p["id"] for p in corpus)]))
    path = CACHE / f"graph-{key[:12]}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)

    extractions, tokens, call_seconds = extract_all(corpus, cfg)
    start = time.time()

    all_names = set()
    for pid, (entities, triples) in extractions.items():
        all_names.update(norm(e) for e in entities)
        for s, _, o in triples:
            all_names.update((norm(s), norm(o)))
        all_names.add(norm(pid))  # the title is always an entity
    all_names.discard("")
    canon = merge_entities(all_names, embedder, cfg["merge_threshold"])

    g = nx.Graph()
    for pid, (entities, triples) in extractions.items():
        for e in set(entities) | {pid}:
            c = canon.get(norm(e))
            if c:
                g.add_node(c)
                g.nodes[c].setdefault("paragraphs", set()).add(pid)
        for s, rel, o in triples:
            cs, co = canon.get(norm(s)), canon.get(norm(o))
            if not cs or not co or cs == co:
                continue
            for c in (cs, co):
                g.add_node(c)
                g.nodes[c].setdefault("paragraphs", set()).add(pid)
            if g.has_edge(cs, co):
                g.edges[cs, co]["paragraphs"].add(pid)
                g.edges[cs, co]["relations"].add(rel)
            else:
                g.add_edge(cs, co, paragraphs={pid}, relations={rel})

    stats = {
        # Extraction wall time is estimated as summed call time over the parallel
        # workers, so a run from the cache reports the same cost as the original.
        "build_seconds": call_seconds / cfg["llm_workers"] + (time.time() - start),
        "build_llm_tokens": tokens,
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "merged_names": len(all_names) - len(set(canon.values())),
        "empty_extractions": sum(1 for e, t in extractions.values() if not e and not t),
    }
    with open(path, "wb") as f:
        pickle.dump((g, stats), f)
    return g, stats
