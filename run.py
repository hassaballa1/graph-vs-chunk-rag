"""One entry point: build both indexes, answer every question, evaluate.

    docker compose run --rm bench python run.py              # full benchmark
    docker compose run --rm bench python run.py --limit 20   # cost test on 20 questions
    docker compose run --rm bench python run.py --chunk-only # no LLM needed for retrieval numbers
    docker compose run --rm bench python run.py --dataset 2wiki  # 2WikiMultiHopQA -> results/2wiki/
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.chunk_rag import ChunkIndex
from src.common import ROOT, Embedder, fit_budget, load_config, load_data
from src.evaluate import results_markdown, score_row
from src.generate import answer, ensure_model
from src.plots import write_all

DATASET_NAMES = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMultiHopQA"}
# Output file per pipeline, kept stable so links to existing outputs don't break.
SLUGS = {"Graph (hub-pruned)": "graph_pruned", "Graph v2": "graph_v2", "Hybrid (fusion)": "hybrid", "Hybrid (routed)": "hybrid_routed"}


def run_pipeline(name, index, questions, cfg, with_answers):
    print(f"[{name}] retrieving", flush=True)
    rows, latencies = [], []
    for q in questions:
        start = time.perf_counter()
        ranked = index.retrieve(q["question"], cfg["k"])
        latencies.append(time.perf_counter() - start)
        context = fit_budget(ranked, cfg["k"], cfg["max_context_tokens"])
        rows.append({"id": q["id"], "retrieved": [p["title"] for p in context], "context": context})

    if with_answers:
        print(f"[{name}] answering {len(questions)} questions", flush=True)
        with ThreadPoolExecutor(cfg["llm_workers"]) as pool:
            preds = list(pool.map(lambda iq: answer(iq[1]["question"], rows[iq[0]]["context"], cfg),
                                  enumerate(questions)))
    else:
        preds = [""] * len(questions)

    for row, pred in zip(rows, preds):
        row["prediction"] = pred
        del row["context"]
    scores = [score_row(r, q) for r, q in zip(rows, questions)]
    return {
        "rows": rows,
        "scores": scores,
        "latencies": latencies,
        "index_tokens": index.build_llm_tokens,
        "index_seconds": index.build_seconds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="use only the first N frozen questions (and their corpus)")
    ap.add_argument("--chunk-only", action="store_true", help="chunk retrieval metrics only, no LLM")
    ap.add_argument("--dataset", choices=DATASET_NAMES, default="hotpotqa")
    args = ap.parse_args()

    cfg = load_config()
    dcfg = cfg["datasets"][args.dataset]
    cfg["dataset_name"] = DATASET_NAMES[args.dataset]
    if not (ROOT / dcfg["data_dir"] / "questions.jsonl").exists():
        if args.dataset == "2wiki":
            from data.build_2wiki import main as build_data
        else:
            from data.build_data import main as build_data
        build_data()
    questions, corpus = load_data(dcfg["data_dir"], args.limit)
    print(f"{len(questions)} questions, {len(corpus)} paragraphs in the pooled corpus", flush=True)

    embedder = Embedder(cfg["embedding_model"])
    chunk = ChunkIndex(corpus, embedder)
    with_answers = not args.chunk_only
    runs = {"Chunk": run_pipeline("chunk", chunk, questions, cfg, with_answers)}
    no_context = None

    if not args.chunk_only:
        ensure_model(cfg)
        from src.graph_rag import GraphIndex

        print("[graph] building index (one LLM call per paragraph, cached)", flush=True)
        graph = GraphIndex(corpus, embedder, cfg, chunk)
        print(f"[graph] {graph.stats}", flush=True)
        runs["Graph"] = run_pipeline("graph", graph, questions, cfg, True)
        from src.graph_rag import GraphIndexV2
        from src.hybrid_rag import FusionHybrid, RoutedHybrid

        graph_v2 = GraphIndexV2(corpus, embedder, cfg, chunk)
        print(f"[graph v2] {graph_v2.stats}", flush=True)
        runs["Graph v2"] = run_pipeline("graph v2", graph_v2, questions, cfg, True)

        runs["Hybrid (fusion)"] = run_pipeline(
            "hybrid, fusion", FusionHybrid(chunk, graph, cfg["rrf_k"]), questions, cfg, True)
        runs["Hybrid (routed)"] = run_pipeline(
            "hybrid, routed", RoutedHybrid(chunk, graph), questions, cfg, True)
        if dcfg["hub_ablation"]:
            runs["Graph (hub-pruned)"] = run_pipeline(
                "graph, hub-pruned", graph.pruned(cfg["hub_cap"]), questions, cfg, True)

        print("[no-context] answering from memory", flush=True)
        with ThreadPoolExecutor(cfg["llm_workers"]) as pool:
            preds = list(pool.map(lambda q: answer(q["question"], None, cfg), questions))
        no_context = [score_row({"retrieved": [], "prediction": p}, q) for p, q in zip(preds, questions)]

    out = ROOT / dcfg["results_dir"]
    if args.limit or args.chunk_only:
        out = out / "dev"  # never overwrite the real table with a partial run
    out.mkdir(parents=True, exist_ok=True)
    for name, r in runs.items():
        slug = SLUGS.get(name, name.lower())
        with open(out / f"{slug}_outputs.jsonl", "w") as f:
            for row, s, q in zip(r["rows"], r["scores"], questions):
                f.write(json.dumps({**row, "question": q["question"], "answer": q["answer"],
                                    "type": q["type"], "gold_titles": q["gold_titles"], **s}) + "\n")
    md = results_markdown(questions, runs, cfg, no_context)
    (out / "results.md").write_text(md)
    write_all(questions, runs, cfg, no_context, out)
    print("\n" + md)
    print(f"wrote {Path(out, 'results.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
