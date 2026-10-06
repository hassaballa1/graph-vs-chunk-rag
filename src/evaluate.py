"""Metrics, bootstrap intervals, and the results table."""
import re
import string
from collections import Counter

import numpy as np


# --- Answer scoring, ported from the official hotpot_evaluate_v1.py ---------

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    special = ("yes", "no", "noanswer")
    if normalized_prediction in special and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in special and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction, ground_truth):
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# --- Retrieval scoring against gold supporting-paragraph titles -------------

def support_recall(retrieved_titles, gold_titles):
    gold = set(gold_titles)
    return len(gold & set(retrieved_titles)) / len(gold)


def full_support(retrieved_titles, gold_titles):
    return float(set(gold_titles) <= set(retrieved_titles))


def score_row(row, question):
    titles = row["retrieved"]
    return {
        "recall": support_recall(titles, question["gold_titles"]),
        "full_support": full_support(titles, question["gold_titles"]),
        "f1": f1_score(row["prediction"], question["answer"]),
        "em": exact_match_score(row["prediction"], question["answer"]),
    }


# --- Paired bootstrap on the per-question difference ------------------------

def bootstrap_diff(a, b, n, seed):
    """95% interval for mean(b - a), resampling questions with replacement."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    return float((b - a).mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --- Report -----------------------------------------------------------------

def pct(x):
    return f"{100 * x:.1f}"


TYPE_ORDER = ("bridge", "comparison", "bridge_comparison", "compositional", "inference")


def question_types(questions):
    """Question types present, in a fixed display order."""
    present = {q["type"] for q in questions}
    return [t for t in TYPE_ORDER if t in present] + sorted(present - set(TYPE_ORDER))


def type_label(t):
    return t.replace("_", "-")


def summarise(scores, questions):
    """Overall means, plus means per question type under ("by_type", type)."""
    def mean(metric, qtype=None):
        vals = [s[metric] for s, q in zip(scores, questions) if qtype in (None, q["type"])]
        return float(np.mean(vals)) if vals else float("nan")

    out = {m: mean(m) for m in ("recall", "full_support", "f1", "em")}
    out["by_type"] = {t: {m: mean(m, t) for m in ("recall", "full_support", "f1")}
                      for t in question_types(questions)}
    return out


def gap_rows(questions, runs, cfg, baseline="Chunk", only=None):
    """Each pipeline (or just `only`) minus the baseline, per metric and question type."""
    rows = []
    for name in only or runs:
        if name == baseline:
            continue
        for metric in ("recall", "full_support", "f1"):
            for qtype in (None, *question_types(questions)):
                keep = [i for i, q in enumerate(questions) if qtype in (None, q["type"])]
                a = [runs[baseline]["scores"][i][metric] for i in keep]
                b = [runs[name]["scores"][i][metric] for i in keep]
                d, lo, hi = bootstrap_diff(a, b, cfg["bootstrap_samples"], cfg["seed"])
                rows.append({"pipeline": name, "metric": metric, "subset": qtype or "all",
                             "diff": d, "lo": lo, "hi": hi})
    return rows


def results_markdown(questions, runs, cfg, no_context=None):
    """runs: {name: {"scores": [...], "index_tokens", "index_seconds", "latencies"}}"""
    k = cfg["k"]
    types = question_types(questions)
    counts = ", ".join(f"{sum(q['type'] == t for q in questions)} {type_label(t)}" for t in types)
    lines = [
        f"# Results\n",
        f"{len(questions)} {cfg['dataset_name']} questions ({counts}), "
        f"generator `{cfg['llm_model']}`, embeddings `{cfg['embedding_model']}`, "
        f"k = {k}, context cap {cfg['max_context_tokens']} tokens.\n",
        f"| Pipeline | Recall@{k} | Full support@{k} | F1 all | "
        + " | ".join(f"F1 {type_label(t)}" for t in types)
        + " | EM | Index tokens | Index time | Latency |",
        "|---|" + "---|" * (len(types) + 7),
    ]
    for name, r in runs.items():
        s = summarise(r["scores"], questions)
        lines.append(
            f"| {name} | {pct(s['recall'])} | {pct(s['full_support'])} | {pct(s['f1'])} | "
            + " | ".join(pct(s["by_type"][t]["f1"]) for t in types)
            + f" | {pct(s['em'])} | {r['index_tokens']:,} | {r['index_seconds']:.0f} s "
            f"| {1000 * float(np.median(r['latencies'])):.0f} ms |"
        )
    if no_context is not None:
        s = summarise(no_context, questions)
        lines.append(
            f"\nNo-context floor (model answers from memory): F1 {pct(s['f1'])}, EM {pct(s['em'])}."
        )

    if "Chunk" in runs and "Graph" in runs:
        # Upper bound for any hybrid of the two: a router that knows which one wins each question.
        oracle = [{m: max(a[m], b[m]) for m in a}
                  for a, b in zip(runs["Chunk"]["scores"], runs["Graph"]["scores"])]
        s = summarise(oracle, questions)
        lines.append(
            f"\nOracle router (best of chunk and graph per question, a ceiling, not a pipeline): "
            f"Recall@{k} {pct(s['recall'])}, F1 {pct(s['f1'])} ("
            + ", ".join(f"{type_label(t)} {pct(s['by_type'][t]['f1'])}" for t in types) + ")."
        )

    sections = [("Difference from chunk", gap_rows(questions, runs, cfg))]
    vs_graph = [n for n in runs if n.startswith("Hybrid") or n == "Graph v2"]
    if vs_graph and "Graph" in runs:
        sections.append(("Difference from graph", gap_rows(questions, runs, cfg, baseline="Graph", only=vs_graph)))
    for heading, gaps in sections:
        if not gaps:
            continue
        lines.append(f"\n## {heading}, with paired bootstrap 95% intervals\n")
        lines.append("| Pipeline | Metric | Subset | Difference | 95% interval | Crosses zero |")
        lines.append("|---|---|---|---|---|---|")
        for g in gaps:
            lines.append(
                f"| {g['pipeline']} | {g['metric']} | {type_label(g['subset'])} | {pct(g['diff'])} "
                f"| [{pct(g['lo'])}, {pct(g['hi'])}] | {'yes' if g['lo'] <= 0 <= g['hi'] else 'no'} |"
            )
    return "\n".join(lines) + "\n"
