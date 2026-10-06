"""Sample and freeze the question set and pooled corpus from HotpotQA.

Run once. Writes data/questions.jsonl and data/corpus.jsonl; the first line of
each is a {"_meta": ...} header recording the seed and source.
"""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402


def main():
    cfg = load_config()
    dcfg = cfg["datasets"]["hotpotqa"]
    out_q = ROOT / dcfg["data_dir"] / "questions.jsonl"
    out_c = ROOT / dcfg["data_dir"] / "corpus.jsonl"
    if out_q.exists() and "--force" not in sys.argv:
        print(f"{out_q} already exists; the sample is frozen. Pass --force to rebuild.")
        return

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(cfg["seed"])
    by_type = {"bridge": [], "comparison": []}
    for i, t in enumerate(ds["type"]):
        by_type[t].append(i)
    picked = rng.sample(by_type["bridge"], dcfg["n_bridge"]) + rng.sample(
        by_type["comparison"], dcfg["n_comparison"]
    )
    # Shuffle so any prefix (used by --limit) has a mix of both types.
    rng.shuffle(picked)

    questions, corpus = [], {}
    for i in picked:
        row = ds[i]
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            corpus.setdefault(title, "".join(sentences).strip())
        questions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "level": row["level"],
                "gold_titles": sorted(set(row["supporting_facts"]["title"])),
                "context_titles": list(ctx["title"]),
            }
        )

    meta = {
        "source": "hotpotqa/hotpot_qa distractor validation",
        "license": "CC BY-SA 4.0",
        "seed": cfg["seed"],
        "n_bridge": dcfg["n_bridge"],
        "n_comparison": dcfg["n_comparison"],
    }
    with open(out_q, "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for q in questions:
            f.write(json.dumps(q) + "\n")
    with open(out_c, "w") as f:
        f.write(json.dumps({"_meta": {**meta, "n_paragraphs": len(corpus)}}) + "\n")
        for title, text in corpus.items():
            f.write(json.dumps({"id": title, "title": title, "text": text}) + "\n")
    print(f"wrote {len(questions)} questions and {len(corpus)} paragraphs")


if __name__ == "__main__":
    main()
