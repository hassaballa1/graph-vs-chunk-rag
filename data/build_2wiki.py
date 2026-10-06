"""Sample and freeze the question set and pooled corpus from 2WikiMultiHopQA.

Run once. Writes data/2wiki/questions.jsonl and data/2wiki/corpus.jsonl, in the
same format as the HotpotQA files, so every pipeline and metric runs unchanged.
"""
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import load_config  # noqa: E402

SOURCE = "framolfese/2WikiMultihopQA"
REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"
TYPES = ("comparison", "bridge_comparison", "compositional", "inference")


def main():
    cfg = load_config()
    dcfg = cfg["datasets"]["2wiki"]
    out_dir = ROOT / dcfg["data_dir"]
    out_q, out_c = out_dir / "questions.jsonl", out_dir / "corpus.jsonl"
    if out_q.exists() and "--force" not in sys.argv:
        print(f"{out_q} already exists; the sample is frozen. Pass --force to rebuild.")
        return

    ds = load_dataset(SOURCE, split="validation", revision=REVISION)
    rng = random.Random(cfg["seed"])
    by_type = {t: [] for t in TYPES}
    for i, t in enumerate(ds["type"]):
        by_type[t].append(i)
    picked = [i for t in TYPES for i in rng.sample(by_type[t], dcfg["n_per_type"])]
    # Shuffle so any prefix (used by --limit) has a mix of types.
    rng.shuffle(picked)

    questions, corpus = [], {}
    for i in picked:
        row = ds[i]
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            # Unlike HotpotQA, 2Wiki sentences carry no leading space.
            corpus.setdefault(title, " ".join(s.strip() for s in sentences).strip())
        questions.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "gold_titles": sorted(set(row["supporting_facts"]["title"])),
                "context_titles": list(ctx["title"]),
                "evidences": row["evidences"],
            }
        )

    meta = {
        "source": f"{SOURCE} validation @ {REVISION[:7]}",
        "license": "Apache 2.0",
        "seed": cfg["seed"],
        "n_per_type": dcfg["n_per_type"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_q, "w") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(out_c, "w") as f:
        f.write(json.dumps({"_meta": {**meta, "n_paragraphs": len(corpus)}}) + "\n")
        for title, text in corpus.items():
            f.write(json.dumps({"id": title, "title": title, "text": text}, ensure_ascii=False) + "\n")
    print(f"wrote {len(questions)} questions and {len(corpus)} paragraphs")


if __name__ == "__main__":
    main()
