"""Step 8: length filter (10-512 tokens under our tokenizer), final corpus, and the statistics Part 1
asks for.

  * keeps instances of MIN_TOKENS..MAX_TOKENS tokens (</s> included, as the model sees them);
  * writes the final corpus with a seeded 1% held-out split for pre-training;
  * source mix of the final corpus; token-length distributions of the corpus and of each
    fine-tuning task's inputs (prefix included) and outputs, per split; the share of fine-tuning
    instances longer than 512 tokens;
  * the tokenization of three real examples.

Single process on purpose (light on the machine). Output: $DATA/corpus/{train,heldout}.jsonl.gz,
corpus/counts/08_length_and_stats.json, corpus/tokenized_examples.md

Usage: python src/08_length_and_stats.py
"""
import gzip
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
FT = ROOT.parent / "code-review-finetuning-data"
MIN_TOKENS, MAX_TOKENS, HELDOUT, SEED = 10, 512, 0.01, 680
PREFIX = {"code-and-comment-to-code": "code&comment2code: ", "code-to-comment": "code2comment: "}


# Summary of a list of token lengths. p50 is the median (half are shorter), p90 means 90% are
# shorter. E.g. code2comment targets (the reviewer comments), train split:
#     p50 18, p90 52, max 478, over_512 0
def dist(lengths):
    a = np.asarray(lengths)
    return {"n": int(a.size), "mean": round(float(a.mean()), 1),
            **{f"p{q}": int(np.percentile(a, q)) for q in (10, 50, 90, 99)}, "max": int(a.max()),
            "over_512": int((a > MAX_TOKENS).sum()), "over_512_pct": round(100 * float((a > MAX_TOKENS).mean()), 2)}


# token count of each text (</s> included), tokenized in batches of 2,000
def lengths(tok, texts):
    out = []
    for i in range(0, len(texts), 2000):
        out += [len(ids) for ids in tok(texts[i:i + 2000]).input_ids]
    return out


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")
    recs = [json.loads(l) for l in gzip.open(DATA / "corpus/instances.jsonl.gz", "rt", encoding="utf-8")]
    lens = lengths(tok, [r["text"] for r in recs])
    # The assignment's length rule: keep instances of 10 to 512 tokens (</s> included). Too short:
    # mostly commit messages such as "Exclude BlueMapAPI from spotless" (7 tokens). Too long: mostly
    # review threads with long diff hunks. Long ones are dropped, not cut: cutting a review thread
    # would remove the comment at its end.
    removed = Counter()
    kept = []
    for r, n in zip(recs, lens):
        if n < MIN_TOKENS:
            removed[(r["source"], "under 10 tokens")] += 1
        elif n > MAX_TOKENS:
            removed[(r["source"], "over 512 tokens")] += 1
        else:
            r["tokens"] = n; kept.append(r)

    rng = random.Random(SEED)
    # shuffle with a fixed seed, then the first 1% is the held-out split
    rng.shuffle(kept)
    n_held = round(HELDOUT * len(kept))
    for name, part in (("heldout", kept[:n_held]), ("train", kept[n_held:])):
        with gzip.open(DATA / f"corpus/{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # source mix of the final corpus, by instances and by tokens
    by_src = Counter(r["source"] for r in kept)
    tokens_by_src = Counter()
    for r in kept:
        tokens_by_src[r["source"]] += r["tokens"]
    stats = {
        "length_filter": {f"{s}/{why}": v for (s, why), v in sorted(removed.items())},
        "in": len(recs), "final": len(kept), "heldout": n_held, "train": len(kept) - n_held,
        "mix_instances": {s: {"n": v, "pct": round(100 * v / len(kept), 1)} for s, v in by_src.items()},
        "mix_tokens": {s: {"n": v, "pct": round(100 * v / sum(tokens_by_src.values()), 1)} for s, v in tokens_by_src.items()},
        "corpus_tokens_total": sum(tokens_by_src.values()),
        "corpus_length": {s: dist([r["tokens"] for r in kept if r["source"] == s]) for s in by_src} | {"all": dist([r["tokens"] for r in kept])},
        "corpus_length_before_filter": dist(lens),
        "finetuning": {},
    }
    # the same length statistics for each fine-tuning task and split (inputs with their prefix)
    for task, prefix in PREFIX.items():
        for split in ("train", "val", "test"):
            rows = [l.rstrip("\n").split("\t", 1) for l in open(FT / task / f"{split}.tsv", encoding="utf-8")]
            stats["finetuning"][f"{task}/{split}"] = {"input": dist(lengths(tok, [prefix + s for s, _ in rows])),
                                                     "output": dist(lengths(tok, [t for _, t in rows]))}
    (ROOT / "corpus/counts/08_length_and_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

    # three real examples: one method with Javadoc, one commit, one review thread
    rng = random.Random(SEED)
    picks = [rng.choice([r for r in kept if r["source"] == "methods" and r["text"].startswith("/**") and 40 < r["tokens"] < 90]),
             rng.choice([r for r in kept if r["source"] == "commits" and 15 < r["tokens"] < 40]),
             rng.choice([r for r in kept if r["source"] == "reviews" and 60 < r["tokens"] < 140])]
    md = ["# Three tokenized examples\n"]
    for r in picks:
        pieces = tok.convert_ids_to_tokens(tok(r["text"]).input_ids)
        md += [f"## {r['source']} ({r['repo']}, {len(pieces)} tokens)\n", "Text:\n", "```\n" + r["text"] + "\n```\n",
               "Pieces (`▁` marks a preceding space):\n", "```\n" + " | ".join(pieces) + "\n```\n"]
    (ROOT / "corpus/tokenized_examples.md").write_text("\n".join(md))
    print(json.dumps({k: stats[k] for k in ("length_filter", "in", "final", "heldout", "mix_instances", "mix_tokens", "corpus_tokens_total")}, indent=1))
    for k, v in stats["finetuning"].items():
        print(k, "input p50/p90/max", v["input"]["p50"], v["input"]["p90"], v["input"]["max"], ">512:", v["input"]["over_512"],
              "| output p50/p90/max", v["output"]["p50"], v["output"]["p90"], v["output"]["max"], ">512:", v["output"]["over_512"])


if __name__ == "__main__":
    main()
