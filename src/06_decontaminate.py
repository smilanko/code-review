"""Step 6: remove every pre-training instance that contains a validation or test input method or
target of either fine-tuning task.

Patterns (union of both tasks' validation and test splits; the two tasks share their methods, so
the code-to-comment sources stand for the methods inside the code&comment-to-code sources):
  method   code-to-comment sources: the method submitted for review
  revised  code&comment-to-code targets: the method after the review
  comment  code-to-comment targets: the reviewer's comment

Matching is containment on token sequences (the tokenization of 04_dedup.py), so it ignores
whitespace and only matches whole tokens. Searched text per instance: methods -> code with and
without comments, and Javadoc; commits -> message; reviews -> comments, and the old and new side of
the diff hunk (without the +/- markers).

Short comment targets ("final", "why?") occur everywhere, so comment patterns are only matched
from MIN_COMMENT_WORDS words up. --measure reports, without removing anything, how many instances
each threshold would remove.

Output: $DATA/decontaminated/{methods,commits,reviews}.jsonl.gz, corpus/counts/06_decontaminate.json

Usage: python src/06_decontaminate.py --measure
       python src/06_decontaminate.py --min-comment-words N
"""
import argparse
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path

import ahocorasick

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
FT = ROOT.parent / "code-review-finetuning-data"
IN, OUT = DATA / "filtered", DATA / "decontaminated"
TOKEN = re.compile(r"\w+|[^\w\s]")
THRESHOLDS = [1, 2, 3, 4, 5, 6, 8, 10]


# Text -> its tokens separated by single spaces, with a space at each end, e.g.
#     "x.close(); // final"  ->  " x . close ( ) ; / / final "
# Whitespace differences disappear, and because every pattern starts and ends with a space, a
# pattern only matches whole tokens: the comment "final" matches " final ", not inside " finally ".
def norm(s):
    return " " + " ".join(TOKEN.findall(s)) + " "      # padded so matches fall on token boundaries


# one fine-tuning split as (source, target) pairs; split on the first tab only
def read_tsv(task, split):
    with open(FT / task / f"{split}.tsv", encoding="utf-8", newline="") as f:
        for line in f:
            yield line.rstrip("\n").split("\t", 1)


def patterns():
    # every validation and test method, comment and revised method becomes a search pattern
    pats = {}
    for split in ("val", "test"):
        for src, tgt in read_tsv("code-to-comment", split):
            pats.setdefault(norm(src), ("method", 0))
            pats.setdefault(norm(tgt), ("comment", len(tgt.split())))
        for _, tgt in read_tsv("code-and-comment-to-code", split):
            pats.setdefault(norm(tgt), ("revised", 0))
    # Aho-Corasick: we must check ~30,000 patterns against 765,000 instances. Checking each
    # pattern in each text one by one would be 23 billion searches. Instead, all patterns are
    # built once into a single search machine (a tree of their shared prefixes, with links for
    # where to continue when a match fails). It then reads each text once, left to right, and
    # reports every pattern in it, in time that depends on the text's length, not on the number
    # of patterns: like grep with 30,000 search terms at the speed of one.
    automaton = ahocorasick.Automaton()
    for p, (kind, words) in pats.items():
        if p.strip():
            automaton.add_word(p, (kind, words, p))
    automaton.make_automaton()
    return automaton, Counter(k for k, _ in pats.values())


# a diff hunk -> the code before and after the change (drop the +/- markers)
def hunk_sides(hunk):
    old, new = [], []
    for line in hunk.splitlines():
        if line.startswith("@@"):
            continue
        mark, body = line[:1], line[1:]
        if mark != "+":
            old.append(body)
        if mark != "-":
            new.append(body)
    return [" ".join(old), " ".join(new)]


# which text of each instance is searched
def texts(source, r):
    if source == "methods":
        return [r["code"], r["code_c"], r["javadoc"]]
    if source == "commits":
        return [r["text"]]
    return [c["body"] for c in r["comments"]] + hunk_sides(r["diff_hunk"])


def hits(automaton, source, r):
    """(code hit?, longest matching comment target in words, kinds hit)"""
    code, longest, kinds = False, 0, set()
    for t in texts(source, r):
        for _, (kind, words, _) in automaton.iter(norm(t)):
            kinds.add(kind)
            if kind == "comment":
                longest = max(longest, words)
            else:
                code = True
    return code, longest, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--min-comment-words", type=int)
    args = ap.parse_args()
    assert args.measure or args.min_comment_words, "pass --measure or --min-comment-words"
    automaton, n_patterns = patterns()
    print("patterns:", dict(n_patterns), flush=True)
    report, examples = {"patterns": dict(n_patterns)}, []
    for source in ("methods", "commits", "reviews"):
        recs = [json.loads(l) for l in gzip.open(IN / f"{source}.jsonl.gz", "rt", encoding="utf-8")]
        # The expensive part: search every instance once. hits() returns, per instance, whether it
        # contains a validation/test method or revised method (code hit), and the length in words of
        # the longest validation/test comment it contains. The cheap decision comes afterwards, so
        # --measure can compare many thresholds from this one search. With t = 4:
        #     contains a test method                          -> removed
        #     contains "should we use equals here?" (5 words) -> removed
        #     contains "can be removed" (3 words)             -> kept: too common to prove a leak
        res = [hits(automaton, source, r) for r in recs]
        code_hits = sum(c for c, _, _ in res)
        by_threshold = {t: sum(1 for c, w, _ in res if c or w >= t) for t in THRESHOLDS}
        report[source] = {"in": len(recs), "code_hits": code_hits, "removed_by_min_comment_words": by_threshold}
        print(source, json.dumps(report[source]), flush=True)
        # --measure only reports how many each threshold would remove; nothing is written
        if args.measure:
            for r, (c, w, k) in zip(recs, res):
                if (c or w >= 3) and len([e for e in examples if e["source"] == source]) < 4:
                    examples.append({"source": source, "kinds": sorted(k), "comment_words": w,
                                     "text": " | ".join(texts(source, r))[:300]})
            continue
        t = args.min_comment_words
        # remove an instance if it contains a method, or a comment of at least t words
        kept = [r for r, (c, w, _) in zip(recs, res) if not (c or w >= t)]
        OUT.mkdir(parents=True, exist_ok=True)
        with gzip.open(OUT / f"{source}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report[source].update(min_comment_words=t, removed=len(recs) - len(kept), out=len(kept))
    if args.measure:
        (DATA / "decontamination_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    else:
        (ROOT / "corpus/counts/06_decontaminate.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
