"""Step 4: assemble the raw material into per-source records and remove exact and near duplicates.

Runs before any other filter (the assignment requires it). Per source, globally across repositories:

  0. repositories listed twice by GHS (old and new name of a transferred repository, same HEAD
     SHA) are reduced to one: ALIASES below are dropped;
  1. exact duplicates: identical token sequences (whitespace-insensitive);
  2. near duplicates: Jaccard similarity >= THRESHOLD between the sets of token 3-grams,
     estimated with MinHash (NUM_PERM permutations) + LSH and confirmed on the exact sets.

What is compared: methods -> their code only (no comments, no Javadoc); commits -> the message
without trailers; reviews -> the comments and the diff hunk separately: two threads are near
duplicates only if BOTH are >= THRESHOLD similar (a long shared hunk must not make two different
comments on the same code look like copies). Which copy is kept: the one with the
most Javadoc + comment text (methods), then the first in (repository, path, line) order.

Output: $DATA/dedup/{methods,commits,reviews}.jsonl.gz and corpus/counts/04_dedup.json.

Usage: python src/04_dedup.py [--workers 16]
"""
import argparse
import gzip
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from datasketch import LeanMinHash, MinHash, MinHashLSH

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
OUT = DATA / "dedup"
# Near duplicate = Jaccard >= 0.8 on sets of 3-token sequences (see shingles below for examples).
# SHINGLE = 3, a balance between two failure modes:
#   - too small (1): everything looks similar, because all Java shares common tokens (public, int,
#     return, {, ;) and word order is lost. Too much dedup: distinct methods are removed as copies.
#   - too large (5+): everything looks different, because one changed token breaks every window
#     that contains it. Too little dedup: copies with a small edit slip through as different.
#   Our methods (median 50 tokens) and commits (median 15) are short; with 3, copies score ~0.85
#   and different items ~0.35 (examples at shingles below).
# THRESHOLD = 0.8: higher keeps more near-copies, lower starts removing distinct items.
# NUM_PERM = 128: MinHash signature size; more is more precise and slower.
THRESHOLD, NUM_PERM, SHINGLE = 0.8, 128, 3
# old names of 3 repositories GHS also lists under their new names; dropped as duplicates
ALIASES = {"TeamOpenSmartGlasses/AugmentOS", "kyoripowered/adventure", "raiman/sikulix1"}

# Commits marked as written with an AI assistant: a co-author trailer naming one, the Claude Code
# footer, or an agent as the author. Recorded before trailers are stripped; removed in step 5.
AI_TRAILER = re.compile(r"^co-authored-by:.*(claude|anthropic|cursor|copilot|devin|chatgpt|openai|codex|gemini)"
                        r"|generated with \[?claude code", re.IGNORECASE | re.MULTILINE)
AI_AUTHOR = re.compile(r"copilot|cursor ?agent|devin-ai|claude|codex", re.IGNORECASE)
# a token is a word (identifier, number) or one punctuation character; whitespace is ignored
TOKEN = re.compile(r"\w+|[^\w\s]")
TRAILER = re.compile(r"^[A-Za-z][\w-]*: \S")   # git trailer format: "Key: value", key without spaces


def tokens(s):
    return TOKEN.findall(s)


# The set of consecutive 3-token sequences ("shingles") of a text. Two texts are compared by the
# Jaccard similarity of their shingle sets: the size of the overlap divided by the size of the union.
# For example:
#
#     return a + b;   ->  {return a +,  a + b,  + b ;}
#     return a + c;   ->  {return a +,  a + c,  + c ;}
#
#     overlap 1, union 5: Jaccard = 1/5 = 0.2, different
#
# Two copies of a method that differ by one added `final`:
#
#     public int sum(int[] values) { int total = 0; for (int v : values) { total += v; } return total; }
#     public int sum(final int[] values) { int total = 0; for (int v : values) { total += v; } return total; }
#
#     overlap 28, union 33: Jaccard = 0.85 >= THRESHOLD (0.8), a near duplicate
def shingles(toks):
    if len(toks) < SHINGLE:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + SHINGLE]) for i in range(len(toks) - SHINGLE + 1)}


def strip_trailers(msg):
    """Drop the trailer block (Signed-off-by:, Co-authored-by:, Change-Id: ...). As in git, trailers
    are only the lines of the last paragraph, and never the first paragraph (the subject)."""
    paras = [p for p in re.split(r"\n\s*\n", msg.strip()) if p.strip()]
    if len(paras) > 1:
        lines = paras[-1].splitlines()
        while lines and TRAILER.match(lines[-1].strip()):
            lines.pop()
        paras[-1] = "\n".join(lines)
    return " ".join(" ".join(paras).split())


def load():
    # one file per repository; a "name 2.jsonl.gz" copy (made by a sync tool) would be read twice
    stray = [f.name for d in ("raw", "reviews") for f in (DATA / d).iterdir() if " " in f.name]
    assert not stray, f"stray copies, move them away first: {stray}"
    methods, commits, reviews = [], [], []
    collected = {"methods": 0, "commits": 0, "reviews": 0}
    aliased = {"methods": 0, "commits": 0, "reviews": 0}
    for f in sorted((DATA / "raw").glob("*.jsonl.gz"), key=lambda p: p.name.lower()):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line)
            # the raw files hold methods and commits together; split them by kind
            src = "methods" if r["kind"] == "method" else "commits"
            collected[src] += 1
            if r["repo"] in ALIASES:
                aliased[src] += 1
                continue
            if r["kind"] == "method":
                # methods are compared on code only (no comments); the copy with the most English wins
                r["key"] = r["code"]
                r["english"] = len(r["javadoc"]) + sum(len(c) for c in r["comments"])
                methods.append(r)
            else:
                # flag AI-assisted commits before the trailers that reveal them are stripped
                r["ai"] = bool(AI_TRAILER.search(r["text"]) or AI_AUTHOR.search(r["author"] + " " + r["email"]))
                r["text"] = strip_trailers(r["text"])
                r["key"] = r["text"]
                commits.append(r)
    for f in sorted((DATA / "reviews").glob("*.jsonl.gz"), key=lambda p: p.name.lower()):
        # review comments -> threads: follow in_reply_to_id up to the first comment
        by_id, threads = {}, defaultdict(list)
        for line in gzip.open(f, "rt", encoding="utf-8"):
            r = json.loads(line)
            by_id[r["id"]] = r
        for r in by_id.values():
            root = r
            while root.get("in_reply_to_id") in by_id:
                root = by_id[root["in_reply_to_id"]]
            threads[root["id"]].append(r)
        for rid, cs in sorted(threads.items()):
            cs.sort(key=lambda c: (c["created_at"], c["id"]))
            root = by_id[rid]
            collected["reviews"] += 1
            if root["repo"] in ALIASES:
                aliased["reviews"] += 1
                continue
            rec = {"kind": "review", "repo": root["repo"], "path": root["path"], "id": rid,
                   "diff_hunk": root["diff_hunk"] or "",
                   "comments": [{"user": c["user"], "user_type": c["user_type"], "body": c["body"] or ""} for c in cs],
                   "root_missing": root.get("in_reply_to_id") is not None}
            # reviews: comments and hunk are compared separately (key and key2), both must match
            rec["key"] = "\n".join(c["body"] for c in rec["comments"])
            rec["key2"] = rec["diff_hunk"]
            reviews.append(rec)
    return {"methods": methods, "commits": commits, "reviews": reviews}, collected, aliased


# MinHash: a fingerprint of 128 numbers per record, to find likely duplicates fast.
# Comparing all 884,000 records pairwise would be ~390 billion Jaccard computations.
# For each of 128 random hash functions, hash every shingle and keep the smallest value; two records
# share that minimum with probability equal to their Jaccard. So the share of matching positions
# estimates Jaccard, e.g.:
#     sum vs sum + final: 105/128 positions agree -> ~0.82 (exact Jaccard 0.85)
#     sum vs max:          38/128 positions agree -> ~0.30 (exact Jaccard 0.35)
# LSH (in dedup below) buckets the fingerprints so each record is only compared with the few records
# in its buckets; those candidates are then confirmed with the exact Jaccard.
def minhash(key):
    m = MinHash(num_perm=NUM_PERM)
    m.update_batch([s.encode("utf-8") for s in shingles(tokens(key))])
    return LeanMinHash(m)


def dedup(records, workers):
    n0 = len(records)
    # Priority: most English first (methods), then the fixed (repository, path, line) order of loading.
    order = sorted(range(n0), key=lambda i: -records[i].get("english", 0))
    # 1. exact
    # keep the first copy of each identical token sequence
    seen, exact_kept = set(), []
    for i in order:
        k = " ".join(tokens(records[i]["key"] + "\n" + records[i].get("key2", "")))
        if k not in seen:
            seen.add(k); exact_kept.append(i)
    # 2. near
    # compute MinHash signatures in parallel
    with Pool(workers) as pool:
        hv = pool.map(minhash, [records[i]["key"] for i in exact_kept], chunksize=2000)
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    sets, sets2, kept, examples = {}, {}, [], []
    jac = lambda a, b: len(a & b) / len(a | b)
    for pos, i in enumerate(exact_kept):
        sh = shingles(tokens(records[i]["key"]))
        sh2 = shingles(tokens(records[i]["key2"])) if "key2" in records[i] else None
        m = hv[pos]
        dup = None
        # LSH proposes likely duplicates; confirm each with the exact Jaccard similarity
        for j in lsh.query(m):
            if jac(sh, sets[j]) >= THRESHOLD and (sh2 is None or jac(sh2, sets2[j]) >= THRESHOLD):
                dup = j; break
        if dup is None:
            lsh.insert(i, m); sets[i] = sh; sets2[i] = sh2; kept.append(i)
        elif len(examples) < 20:
            examples.append({"kept": records[dup]["key"][:400], "removed": records[i]["key"][:400]})
    kept.sort()
    return [records[i] for i in kept], {"in": n0, "exact_removed": n0 - len(exact_kept),
                                       "near_removed": len(exact_kept) - len(kept), "out": len(kept)}, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "corpus/counts").mkdir(parents=True, exist_ok=True)
    data, collected, aliased = load()
    counts = {"alias_repositories": sorted(ALIASES)}
    for source, recs in data.items():
        kept, c, ex = dedup(recs, args.workers)
        c = {"collected": collected[source], "alias_removed": aliased[source], **c}
        # the counts must add up, for the count table
        assert c["collected"] - c["alias_removed"] - c["exact_removed"] - c["near_removed"] == c["out"]
        counts[source] = c
        print(source, c, flush=True)
        with gzip.open(OUT / f"{source}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in kept:
                # the comparison keys are not saved
                r.pop("key", None); r.pop("key2", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (OUT / f"{source}.near_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    (ROOT / "corpus/counts/04_dedup.json").write_text(json.dumps(counts, indent=2) + "\n")


if __name__ == "__main__":
    main()
