"""Step 13: the Part 4 metrics, from the prediction files of step 12. Runs in .venv-eval
(sacrebleu, codebleu); needs no GPU.

Per task and condition, on the 5,000 test predictions:
  * exact match: correct predictions out of 5,000 and the percentage; correct = equal to the
    target after collapsing runs of whitespace (rechecked here against the "correct" column);
  * code&comment2code: CodeBLEU, package codebleu 0.7.0, lang="java", default weights
    (0.25 n-gram, 0.25 weighted n-gram, 0.25 syntax, 0.25 dataflow), corpus level;
  * code2comment: BLEU-4, sacrebleu 2.6.0, corpus level, default settings (signature reported);
  * exact match by confidence: ten buckets of the score (sequence probability, not
    length-normalized) [0, 0.1), ..., [0.9, 1], with their size and exact match;
  * thresholds: for t = 0.1 ... 0.9, the coverage and exact match of predictions scoring >= t, on
    the validation sample (where the threshold is chosen) and on test (where it is reported);
  * pre-training effect per task: McNemar's exact test on the paired test predictions of the two
    conditions (pairs where only one is correct) and the odds ratio pt/nopt, as in the paper.

--errors <condition> also draws 25 wrong test predictions per task (seed 680) into
results/error_analysis.csv, with an empty label column for manual labelling
(semantically equivalent / valid alternative / wrong).

Output: results/metrics.json, results/metrics.md

Usage: .venv-eval/bin/python src/13_metrics.py [--errors pt]
"""
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TASKS = ["code&comment2code", "code2comment"]
SEED = 680
csv.field_size_limit(sys.maxsize)


def normalize(s):
    return " ".join(s.split())


def load(path):
    return list(csv.DictReader(open(path, encoding="utf-8", newline="")))


def buckets(rows):
    out = []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        inside = [r for r in rows if lo <= float(r["score"]) / 100 < hi or (b == 9 and float(r["score"]) / 100 == 1)]
        hits = sum(r["correct"] == "true" for r in inside)
        out.append({"bucket": f"[{lo:.1f}, {hi:.1f}{']' if b == 9 else ')'}", "n": len(inside), "correct": hits,
                    "exact_match_pct": round(100 * hits / len(inside), 1) if inside else None})
    return out


def thresholds(rows):
    out = []
    for t in [i / 10 for i in range(1, 10)]:
        above = [r for r in rows if float(r["score"]) / 100 >= t]
        hits = sum(r["correct"] == "true" for r in above)
        out.append({"threshold": t, "n": len(above), "coverage_pct": round(100 * len(above) / len(rows), 1),
                    "correct": hits, "exact_match_pct": round(100 * hits / len(above), 1) if above else None})
    return out


def mcnemar(rows_nopt, rows_pt):
    """Exact two-sided McNemar test on paired predictions (aligned by input)."""
    a = {r["input"]: r["correct"] == "true" for r in rows_nopt}
    b = {r["input"]: r["correct"] == "true" for r in rows_pt}
    assert a.keys() == b.keys()
    only_pt = sum(b[k] and not a[k] for k in a)
    only_nopt = sum(a[k] and not b[k] for k in a)
    n, k = only_pt + only_nopt, min(only_pt, only_nopt)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only_pt_correct": only_pt, "only_nopt_correct": only_nopt, "p_value": p,
            "odds_ratio_pt_vs_nopt": round(only_pt / only_nopt, 2) if only_nopt else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", choices=["nopt", "pt"], help="condition to draw the error-analysis sample from")
    args = ap.parse_args()
    test = load(RES / "predictions.csv")
    for r in test:
        assert (normalize(r["predicted"]) == normalize(r["expected"])) == (r["correct"] == "true"), r["input"][:80]
    conditions = sorted({r["condition"] for r in test})
    metrics, md = {}, ["# Test results\n"]
    for cond in conditions:
        val_path = RES / f"val_predictions-{cond}.csv"
        val = load(val_path) if val_path.exists() else []
        for task in TASKS:
            rows = [r for r in test if r["task"] == task and r["condition"] == cond]
            if not rows:
                continue
            hits = sum(r["correct"] == "true" for r in rows)
            m = {"n": len(rows), "exact_match": hits, "exact_match_pct": round(100 * hits / len(rows), 2)}
            preds, refs = [r["predicted"] for r in rows], [r["expected"] for r in rows]
            if task == "code2comment":
                import sacrebleu
                bleu = sacrebleu.BLEU()
                m["bleu4"] = round(bleu.corpus_score(preds, [refs]).score, 2)
                m["bleu_signature"] = str(bleu.get_signature())
            else:
                from codebleu import calc_codebleu
                cb = calc_codebleu(refs, preds, lang="java")
                m["codebleu"] = {k: round(100 * v, 2) for k, v in cb.items()}
            m["buckets"] = buckets(rows)
            m["thresholds_test"] = thresholds(rows)
            m["thresholds_val"] = thresholds([r for r in val if r["task"] == task])
            metrics[f"{task}/{cond}"] = m
            md += [f"## {task} · {cond}\n",
                   f"Exact match: **{hits:,} / {len(rows):,} ({m['exact_match_pct']}%)**. " +
                   (f"BLEU-4: **{m['bleu4']}** (`{m['bleu_signature']}`)." if "bleu4" in m else
                    f"CodeBLEU: **{m['codebleu']['codebleu']}** (n-gram {m['codebleu']['ngram_match_score']}, "
                    f"weighted {m['codebleu']['weighted_ngram_match_score']}, syntax {m['codebleu']['syntax_match_score']}, "
                    f"dataflow {m['codebleu']['dataflow_match_score']})."),
                   "\n| Confidence bucket | Predictions | Exact match |", "| --- | ---: | ---: |"]
            md += [f"| {b['bucket']} | {b['n']:,} | {'' if b['exact_match_pct'] is None else str(b['exact_match_pct']) + '%'} |"
                   for b in m["buckets"]]
            md += ["\n| Threshold | Val coverage | Val exact match | Test coverage | Test exact match |",
                   "| ---: | ---: | ---: | ---: | ---: |"]
            for v, t in zip(m["thresholds_val"] or [{}] * 9, m["thresholds_test"]):
                md.append(f"| ≥ {t['threshold']:.1f} | {v.get('coverage_pct', '')}% | {v.get('exact_match_pct') if v.get('exact_match_pct') is not None else '—'}{'%' if v.get('exact_match_pct') is not None else ''} | "
                          f"{t['n']:,} ({t['coverage_pct']}%) | {t['exact_match_pct'] if t['exact_match_pct'] is not None else '—'}{'%' if t['exact_match_pct'] is not None else ''} |")
            md.append("")
    if set(conditions) >= {"nopt", "pt"}:
        md += ["## Pre-training effect (McNemar's exact test, test set)\n",
               "| Task | Only B correct | Only A correct | p-value | Odds ratio B/A |", "| --- | ---: | ---: | ---: | ---: |"]
        for task in TASKS:
            t = mcnemar([r for r in test if r["task"] == task and r["condition"] == "nopt"],
                        [r for r in test if r["task"] == task and r["condition"] == "pt"])
            metrics[f"{task}/mcnemar"] = t
            md.append(f"| {task} | {t['only_pt_correct']} | {t['only_nopt_correct']} | {t['p_value']:.2g} | {t['odds_ratio_pt_vs_nopt']} |")
        md.append("")
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (RES / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))

    if args.errors:
        rng = random.Random(SEED)
        sample = []
        for task in TASKS:
            wrong = [r for r in test if r["task"] == task and r["condition"] == args.errors and r["correct"] == "false"]
            sample += [{"task": task, "input": r["input"], "expected": r["expected"], "predicted": r["predicted"],
                        "label": "", "note": ""} for r in rng.sample(wrong, min(25, len(wrong)))]
        with open(RES / "error_analysis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sample[0]), quoting=csv.QUOTE_ALL)
            w.writeheader(); w.writerows(sample)
        print(f"error-analysis sample: {len(sample)} rows in results/error_analysis.csv")


if __name__ == "__main__":
    main()
