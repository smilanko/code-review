"""Step 9: the count table, collected -> deduplicated -> filtered -> decontaminated -> length -> final,
per source, assembled from corpus/counts/*.json. Every row states what it removed; the script
asserts that each column closes.

Output: corpus/count_table.md

Usage: python src/09_count_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C = ROOT / "corpus/counts"
SOURCES = ["methods", "commits", "reviews"]
NAMES = {"methods": "Methods", "commits": "Commit messages", "reviews": "Review threads"}
# filter rows in pipeline order (05_filter.py), with the reason shown in the table
FILTERS = {
    "parse errors": "method does not parse cleanly",
    "generated": "generated file (path, header, generator tool or @Generated)",
    "trivial without Javadoc": "getter, setter, empty body or one-call delegation, with no Javadoc",
    "AI-assisted": "commit marked as written with an AI assistant",
    "bots": "commit by a bot account",
    "revert/merge": "Revert \"...\" and Merge branch/pull request messages",
    "bots (whole thread)": "thread with no human comment once bot comments are removed",
    "per-hunk cap": "more than 10 threads on the same diff hunk (earliest kept)",
    "non-English": "2+ letters from a non-Latin script, outside code and URLs",
}
LENGTH = "assignment rule: 10 to 512 tokens under our tokenizer"


def main():
    # the counts each earlier step saved
    d4 = json.loads((C / "04_dedup.json").read_text())
    d5 = json.loads((C / "05_filter.json").read_text())
    d6 = json.loads((C / "06_decontaminate.json").read_text())
    d7 = json.loads((C / "07_tokenizer.json").read_text())
    d8 = json.loads((C / "08_length_and_stats.json").read_text())

    # One row per step, per source, in the order the steps ran. Removal rows are negative; rows starting
# with "=" are running totals. E.g. for methods:
#     Collected 584,610 | Duplicate repositories -16,076 | Exact duplicates -69,759 |
#     Near duplicates -19,722 | = Deduplicated 479,053
    rows = [("Collected", "", {s: d4[s]["collected"] for s in SOURCES})]
    rows.append(("Duplicate repositories", "same project listed twice by GHS (old and new name)",
                 {s: -d4[s]["alias_removed"] for s in SOURCES}))
    rows.append(("Exact duplicates", "identical token sequence", {s: -d4[s]["exact_removed"] for s in SOURCES}))
    rows.append(("Near duplicates", "Jaccard >= 0.8 on token 3-grams", {s: -d4[s]["near_removed"] for s in SOURCES}))
    rows.append(("= Deduplicated", "", {s: d4[s]["out"] for s in SOURCES}))
    # every filter name in the step-5 counts must have a reason above
    names = {name for src in SOURCES for name in d5[src]["removed"]}
    assert names <= FILTERS.keys(), names - FILTERS.keys()
    for name, why in FILTERS.items():
        if name in names:
            rows.append((f"Filter: {name}", why, {s: -d5[s]["removed"].get(name, 0) for s in SOURCES}))
    rows.append(("= Filtered", "", {s: d5[s]["out"] for s in SOURCES}))
    rows.append(("Decontamination", "contains a validation/test method, revised method, or comment of >= 4 words",
                 {s: -d6[s]["removed"] for s in SOURCES}))
    rows.append(("= Decontaminated", "", {s: d6[s]["out"] for s in SOURCES}))
    rows.append(("Empty message", "commit with no message text", {s: -d7["empty_removed"][s] for s in SOURCES}))
    lf = d8["length_filter"]
    rows.append(("Under 10 tokens", LENGTH, {s: -lf.get(f"{s}/under 10 tokens", 0) for s in SOURCES}))
    rows.append(("Over 512 tokens", LENGTH, {s: -lf.get(f"{s}/over 512 tokens", 0) for s in SOURCES}))
    rows.append(("= Final", "", {s: d8["mix_instances"][s]["n"] for s in SOURCES}))

    # the arithmetic must close: replay the subtractions; each "=" row must equal the running sum
    running = dict(rows[0][2])
    for label, _, vals in rows[1:]:
        if label.startswith("="):
            assert running == vals, (label, running, vals)
        else:
            running = {s: running[s] + vals[s] for s in SOURCES}

    # write the table as Markdown, totals in bold
    lines = ["| Step | Reason | " + " | ".join(NAMES[s] for s in SOURCES) + " | Total |",
             "| --- | --- | " + " | ".join("---:" for _ in SOURCES) + " | ---: |"]
    for label, why, vals in rows:
        tot = sum(vals.values())
        cells = [f"{v:,}".replace("-", "−") for v in vals.values()] + [f"{tot:,}".replace("-", "−")]
        if label.startswith("="):
            label, cells = f"**{label[2:]}**", [f"**{c}**" for c in cells]
        lines.append(f"| {label} | {why} | " + " | ".join(cells) + " |")
    (ROOT / "corpus/count_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
