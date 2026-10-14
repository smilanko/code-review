"""Step 1: select the mining frame with GHS (https://seart-ghs.si.usi.ch).

Queries the GHS search API with the parameters in QUERY, pages through every result,
and writes:
  corpus/ghs_query.json   the exact query, the snapshot date and the number of repositories returned
  corpus/repos_ghs.csv    one row per repository (name, default branch, stars, commits, ...)

Usage: python src/01_ghs_query.py
"""
import csv
import datetime as dt
import json
import time
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()  # use the OS trust store; certifi lacks GHS's issuer chain

API = "https://seart-ghs.si.usi.ch/api/r/search"
OUT = Path(__file__).resolve().parent.parent / "corpus"

# The frame: which repositories we mine. Every filter is a GHS search parameter, with the reason
# for its threshold beside it (corpus/README.md and the write-up give more).
QUERY = {
    "language": "Java",
    "license": "MIT License",       # the only license allowed by the assignment
    "excludeForks": "true",         # forks repeat their upstream with small edits; their docs may no longer match the code
    "starsMin": 50,                 # popular repositories: screens out homework and toy projects
    "commitsMin": 1000,             # a long development history
    "contributorsMin": 5,           # several human authors; excludes solo repos written mostly by AI assistants or bots
    "pullsMin": 50,                 # changes are reviewed by others, not blindly merged; enables PR review comments
    "codeLinesMin": 20000,          # Java lines (GHS filters on the main language): mid-sized projects ...
    "codeLinesMax": 80000,          # ... not monorepos mixing many architectural styles
    "commentLinesMin": 1000,        # Java comment lines: English written about code, fundamental to code2comment
    "committedMin": "2025-01-01",   # maintained projects, modern Java idioms
}
PAGE_SIZE = 100

FIELDS = ["name", "defaultBranch", "stargazers", "commits", "contributors", "codeLines", "javaCodeLines", "javaCommentLines", "totalPullRequests",
          "createdAt", "lastCommit", "lastCommitSHA", "license", "isFork"]


def main():
    OUT.mkdir(exist_ok=True)
    # collect every repository, one result page at a time
    rows, page = [], 0
    while True:
        # ask GHS for one page of repositories matching the frame
        r = requests.get(API, params={**QUERY, "page": page, "size": PAGE_SIZE}, timeout=120)
        r.raise_for_status()
        data = r.json()
        for item in data["items"]:
            # GHS reports lines per language; keep the Java totals
            java = [m for m in item["metrics"] if m["language"] == "Java"]
            item["javaCodeLines"] = sum(m["codeLines"] for m in java)
            item["javaCommentLines"] = sum(m["commentLines"] for m in java)
            rows.append({k: item.get(k) for k in FIELDS})
        total = data["totalItems"]
        print(f"page {page}: {len(rows)}/{total}")
        # stop after the last page; otherwise wait briefly and fetch the next
        if data["last"]:
            break
        page += 1
        time.sleep(0.5)

    # sanity checks: every repository arrived, all MIT and none a fork
    assert len(rows) == total, (len(rows), total)
    assert all(r["license"] == "MIT License" and r["isFork"] is False for r in rows)
    snapshot = dt.date.today().isoformat()
    # save the exact query and today's date, so the frame can be reproduced
    (OUT / "ghs_query.json").write_text(json.dumps({
        "endpoint": API, "query": QUERY, "snapshot_date": snapshot, "repositories_returned": total,
    }, indent=2) + "\n")
    # one row per repository, sorted by name
    with open(OUT / "repos_ghs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["name"].lower()))
    print(f"{total} repositories, snapshot {snapshot}")


if __name__ == "__main__":
    main()
