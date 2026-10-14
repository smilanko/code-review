"""Step 3: mine pull-request review comments of every repository in the frame (GitHub REST API).

Review comments are the inline comments reviewers attach to a diff; each one comes with the diff
hunk it refers to. They are the only source in the corpus written in a reviewer's voice, which is
what code-to-comment has to produce. They are not in git, hence the API.

GET /repos/{repo}/pulls/comments, newest first, every comment (any per-repository cap is applied
later, in 05_filter.py, so that it is counted). Needs GITHUB_TOKEN (read-only,
public repositories) in the environment or in .env. Output: $DATA/reviews/<owner>__<name>.jsonl.gz.

Resumable: a repository is written only once all its pages are fetched; restarting skips finished ones.

Usage: python src/03_mine_reviews.py [--workers 4]
"""
import argparse
import csv
import gzip
import itertools
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
OUT = DATA / "reviews"
KEEP = ["id", "in_reply_to_id", "pull_request_url", "path", "diff_hunk", "body", "created_at"]

_lock = threading.Lock()


def token():
    if "GITHUB_TOKEN" in os.environ:
        return os.environ["GITHUB_TOKEN"]
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("GITHUB_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("GITHUB_TOKEN not set")


SESSION = requests.Session()
SESSION.headers.update({"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28"})


def get(url, params):
    while True:
        r = SESSION.get(url, params=params, timeout=60)
        remaining = int(r.headers.get("X-RateLimit-Remaining", 1))
        if r.status_code in (403, 429) and remaining == 0:
            with _lock:   # one thread sleeps; the others block on the lock
                wait = int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - time.time() + 5
                print(f"rate limit: sleeping {wait:.0f}s", flush=True)
                time.sleep(max(wait, 5))
            continue
        if r.status_code in (403, 429) and "Retry-After" in r.headers:   # secondary rate limit
            time.sleep(int(r.headers["Retry-After"]) + 1)
            continue
        if r.status_code >= 500:
            time.sleep(10)
            continue
        return r


def mine(name):
    slug = name.replace("/", "__")
    out = OUT / f"{slug}.jsonl.gz"
    meta = OUT / f"{slug}.meta.json"
    if out.exists() and meta.exists():
        m = json.loads(meta.read_text())
        return name, m["status"], m["comments"]
    n, status = 0, "ok"
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for page in itertools.count(1):
            r = get(f"https://api.github.com/repos/{name}/pulls/comments",
                    {"sort": "created", "direction": "desc", "per_page": 100, "page": page})
            if r.status_code != 200:
                status = f"http-{r.status_code}"
                break
            items = r.json()
            for c in items:
                user = c.get("user") or {}
                rec = {k: c.get(k) for k in KEEP}
                rec.update(kind="review", repo=name, user=user.get("login"), user_type=user.get("type"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if len(items) < 100:
                break
    tmp.rename(out)
    meta.write_text(json.dumps({"name": name, "status": status, "comments": n}))
    return name, status, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in csv.DictReader(open(ROOT / "corpus/repos_ghs.csv"))]
    for p in OUT.glob("*.tmp"):   # pages of an interrupted repository
        p.unlink()
    total = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(mine, n) for n in names]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status, n = fut.result()
            total += n
            print(f"[{i}/{len(names)}] {name}: {status} {n}", flush=True)
    print(f"review comments: {total}")


if __name__ == "__main__":
    main()
