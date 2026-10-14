# Data card: pre-training corpus

A corpus of Java code and code-review text for pre-training T5 from random weights. The model is
later fine-tuned on two code-review tasks: code-to-comment and code&comment-to-code.

| | |
| --- | --- |
| Instances | 605,550 (599,494 pre-training + 6,056 held out) |
| Tokens | 46.7M under our tokenizer (`</s>` included) |
| Language | Java code; English text |
| Sources | 152 MIT-licensed GitHub repositories |
| Mined | 2026-10-12 |

## Where it comes from

- **Repository selection.** One query to GitHub Search (GHS, https://seart-ghs.si.usi.ch), run on 2026-10-12:
  - Java, MIT license, no forks;
  - at least 50 stars, 1,000 commits, 5 contributors and 50 pull requests;
  - 20k–80k lines of Java code and at least 1k lines of Java comments;
  - last commit on or after 2025-01-01.

  The exact query is in `ghs_query.json`, and the 155 repositories it returned are in `repos_ghs.csv`.
- **Duplicate repositories.** Three of the 155 are the same project listed twice under an old and a new name. Their copies are removed in step 4, which leaves 152 repositories.
- **Repository list with SHAs.** `repos.csv` gives, for each repository, the default branch, the commit (SHA) it was mined at, and how many files, methods and commits it yielded. It includes the 3 duplicate names.
- **Methods and commits** come from a clone of each repository:
  - methods from the Java files at that commit;
  - commit messages from the full history of the default branch.
- **Review threads** come from the GitHub REST API (`/repos/{repo}/pulls/comments`): every inline review comment with the diff hunk it refers to, grouped into threads.

## Instances

Each instance is one line of text, in one of three formats:

| Source | Format |
| --- | --- |
| Method | `/** Javadoc */ ` (if any) + the method, with inline `//` comments written as `/* */` |
| Commit message | the message, with trailers such as `Signed-off-by:` removed |
| Review thread | `<code>diff hunk</code><technical_language>comment comment …</technical_language>` |

The review format uses the same tags as the code&comment-to-code inputs.

## Cleaning, in order

`count_table.md` gives the number removed at each step, per source, and its arithmetic closes.

1. **Deduplication**, per source, across all repositories:
   - exact duplicates: identical token sequence;
   - near duplicates: Jaccard ≥ 0.8 on token 3-grams.

   Methods are compared on code only, and the copy with the most Javadoc and comments is kept.
2. **Filters:**
   - methods: parse errors, generated code, trivial methods without Javadoc;
   - commits: AI-assisted (marked by a trailer or author), bots, reverts and merges, non-English;
   - reviews: bot comments, more than 10 threads per hunk, non-English.

   Non-English means at least 2 letters from a non-Latin script outside code and URLs.
3. **Decontamination** against the validation and test splits of both fine-tuning tasks. An instance is removed if it contains:
   - a method under review;
   - a revised method;
   - a reviewer comment of at least 4 words. Shorter comments ("why?") occur everywhere and can't identify a test instance.
4. **Length:** only instances of 10–512 tokens under our tokenizer are kept, as the assignment requires.

## Splits

A seeded 1% held-out split (seed 680) is used to measure pre-training loss and masked-token accuracy.

| Source | Instances | Share | Tokens | Share |
| --- | ---: | ---: | ---: | ---: |
| Methods | 396,742 | 65.5% | 30.9M | 66.3% |
| Commit messages | 149,924 | 24.8% | 4.1M | 8.9% |
| Review threads | 58,884 | 9.7% | 11.6M | 24.9% |

Length distributions, the tokenizer and three tokenized examples are in `counts/08_length_and_stats.json`, `counts/07_tokenizer.json` and `tokenized_examples.md`.

## Rebuilding it

The corpus files are not in git (about 150 MB, plus 13 GB of clones). The scripts in `src/` rebuild them:

```bash
# from t5-code-review/, with Python 3.14 and the pinned requirements installed
# GITHUB_TOKEN (read-only, public repositories) in the environment or in .env
python src/01_ghs_query.py                    # frame -> corpus/repos_ghs.csv
python src/02_mine_git.py --workers 4 --pin   # clones at the SHAs in repos.csv -> $DATA/raw
python src/03_mine_reviews.py --workers 4     # review comments -> $DATA/reviews
python src/04_dedup.py --workers 4
python src/05_filter.py
python src/06_decontaminate.py --min-comment-words 4
RAYON_NUM_THREADS=4 python src/07_tokenizer.py
python src/08_length_and_stats.py             # -> $DATA/corpus/{train,heldout}.jsonl.gz
python src/09_count_table.py
```

Things to know:
- **Data location.** `$DATA` defaults to `../data.nosync`, next to this repository. The fine-tuning data is expected in `../code-review-finetuning-data`.
- **Step 1 is a live query.** GHS changes over time, so a later run can return different repositories. To reproduce this corpus, skip step 1 and use the committed `repos_ghs.csv`.
- **SHAs.** With `--pin`, step 2 checks out each repository at the SHA in `repos.csv`, so methods and commits are exactly the ones we mined. This was tested on one repository: the output was byte-identical. Without `--pin` it mines the latest commit.
- **Review comments can change.** They are fetched live, so comments deleted or edited after 2026-10-12 will differ.

## Known limitations

- **Short commit messages.** The 10-token minimum removes 137k of them (e.g. "Exclude BlueMapAPI from spotless"), so commits are the source cut most by the length rule.
- **Long reviews.** The 512-token maximum removes 20% of review threads, the ones with long diff hunks. They are dropped, not truncated, because truncation would cut off the comment at the end.
- **Language detection** is script-based. It removes Chinese, Russian and similar text, but not, say, German or Spanish written in Latin letters.
- **AI-written text** is detected only through explicit markers (commit trailers, bot and assistant author names). Unmarked AI-written code or messages remain.
- **Tokenizer.** It splits camelCase names into their parts (`get Aggregate Type`) but glues punctuation onto names (`this.x`, `Type(`).

## License

All code comes from MIT-licensed repositories. Review comments are public GitHub content written by those projects' contributors. The corpus is used only for this course assignment.
