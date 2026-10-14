# Does pre-training earn its keep? T5 for code review (CSCI 680, Assignment 1)

T5-small is pre-trained from random weights on a corpus we mined ourselves: 152 MIT-licensed Java repositories, giving methods, commit messages and pull-request review threads. We train our own BPE tokenizer and pre-train with span corruption. The model is then fine-tuned on two code-review tasks (code&comment-to-code and code-to-comment), with pre-training (condition B) and without it (condition A). Both are evaluated on the test sets.

## Layout

| Path | Contents |
| --- | --- |
| `README.md` | this file: commands, seeds, requirements |
| `corpus/` | GHS query (`ghs_query.json`), repository list with SHAs (`repos.csv`), count table (`count_table.md`), data card (`README.md`), counts per step (`counts/`), tokenized examples |
| `tokenizer/` | the trained tokenizer (Hugging Face `tokenizer.json`; 32,000 BPE pieces + 100 sentinels) |
| `src/` | mining, cleaning, tokenizer, span corruption, training, prediction, evaluation, plots (numbered in run order) |
| `results/` | `predictions.csv` (every test prediction of both runs), `metrics.md` / `metrics.json`, training curves (`*_curves.png/pdf`), span-corruption examples, error analysis, diagnostic runs |

Large data (clones, corpus files, checkpoints) lives outside the repository in `../data.nosync/`. The `.nosync` suffix keeps iCloud from syncing it. Every script reads it from `$DATA`, which defaults to `../data.nosync`. The fine-tuning data is expected in `../code-review-finetuning-data/`.

## Setup

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.14 -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt   # metrics only
echo "GITHUB_TOKEN=<read-only token for public repositories>" > .env                     # step 3 only
```

There are two environments because `codebleu` 0.7.0 pins tree-sitter 0.22, which breaks the Java parsing used in mining (tree-sitter 0.26).

## Exact commands, from mining to evaluation

Run from this directory, with `.venv/bin/python` unless noted.

```bash
# Part 1: corpus and tokenizer
python src/01_ghs_query.py                     # GHS query -> corpus/repos_ghs.csv (live query; skip to reuse ours)
python src/02_mine_git.py --workers 4 --pin    # clones at the SHAs in corpus/repos.csv; methods + commits
python src/03_mine_reviews.py --workers 4      # PR review comments (GitHub REST API; live, may drift)
python src/04_dedup.py --workers 4             # exact + near-duplicate removal (MinHash/LSH, Jaccard >= 0.8)
python src/05_filter.py                        # quality filters (generated, trivial, bots, AI, non-English, ...)
python src/06_decontaminate.py --min-comment-words 4   # remove validation/test methods and targets
RAYON_NUM_THREADS=4 python src/07_tokenizer.py # render instances; train the BPE tokenizer
python src/08_length_and_stats.py              # 10-512 token rule, 1% held-out split, statistics
python src/09_count_table.py                   # corpus/count_table.md (asserts that every total closes)

# Part 2: pre-training (T5-small from random weights, span corruption)
python src/span_corruption.py                  # writes two real input/target pairs
python src/10_pretrain.py                      # 10,000 steps x 32 x 512, Adafactor (T5 recipe), bf16

# Part 3: multi-task fine-tuning, both conditions with identical settings
python src/11_finetune.py --condition nopt     # A: random weights
python src/11_finetune.py --condition pt       # B: from the pre-trained checkpoint

# Part 4: evaluation
python src/12_predict.py --condition nopt      # greedy test predictions + sequence probabilities
python src/12_predict.py --condition pt
.venv-eval/bin/python src/13_metrics.py --errors pt    # exact match, BLEU-4, CodeBLEU, buckets, McNemar
python src/14_plots.py                         # training curves
```

Every long-running step resumes where it stopped if run again with the same command.

## Seeds and determinism

The seed is **680** everywhere. It governs:
- the corpus shuffle and the 1% held-out split;
- the packing order and span-corruption masks, which depend only on (seed, step), so a resumed run sees the same batches;
- the model initialization;
- the fine-tuning batch order, which is identical for A and B;
- the validation sample (1,000 per task);
- the error-analysis sample.

Decoding is greedy (beam width 1).

## Hardware and compute

All runs used an Apple M5 Max (PyTorch MPS backend), in bf16.

| Run | Steps | GPU-hours |
| --- | ---: | ---: |
| Pre-training | 10,000 | 2.04 |
| Fine-tuning A (no pre-training) | 30,000 (budget reached) | 5.63 |
| Fine-tuning B (pre-training) | 30,000 (budget reached) | 5.71 |
| Test decoding (A and B) | — | 0.42 |

## Checkpoints and corpus

The checkpoints are not in git. They are on the Hugging Face Hub, in private repositories:
- pre-trained model: https://huggingface.co/dmilanko/code-review-t5-pretrain
- condition A (no pre-training), selected checkpoint: https://huggingface.co/dmilanko/code-review-t5-nopt
- condition B (pre-training), selected checkpoint: https://huggingface.co/dmilanko/code-review-t5-pt

Each loads with `T5ForConditionalGeneration.from_pretrained(...)` and `PreTrainedTokenizerFast.from_pretrained(...)`. To score a checkpoint from the Hub, pass `--checkpoint <local folder>` to `src/12_predict.py`. The corpus is rebuilt by steps 1–9 above. `corpus/README.md` (the data card) describes it.
