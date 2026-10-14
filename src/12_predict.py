"""Step 12: greedy predictions of one fine-tuned model, with their confidence scores.

For a condition's selected checkpoint ($DATA/checkpoints/finetune-<condition>/best):
  * test: every instance of both tasks' test sets (5,000 each), used once per final model;
    rows appended to results/predictions.csv (columns as the assignment requires: task,
    condition, input with prefix, expected, predicted, correct, score);
  * validation: the same 1,000-per-task sample used for checkpoint selection, written to
    results/val_predictions-<condition>.csv; the confidence threshold is chosen on it (step 13).

Decoding: greedy (beam width 1), at most 512 new tokens, bf16 as in validation.
Score: the probability the model assigns to the generated sequence, i.e. the product of the
token probabilities along the greedy path, </s> included, NOT length-normalized. It is computed
in a second, fp32 pass that feeds the generated token ids back as labels; the CSV holds it as 0-100.
correct: exact match after collapsing whitespace (11_finetune.normalize).

Usage: python src/12_predict.py --condition nopt [--limit N]
"""
import argparse
import csv
import importlib.util
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")    # cap GPU memory (set before torch)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
# reuse the fine-tuning script's settings and helpers, so decoding is identical
spec = importlib.util.spec_from_file_location("finetune", ROOT / "src/11_finetune.py")
ft = importlib.util.module_from_spec(spec); spec.loader.exec_module(ft)
TASK_NAMES = {"code-and-comment-to-code": "code&comment2code", "code-to-comment": "code2comment"}
COLUMNS = ["task", "condition", "input", "expected", "predicted", "correct", "score"]


@torch.no_grad()
def predict(model, tok, rows, condition):
    """Greedy predictions of rows [(task, source, target)] and the probability of each generated
    token sequence: one fp32 pass with the generated ids as labels; tokens after the first </s>
    (padding) are excluded."""
    model.eval()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))     # similar lengths batch together
    preds, probs = [None] * len(rows), [None] * len(rows)
    for i in range(0, len(order), ft.GEN_BATCH):
        part = order[i:i + ft.GEN_BATCH]
        x = tok([rows[j][1] for j in part], truncation=True, max_length=ft.MAX_LENGTH, padding=True,
                return_tensors="pt").to(ft.DEVICE)
        with torch.autocast(ft.DEVICE, dtype=torch.bfloat16):
            # greedy decoding, exactly as in validation
            seqs = model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=ft.MAX_LENGTH)
        y = seqs[:, 1:]                                                  # drop the decoder start token
        # Positions after the first </s> are padding (a batch is as long as its longest output),
        # not part of the prediction, so they are left out of the score.
        after_eos = torch.cumsum((y == tok.eos_token_id).int(), 1) - (y == tok.eos_token_id).int() > 0
        labels = y.masked_fill(after_eos, -100)
        # score: feed the generated tokens back in, in fp32, and read each token's probability
        logits = model(**x, labels=labels).logits.float()
        logp = torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        # Sequence probability = the product of the token probabilities (computed as a sum of logs),
        # not length-normalized. A real prediction of model B:
        #     "This is not a good practice."  tokens  ▁This ▁is ▁not ▁a ▁good ▁practice . </s>
        #                                     probs   0.10  0.20 0.21 0.13 0.19 0.15    0.32 0.18
        #     product = 0.0000009 (score 0.00009 on the 0-100 scale)
        # Even fairly likely tokens multiply to a tiny number, which is why every code2comment
        # prediction lands in the lowest confidence bucket.
        p = torch.exp((logp * ~after_eos).sum(-1)).tolist()
        for j, text, q in zip(part, tok.batch_decode(seqs, skip_special_tokens=True), p):
            preds[j], probs[j] = text, q
        ft.free_gpu_cache()                                              # varying shapes: see 11_finetune.py
    return [{"task": TASK_NAMES[k], "condition": condition, "input": s, "expected": t, "predicted": pr,
             # the columns the assignment requires; score on a 0-100 scale
             "correct": str(ft.normalize(pr) == ft.normalize(t)).lower(), "score": f"{100 * q:.6g}"}
            for (k, s, t), pr, q in zip(rows, preds, probs)]


# CSV with every field quoted: code contains commas and quotes
def write(path, records, append):
    new = not (append and path.exists())
    with open(path, "a" if append else "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL)
        if new:
            w.writeheader()
        w.writerows(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["nopt", "pt"], required=True)
    ap.add_argument("--checkpoint", help="default: $DATA/checkpoints/finetune-<condition>/best")
    ap.add_argument("--limit", type=int, help="first N instances per split (testing only)")
    args = ap.parse_args()
    ckpt = Path(args.checkpoint or DATA / f"checkpoints/finetune-{args.condition}/best")
    tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")
    # the run's selected (best) checkpoint
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to(ft.DEVICE)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)

    # validation sample: the same seeded draw as in 11_finetune.py
    val_all = ft.read_split("val")
    rng = np.random.default_rng(ft.SEED)
    val = []
    for task in ft.TASKS:
        task_rows = [r for r in val_all if r[0] == task]
        val += [task_rows[i] for i in sorted(rng.choice(len(task_rows), ft.VAL_PER_TASK, replace=False))]
    test = ft.read_split("test")
    # --limit: only the first N per task, for quick tests
    if args.limit:
        val = [r for t in ft.TASKS for r in [x for x in val if x[0] == t][:args.limit]]
        test = [r for t in ft.TASKS for r in [x for x in test if x[0] == t][:args.limit]]

    write(out / f"val_predictions-{args.condition}.csv", predict(model, tok, val, args.condition), append=False)
    print(f"validation: {len(val):,} predictions", flush=True)

    # test: replace this condition's earlier rows, if any, so a re-run never duplicates them
    path = out / "predictions.csv"
    kept = []
    if path.exists():
        kept = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["condition"] != args.condition]
    records = predict(model, tok, test, args.condition)
    write(path, kept + records, append=False)
    for name in TASK_NAMES.values():
        rs = [r for r in records if r["task"] == name]
        print(f"test {name}: exact match {sum(r['correct'] == 'true' for r in rs):,} / {len(rs):,}", flush=True)


if __name__ == "__main__":
    main()
