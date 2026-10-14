"""Step 14: the training curves for the write-up, from the CSV logs of each run.

  results/pretrain_curves.png   pre-training: training loss and held-out loss (left),
                                held-out masked-token accuracy (right)
  results/finetune_curves.png   fine-tuning A (nopt) and B (pt): training loss (left),
                                validation exact match per task (right); drawn once the runs exist
Each figure is also saved as PDF for the LaTeX write-up.

Usage: python src/14_plots.py
"""
import csv
import os
from pathlib import Path

import matplotlib
# draw to files only, no window
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
RUNS = DATA / "checkpoints"
OUT = ROOT / "results"


# a CSV log -> {column: list of numbers}
def read(path):
    rows = list(csv.DictReader(open(path)))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


# save each figure as PNG (to view) and PDF (for the LaTeX write-up)
def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    print(f"results/{name}.png")


def pretrain():
    # left: training and held-out loss; right: held-out masked-token accuracy
    tr, ev = read(RUNS / "pretrain/train_log.csv"), read(RUNS / "pretrain/eval_log.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.plot(tr["step"], tr["loss"], label="training (mean of 50 steps)", lw=1)
    a.plot(ev["step"], ev["heldout_loss"], "o-", label="held-out (1%)", ms=3)
    a.set(xlabel="step", ylabel="loss (cross-entropy, nats)", title="Pre-training loss")
    a.legend()
    b.plot(ev["step"], [100 * v for v in ev["masked_acc"]], "o-", color="C2", ms=3)
    b.set(xlabel="step", ylabel="accuracy (%)", title="Held-out accuracy on masked tokens")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    save(fig, "pretrain_curves")


def finetune():
    # only the runs that have a validation log so far
    runs = {c: RUNS / f"finetune-{c}" for c in ("nopt", "pt") if (RUNS / f"finetune-{c}/val_log.csv").exists()}
    if not runs:
        return
    names = {"nopt": "A: no pre-training", "pt": "B: pre-training"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    # One colour per condition (A blue, B orange). On the right, solid lines are code&comment2code and
# dashed lines code2comment, which stays at 0% for both runs.
    for i, (c, d) in enumerate(runs.items()):
        tr, va = read(d / "train_log.csv"), read(d / "val_log.csv")
        a.plot(tr["step"], tr["loss"], color=f"C{i}", lw=1, label=names[c])
        b.plot(va["step"], [100 * v for v in va["em_code-and-comment-to-code"]], "o-", color=f"C{i}", ms=3,
               label=f"{names[c]}, code&comment2code")
        b.plot(va["step"], [100 * v for v in va["em_code-to-comment"]], "s--", color=f"C{i}", ms=3,
               label=f"{names[c]}, code2comment")
    a.set(xlabel="step", ylabel="loss", title="Fine-tuning training loss")
    b.set(xlabel="step", ylabel="exact match (%)", title="Validation exact match (1,000 per task)")
    a.legend(); b.legend(fontsize=7)
    for ax in (a, b):
        ax.grid(alpha=0.3)
    save(fig, "finetune_curves")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    pretrain()
    finetune()
