"""Step 11: multi-task fine-tuning on both code-review tasks, under one of two conditions.

  nopt  condition A: starts from random weights (same config, same seed as pre-training)
  pt    condition B: starts from our pre-trained checkpoint ($DATA/checkpoints/pretrain/final)

Everything else is identical for A and B:
  * data: the 50,000 training pairs of each task, each source carrying its task prefix
    ("code&comment2code: " or "code2comment: ", then the source), merged into 100,000 pairs and
    shuffled with a fixed seed per pass, so both runs see the same pairs in the same order;
    batches are length-grouped (sorted by length within groups of 50 batches, then the batch order
    is shuffled), since mixing short comments with long code wasted 60-77% of each batch on padding;
    inputs and targets longer than 512 tokens are truncated (25 inputs and 4 targets of 100,000);
  * budget: batch 32, at most 30,000 optimizer steps, bf16; optimizer: the T5 paper's fine-tuning
    recipe, Adafactor with parameter scaling at a constant learning rate of 1e-3 (no warm-up),
    gradient clipping 1.0 (plain AdamW stalled in pre-training; see 10_pretrain.py);
  * validation every 2,000 steps on a fixed sample of 1,000 instances per task (seeded): greedy
    decoding, exact match after collapsing whitespace; the score is the mean of the two tasks;
    the validation loss (teacher-forced, same instances) breaks ties: a check improves on the best
    if its mean exact match is higher, or equal with a lower loss (exact match alone stays flat,
    e.g. at 0% early in a run from random weights, and would stop a run that is still learning);
  * early stopping: the run stops after 3 checks in a row without improvement; the best checkpoint
    is kept in best/ (Hugging Face format, tokenizer included).

Memory (Apple GPU): batch lengths are padded up to a multiple of 64 and PyTorch's GPU cache is
released every 10 steps and after each validation (a step that still runs out of memory is
retried once on an emptied cache: same batch, same result); without this, the cache kept a block for every
new batch shape and grew past 130 GB. PyTorch's GPU memory is also capped at 45% of RAM.

Logs (in the run directory): train_log.csv (step, loss, lr, hours) every 100 steps; val_log.csv
(step, exact match per task, mean, hours) at every check. "hours" is GPU wall time, checks
included, summed over resumed sessions. last/ holds the latest state; running the same command
again resumes from it.

Output: $DATA/checkpoints/finetune-<condition>/

Usage: python src/11_finetune.py --condition nopt
       python src/11_finetune.py --condition pt
"""
import argparse
import csv
import json
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")    # cap GPU memory (set before torch)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, get_constant_schedule
from transformers.optimization import Adafactor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import MODEL_CONFIG  # noqa: E402

# transformers 5.19 bug: on Apple GPUs (mps) generate() defers its stop check, and the clean-up of that
# shortcut crashes for encoder-decoder models like T5 ("EncoderDecoderCache has no attribute layers").
# Turning the shortcut off gives the ordinary stop check that every other device uses; same outputs.
import transformers.generation.utils as _gen  # noqa: E402
_gen.DeferredStopCheck.is_supported = staticmethod(lambda *args, **kwargs: False)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
FT = ROOT.parent / "code-review-finetuning-data"
TASKS = {"code-and-comment-to-code": "code&comment2code: ", "code-to-comment": "code2comment: "}
SEED = 680
BATCH, MAX_STEPS, LR, CLIP = 32, 30000, 1e-3, 1.0
MAX_LENGTH, VAL_PER_TASK, VAL_EVERY, PATIENCE, LOG_EVERY, GEN_BATCH = 512, 1000, 2000, 3, 100, 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def read_split(split):
    """[(task, prefixed source, target)] of both tasks, in file order."""
    rows = []
    for task, prefix in TASKS.items():
        for line in open(FT / task / f"{split}.tsv", encoding="utf-8"):
            src, tgt = line.rstrip("\n").split("\t", 1)
            rows.append((task, prefix + src, tgt))
    return rows


def normalize(s):
    """Whitespace normalization for exact match: runs of whitespace become one space, ends stripped."""
    return " ".join(s.split())


GROUP = 50                                              # batches per length-sorting group
PAD_MULTIPLE, EMPTY_CACHE_EVERY = 64, 10


def epoch_batches(lengths, epoch, _cache={}):
    """The batches (lists of pair indices) of one pass, length-grouped; depend only on (SEED, epoch)."""
    if epoch not in _cache:
        order = np.random.default_rng([SEED, epoch]).permutation(len(lengths))
        batches = []
        for g in range(0, len(order), GROUP * BATCH):
            group = sorted(order[g:g + GROUP * BATCH], key=lambda j: lengths[j])
            batches += [group[i:i + BATCH] for i in range(0, len(group) - BATCH + 1, BATCH)]
        np.random.default_rng([SEED, epoch, 1]).shuffle(batches)
        _cache.clear(); _cache[epoch] = batches
    return _cache[epoch]


def batch_for(x_ids, y_ids, step):
    """The BATCH training pairs of a given step, padded; depends only on (SEED, step)."""
    lengths = [len(x) + len(y) for x, y in zip(x_ids, y_ids)]
    per_epoch = len(x_ids) // BATCH
    epoch, i = divmod(step, per_epoch)
    idx = epoch_batches(lengths, epoch)[i]
    def pad(seqs, value):
        width = -(-max(map(len, seqs)) // PAD_MULTIPLE) * PAD_MULTIPLE    # round up: few distinct shapes
        out = torch.full((len(seqs), width), value)
        for r, s in enumerate(seqs):
            out[r, :len(s)] = torch.tensor(s)
        return out
    return pad([x_ids[j] for j in idx], 0), pad([y_ids[j] for j in idx], -100)


@torch.no_grad()
def generate(model, tok, sources):
    """Greedy decoding (beam width 1) of every source; returns the decoded predictions in order."""
    model.eval()
    order = sorted(range(len(sources)), key=lambda i: len(sources[i]))   # similar lengths batch together
    preds = [None] * len(sources)
    for i in range(0, len(order), GEN_BATCH):
        part = order[i:i + GEN_BATCH]
        enc = tok([sources[j] for j in part], truncation=True, max_length=MAX_LENGTH, padding=True,
                  return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=MAX_LENGTH)
        for j, text in zip(part, tok.batch_decode(out, skip_special_tokens=True)):
            preds[j] = text
    model.train()
    return preds


@torch.no_grad()
def validation_loss(model, tok, val):
    """Teacher-forced loss on the validation sample: mean over all target tokens."""
    model.eval()
    total = n = 0.0
    for i in range(0, len(val), BATCH):
        part = val[i:i + BATCH]
        x = tok([s for _, s, _ in part], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
        y = tok([t for _, _, t in part], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt").input_ids
        labels = y.masked_fill(y == tok.pad_token_id, -100).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            loss = model(**x, labels=labels).loss
        k = (labels != -100).sum().item()
        total += loss.item() * k; n += k
    model.train()
    return total / n


def validate(model, tok, val):
    """Exact match per task on the validation sample, their mean, and the validation loss."""
    preds = generate(model, tok, [s for _, s, _ in val])
    em = {}
    for task in TASKS:
        hits = [normalize(p) == normalize(t) for p, (k, _, t) in zip(preds, val) if k == task]
        em[task] = sum(hits) / len(hits)
    return em, sum(em.values()) / len(em), validation_loss(model, tok, val)


def better(score, loss, best):
    """Higher mean exact match wins; on a tie, lower validation loss."""
    return score > best[0] or (score == best[0] and loss < best[1])


def free_gpu_cache():
    if DEVICE == "mps":
        torch.mps.empty_cache()


def append_csv(path, row, header):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def save(model, tok, path, state=None):
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp); tok.save_pretrained(tmp)
    if state is not None:
        torch.save(state, tmp / "trainer_state.pt")
    if path.exists():
        shutil.rmtree(path)
    tmp.rename(path)                                    # never leaves a half-written checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["nopt", "pt"], required=True)
    ap.add_argument("--init", default=str(DATA / "checkpoints/pretrain/final"), help="pre-trained model (pt only)")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--run-name")
    args = ap.parse_args()
    run = DATA / "checkpoints" / (args.run_name or f"finetune-{args.condition}")
    run.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")

    train = read_split("train")
    x_ids = tok([s for _, s, _ in train], truncation=True, max_length=MAX_LENGTH).input_ids
    y_ids = tok([t for _, _, t in train], truncation=True, max_length=MAX_LENGTH).input_ids
    val_all = read_split("val")
    rng = np.random.default_rng(SEED)
    val = []
    for task in TASKS:
        rows = [r for r in val_all if r[0] == task]
        val += [rows[i] for i in sorted(rng.choice(len(rows), VAL_PER_TASK, replace=False))]
    print(f"{args.condition}: {len(train):,} training pairs ({len(train) // BATCH:,} steps per pass), "
          f"{len(val):,} validation instances; device {DEVICE}", flush=True)

    torch.manual_seed(SEED)
    if args.condition == "pt":
        model = T5ForConditionalGeneration.from_pretrained(args.init)
        assert {k: getattr(model.config, k) for k in MODEL_CONFIG} == MODEL_CONFIG, "checkpoint config differs"
    else:
        model = T5ForConditionalGeneration(T5Config(**MODEL_CONFIG))
    step, hours, best, bad_checks = 0, 0.0, (-1.0, float("inf")), 0
    last = run / "last"
    if (last / "trainer_state.pt").exists():
        model = T5ForConditionalGeneration.from_pretrained(last)
        state = torch.load(last / "trainer_state.pt", weights_only=False)
        step, hours, best, bad_checks = state["step"], state["hours"], tuple(state["best"]), state["bad_checks"]
        print(f"resumed at step {step:,} (best exact match {best[0]:.4f}, loss {best[1]:.4f}; "
              f"{bad_checks} checks without improvement)", flush=True)
    else:
        state = None
        json.dump({"condition": args.condition, "init": args.init if args.condition == "pt" else "random",
                   "batch": BATCH, "max_steps": args.max_steps, "optimizer": "Adafactor (scale_parameter=True)",
                   "lr": LR, "schedule": "constant", "clip": CLIP, "max_length": MAX_LENGTH, "precision": "bf16",
                   "val_per_task": VAL_PER_TASK, "val_every": VAL_EVERY, "patience": PATIENCE, "seed": SEED,
                   "device": DEVICE}, open(run / "run_config.json", "w"), indent=2)
    model.to(DEVICE).train()
    opt = Adafactor(model.parameters(), lr=LR, scale_parameter=True, relative_step=False, warmup_init=False)
    sched = get_constant_schedule(opt)
    if state:
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])

    t0, running = time.time(), []
    while step < args.max_steps and bad_checks < PATIENCE:
        x, y = batch_for(x_ids, y_ids, step)
        for attempt in (1, 2):
            try:
                with torch.autocast(DEVICE, dtype=torch.bfloat16):
                    loss = model(input_ids=x.to(DEVICE), attention_mask=(x != 0).to(DEVICE), labels=y.to(DEVICE)).loss
                loss.backward()
                break
            except RuntimeError as e:
                if "out of memory" not in str(e) or attempt == 2:
                    raise
                opt.zero_grad(set_to_none=True); loss = None; free_gpu_cache()
                print(f"step {step + 1}: out of memory, retrying on an emptied cache", flush=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        running.append(loss.item())
        del loss
        if step % EMPTY_CACHE_EVERY == 0:
            free_gpu_cache()
        if not np.isfinite(running[-1]):
            raise SystemExit(f"loss is {running[-1]} at step {step}")
        if step % LOG_EVERY == 0:
            now = hours + (time.time() - t0) / 3600
            append_csv(run / "train_log.csv", [step, f"{np.mean(running):.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{now:.4f}"],
                       ["step", "loss", "lr", "hours"])
            print(f"step {step:,}  loss {np.mean(running):.3f}  {now:.2f} h", flush=True)
            running = []
        if step % VAL_EVERY == 0 or step == args.max_steps:
            em, score, vloss = validate(model, tok, val)
            free_gpu_cache()
            now = hours + (time.time() - t0) / 3600
            improved = better(score, vloss, best)
            if improved:
                best, bad_checks = (score, vloss), 0
                save(model, tok, run / "best")
                json.dump({"step": step, "exact_match": em, "mean": score, "val_loss": vloss},
                          open(run / "best" / "val_scores.json", "w"), indent=2)
            else:
                bad_checks += 1
            append_csv(run / "val_log.csv", [step, *(f"{em[t]:.4f}" for t in TASKS), f"{score:.4f}", f"{vloss:.4f}", f"{now:.4f}"],
                       ["step", *(f"em_{t}" for t in TASKS), "mean", "val_loss", "hours"])
            print(f"  validation exact match: " + ", ".join(f"{t} {em[t]:.3f}" for t in TASKS) +
                  f"; mean {score:.3f}; loss {vloss:.3f}{'  (best)' if improved else f'  ({bad_checks} without improvement)'}", flush=True)
            save(model, tok, last, {"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step,
                                    "hours": now, "best": best, "bad_checks": bad_checks})
    hours += (time.time() - t0) / 3600
    reason = "early stopping" if bad_checks >= PATIENCE else "step budget reached"
    json.dump({"stopped_at_step": step, "reason": reason, "best": json.load(open(run / "best" / "val_scores.json")),
               "gpu_hours": round(hours, 3)}, open(run / "summary.json", "w"), indent=2)
    print(f"done: stopped at step {step:,} ({reason}), {hours:.2f} h; best checkpoint in {run / 'best'}", flush=True)


if __name__ == "__main__":
    main()
