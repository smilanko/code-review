"""Step 10: pre-train T5-small from random weights on our corpus with span corruption.

Model: T5-small built from a T5Config (6 + 6 layers, 8 heads, d_model 512, feed-forward 2,048,
our 32,100-token vocabulary), randomly initialized, never from_pretrained. The same config is used
for every run (src/model_config.py).

Data: the training split, tokenized, shuffled (seed 680) and packed into 568-token chunks
(span_corruption.py); each chunk is corrupted anew every time it is drawn: input 512 tokens,
target 114. The held-out split is packed and corrupted once with a fixed seed, so every
evaluation sees the same inputs.

Budget, fixed: STEPS optimizer steps x BATCH chunks x 512 input tokens; bf16 autocast.
Optimizer: the T5 paper's recipe, Adafactor with parameter scaling (each update proportional to the
weight's own RMS) and its built-in rate min(1e-2, 1/sqrt(step)), i.e. 0.01 throughout our 10,000
steps; gradient clipping 1.0. Plain AdamW (lr 5e-4, warm-up + cosine) stalled at the level of a
token-frequency model; four 800-step tests showed that precision (bf16/fp32) and a lower learning
rate did not help and that parameter scaling did (results/diagnostics/).

Logs (in the run directory): train_log.csv (step, loss, lr, hours) every 50 steps; eval_log.csv
(step, held-out loss, masked-token accuracy, hours) every 500 steps. "hours" is training wall time
on the GPU, summed over resumed sessions.

Checkpoints: $DATA/checkpoints/pretrain/step-N every 1,000 steps and final/ at the end, in Hugging
Face format: T5ForConditionalGeneration.from_pretrained(<dir>) loads our own weights (the tokenizer
is saved alongside). The latest checkpoint also holds trainer_state.pt (optimizer, scheduler, step),
so an interrupted run resumes where it stopped: just run the same command again. Data order and
masking are derived from (seed, step), so a resumed run sees the same batches as an unbroken one.

Usage: python src/10_pretrain.py [--steps 10000] [--run-name pretrain]
"""
import argparse
import csv
import gzip
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_config import MODEL_CONFIG  # noqa: E402
from span_corruption import INPUT_LENGTH, TARGET_LENGTH, corrupt, is_masked_token, pack  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
SEED = 680
BATCH, LR, WARMUP, WEIGHT_DECAY, CLIP = 32, 5e-4, 1000, 0.01, 1.0
LOG_EVERY, EVAL_EVERY, SAVE_EVERY, EVAL_BATCH = 50, 500, 1000, 64
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def packed(tok, split, shuffle):
    """Token chunks of one corpus split, cached as $DATA/pretrain_cache/<split>.npy."""
    cache = DATA / "pretrain_cache" / f"{split}.npy"
    # tokenizing 600k instances takes minutes, so the packed chunks are cached on disk
    if cache.exists():
        return np.load(cache)
    texts = [json.loads(l)["text"] for l in gzip.open(DATA / f"corpus/{split}.jsonl.gz", "rt", encoding="utf-8")]
    if shuffle:
        # shuffle before packing, so a chunk mixes instances from different repositories
        np.random.default_rng(SEED).shuffle(texts)
    ids = []
    for i in range(0, len(texts), 5000):
        ids += tok(texts[i:i + 5000]).input_ids
    chunks = pack(ids)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, chunks)
    return chunks


def batch_for(chunks, step):
    """The BATCH chunks of a given step and their corruption; depends only on (SEED, step)."""
    per_epoch = len(chunks) // BATCH
    # Which batch to train on is computed from the step number alone, e.g. with 2,542 steps per pass:
    #     step 5,000  ->  pass 1 (the second), batch 2,458 of that pass
    # Each pass shuffles the chunks with seed (680, pass), and each batch masks them with seed
    # (680, pass, batch). So a run resumed at step 5,000 trains on exactly the batches an
    # uninterrupted run would have, and the results are reproducible.
    epoch, i = divmod(step, per_epoch)
    # a fresh, seeded chunk order on every pass; masks are redrawn every time
    order = np.random.default_rng([SEED, epoch]).permutation(len(chunks))
    rng = np.random.default_rng([SEED, epoch, i])
    pairs = [corrupt(chunks[j].astype(np.int64), rng) for j in order[i * BATCH:(i + 1) * BATCH]]
    return (torch.tensor(np.stack([p[0] for p in pairs])), torch.tensor(np.stack([p[1] for p in pairs])))


@torch.no_grad()
def evaluate(model, inputs, labels, bf16=True):
    """Held-out loss (mean over all target tokens) and accuracy on the hidden tokens."""
    model.eval()
    # Two numbers on the held-out set:
    #   - loss: how surprised the model is by the correct target tokens (lower is better)
    #   - accuracy: of the hidden tokens only (not sentinels or </s>), how often the model's most
    #     likely token is the right one. A model that knew only word frequencies would score ~2%.
    # Evaluated in batches, without gradients; dropout is off in eval mode.
    loss_sum = n_tokens = correct = n_masked = 0
    for i in range(0, len(inputs), EVAL_BATCH):
        x, y = inputs[i:i + EVAL_BATCH].to(DEVICE), labels[i:i + EVAL_BATCH].to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=bf16):
            out = model(input_ids=x, labels=y)
        loss_sum += out.loss.item() * y.numel(); n_tokens += y.numel()
        # accuracy counts only the hidden tokens, not the sentinels or </s>
        m = is_masked_token(y)
        correct += (out.logits.argmax(-1)[m] == y[m]).sum().item(); n_masked += m.sum().item()
    model.train()
    return loss_sum / n_tokens, correct / n_masked


# append one row to a CSV log, writing the header the first time
def append_csv(path, row, header):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


# The optimizer. Plain AdamW stalled: after 2,000 steps it was no better than guessing by token
# frequency. T5's weights start at very different scales (embeddings ~25x larger than attention
# weights), and AdamW moves every weight by about the same amount, so the large ones barely changed.
# Adafactor with parameter scaling moves each weight in proportion to its own size, as in the T5
# paper; it fixed the stall. The other options remain for the diagnostic runs in results/diagnostics/.
def make_optimizer(model, args):
    """(optimizer, schedule). adafactor-t5 is the T5 paper's recipe: Adafactor with parameter scaling
    (each update proportional to the weight's own RMS) and its built-in rate min(1e-2, 1/sqrt(step))."""
    if args.optimizer == "adafactor-t5":
        opt = Adafactor(model.parameters(), lr=None, scale_parameter=True, relative_step=True, warmup_init=False)
        return opt, AdafactorSchedule(opt)
    if args.optimizer == "adafactor":     # as in run_t5_mlm_flax.py: fixed lr, no parameter scaling
        opt = Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    return opt, get_cosine_schedule_with_warmup(opt, args.warmup, args.steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--run-name", default="pretrain")
    # settings for diagnosing runs; the defaults are the declared configuration
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--optimizer", choices=["adamw", "adafactor", "adafactor-t5"], default="adafactor-t5")
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--stop-at", type=int, help="stop after this many steps (schedule still spans --steps)")
    args = ap.parse_args()
    bf16 = args.precision == "bf16"
    stop = min(args.steps, args.stop_at or args.steps)
    run = DATA / "checkpoints" / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")

    # the corpus as 568-token chunks: 81,349 for training, 842 held out
    train = packed(tok, "train", shuffle=True)
    held = packed(tok, "heldout", shuffle=False)
    # The held-out set is masked once (seed 0), so every evaluation sees exactly the same inputs and
    # the points on the curve are comparable.
    rng = np.random.default_rng(0)
    held_pairs = [corrupt(c.astype(np.int64), rng) for c in held]
    held_x = torch.tensor(np.stack([p[0] for p in held_pairs]))
    held_y = torch.tensor(np.stack([p[1] for p in held_pairs]))
    print(f"train chunks {len(train):,} ({len(train) // BATCH:,} steps per pass); held-out chunks {len(held):,}; "
          f"device {DEVICE}", flush=True)

    # T5-small from random weights (seeded), never from_pretrained
    torch.manual_seed(SEED)
    model = T5ForConditionalGeneration(T5Config(**MODEL_CONFIG)).to(DEVICE)
    opt, sched = make_optimizer(model, args)
    step, hours = 0, 0.0
    # resume: if a checkpoint with optimizer state exists, continue from it
    ckpts = sorted(run.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts and (ckpts[-1] / "trainer_state.pt").exists():
        last = ckpts[-1]
        model = T5ForConditionalGeneration.from_pretrained(last).to(DEVICE)
        opt, sched = make_optimizer(model, args)
        state = torch.load(last / "trainer_state.pt", weights_only=False)
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        step, hours = state["step"], state["hours"]
        print(f"resumed from {last.name}", flush=True)
    else:
        json.dump({"steps": args.steps, "batch": BATCH, "input_length": INPUT_LENGTH, "target_length": TARGET_LENGTH,
                   "lr": "min(1e-2, 1/sqrt(step)), scaled by each weight's RMS" if args.optimizer == "adafactor-t5" else args.lr,
                   "warmup": 0 if args.optimizer == "adafactor-t5" else args.warmup, "optimizer": args.optimizer,
                   "weight_decay": WEIGHT_DECAY if args.optimizer == "adamw" else 0, "clip": CLIP,
                   "precision": args.precision, "stop_at": stop,
                   "seed": SEED, "device": DEVICE, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_chunks": len(train), "heldout_chunks": len(held),
                   "input_tokens_seen": args.steps * BATCH * INPUT_LENGTH}, open(run / "run_config.json", "w"), indent=2)
        # step-0 evaluation: the random model's loss and accuracy
        loss, acc = evaluate(model, held_x, held_y, bf16)
        append_csv(run / "eval_log.csv", [0, f"{loss:.4f}", f"{acc:.4f}", 0], ["step", "heldout_loss", "masked_acc", "hours"])

    model.train()
    t0, running = time.time(), []
    while step < stop:
        # one optimizer step: forward in bf16, backward, clip gradients, update
        x, y = batch_for(train, step)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=bf16):
            loss = model(input_ids=x.to(DEVICE), labels=y.to(DEVICE)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        running.append(loss.item())
        # stop at once if the loss becomes NaN or infinite
        if not math.isfinite(running[-1]):
            raise SystemExit(f"loss is {running[-1]} at step {step}")
        now = hours + (time.time() - t0) / 3600
        if step % LOG_EVERY == 0:
            append_csv(run / "train_log.csv", [step, f"{np.mean(running):.4f}", f"{sched.get_last_lr()[0]:.2e}", f"{now:.4f}"],
                       ["step", "loss", "lr", "hours"])
            print(f"step {step:,}  loss {np.mean(running):.3f}  lr {sched.get_last_lr()[0]:.1e}  {now:.2f} h", flush=True)
            running = []
        if step % args.eval_every == 0 or step == stop:
            loss_h, acc = evaluate(model, held_x, held_y, bf16)
            append_csv(run / "eval_log.csv", [step, f"{loss_h:.4f}", f"{acc:.4f}", f"{now:.4f}"],
                       ["step", "heldout_loss", "masked_acc", "hours"])
            print(f"  held-out loss {loss_h:.3f}  masked-token accuracy {acc:.3f}", flush=True)
        # checkpoint every 1,000 steps in Hugging Face format, with the optimizer state
        if step % SAVE_EVERY == 0 or step == stop:
            out = run / f"step-{step}"
            model.save_pretrained(out); tok.save_pretrained(out)
            torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, "hours": now},
                       out / "trainer_state.pt")
            for old in run.glob("step-*"):              # optimizer state only in the latest checkpoint
                if old != out and (old / "trainer_state.pt").exists():
                    (old / "trainer_state.pt").unlink()
    # the model after the last step, used to start condition B
    final = run / "final"
    if final.exists():
        shutil.rmtree(final)
    model.save_pretrained(final); tok.save_pretrained(final)
    print(f"done: {step:,} steps, {hours + (time.time() - t0) / 3600:.2f} h; final model in {final}", flush=True)


if __name__ == "__main__":
    main()
