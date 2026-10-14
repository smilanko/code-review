"""Span corruption, T5's pre-training objective, and the packing of our corpus into fixed-length chunks.

Packing: the tokenized instances (each ending in </s>) are joined into one stream and cut into
chunks of RAW_LENGTH tokens, as in T5, so no compute is spent on padding (our median instance is
42 tokens).

Corruption of one chunk: NOISE_DENSITY (15%) of its tokens are hidden, in spans of MEAN_SPAN (3)
tokens on average. Each hidden span is replaced in the input by one sentinel (<extra_id_0>,
<extra_id_1>, ...); the target lists each sentinel followed by the tokens it hides. Example:
    chunk   public int getX ( ) { return x ; }
    input   public int <extra_id_0> ) { return <extra_id_1> }</s>
    target  <extra_id_0> getX (<extra_id_1> x ;</s>
With a 568-token chunk: 85 hidden tokens in 28 spans, so the input is 568 - 85 + 28 + 1 = 512 tokens
and the target 85 + 28 + 1 = 114 (the + 1 is </s>).

random_spans_noise_mask is adapted from Hugging Face's run_t5_mlm_flax.py (Apache 2.0), which
ports the original T5 code; the masking is drawn anew every time a chunk is used.

Run directly, it writes two real input/target pairs from the held-out corpus to
results/span_corruption_examples.md.

Usage: python src/span_corruption.py
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
NOISE_DENSITY, MEAN_SPAN = 0.15, 3.0
RAW_LENGTH, INPUT_LENGTH, TARGET_LENGTH = 568, 512, 114
EOS, SENTINEL_0 = 1, 32099          # <extra_id_0> = 32099, <extra_id_1> = 32098, ...


# Which tokens to hide. For our 568-token chunks: 85 tokens (15%) hidden in 28 spans (mean length 3).
# The 483 kept tokens are also cut into 28 runs, and the two alternate: kept, hidden, kept, hidden ...
# Span lengths are random, so each chunk is masked differently every time it is used.
def random_spans_noise_mask(length, rng, noise_density=NOISE_DENSITY, mean_span=MEAN_SPAN):
    """Boolean mask of `length`: True = hidden. Alternates kept and hidden spans, starting with kept."""
    # how many tokens to hide (15%) and in how many spans (mean length 3)
    num_noise = min(max(int(np.round(length * noise_density)), 1), length - 1)
    num_spans = max(int(np.round(num_noise / mean_span)), 1)
    num_kept = length - num_noise

    def segment(num_items, num_segments):
        """Random split of num_items into num_segments non-empty parts; returns their lengths."""
        # choose num_segments - 1 random cut points among the items
        first = np.arange(num_items - 1) < (num_segments - 1)
        rng.shuffle(first)
        segment_id = np.cumsum(np.pad(first, [[1, 0]]))
        return np.unique(segment_id, return_counts=True)[1]

    # interleave span lengths: kept, hidden, kept, hidden, ...
    spans = np.stack([segment(num_kept, num_spans), segment(num_noise, num_spans)], axis=1).reshape(-1)
    # mark where each span starts; odd-numbered spans are the hidden ones
    starts = np.zeros(length, dtype=np.int8)
    starts[np.cumsum(spans)[:-1]] = 1
    return np.cumsum(starts) % 2 == 1


# Turn a chunk into a training pair. A real example (a 39-token method, 2 spans hidden):
#
#     text    /** Returns the larger of two numbers, ... */ public static int max(int a, int b)
#             { if (a >= b) { return a; } return b; }
#     input   ... public static int max<extra_id_0> { if (a >= b) { return a; } return<extra_id_1></s>
#     target  <extra_id_0>(int a, int b)<extra_id_1> b; }</s>
#
# The model reads the input and must produce the target: each sentinel followed by what it hides.
def corrupt(chunk, rng):
    """One chunk of token ids -> (input_ids, labels), as numpy arrays."""
    mask = random_spans_noise_mask(len(chunk), rng)
    starts = mask & ~np.concatenate([[False], mask[:-1]])        # first token of each hidden span
    # span k gets sentinel <extra_id_k>, whose id is 32099 - k
    sentinel = SENTINEL_0 - (np.cumsum(starts) - 1)              # sentinel of the span a position is in
    # input: kept tokens, and each hidden span collapsed to its sentinel
    keep = ~mask | starts                                        # kept tokens, and one slot per span
    input_ids = np.append(np.where(starts, sentinel, chunk)[keep], EOS)
    # target: each sentinel followed by the tokens it hides
    hidden, first = chunk[mask], starts[mask]
    labels = np.append(np.insert(hidden, np.flatnonzero(first), sentinel[starts]), EOS)
    return input_ids, labels


# Instances are short (median 42 tokens), so they are packed end to end, each ending in </s>, and
# cut into 568-token chunks. No compute is spent on padding, as in T5:
#     [method</s> commit</s> review thread</s> method</s> ...]  ->  rows of 568 tokens
def pack(token_lists, length=RAW_LENGTH):
    """Join token lists into one stream and cut it into rows of `length`; the remainder is dropped."""
    # all token ids end to end, as 16-bit integers (our ids fit below 65,536)
    stream = np.fromiter((t for ids in token_lists for t in ids), dtype=np.uint16)
    n = len(stream) // length
    return stream[:n * length].reshape(n, length)


def is_masked_token(labels):
    """Target positions that hold a hidden token (not a sentinel, </s> or padding):
    the positions of the masked-token accuracy."""
    return (labels > 2) & (labels < 32000)


def main():
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast.from_pretrained(ROOT / "tokenizer")
    texts = [json.loads(l)["text"] for l in gzip.open(DATA / "corpus/heldout.jsonl.gz", "rt", encoding="utf-8")]
    chunks = pack(tok(texts[:200]).input_ids)
    rng = np.random.default_rng(680)
    md = ["# Span corruption: two real input/target pairs\n",
          f"Chunks of {RAW_LENGTH} tokens packed from the held-out corpus; {NOISE_DENSITY:.0%} of tokens hidden "
          f"in spans of mean length {MEAN_SPAN:g}; input {INPUT_LENGTH} tokens, target {TARGET_LENGTH}.\n"]
    for i in (0, 1):
        inp, lab = corrupt(chunks[i].astype(np.int64), rng)
        assert len(inp) == INPUT_LENGTH and len(lab) == TARGET_LENGTH, (len(inp), len(lab))
        md += [f"## Pair {i + 1}\n", "Input:\n", "```\n" + tok.decode(inp) + "\n```\n",
               "Target:\n", "```\n" + tok.decode(lab) + "\n```\n"]
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/span_corruption_examples.md").write_text("\n".join(md))
    print("\n".join(md)[:3000])


if __name__ == "__main__":
    main()
