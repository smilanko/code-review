"""Step 7: render every decontaminated record as its final instance text and train the tokenizer.

Instance formats (Step 3):
  method   [/** Javadoc */ ] code with inline comments as /* ... */, on one line
  commit   the message, trailers removed, on one line
  review   <code>diff hunk</code><technical_language>comment comment ...</technical_language>
           (the tags of the code&comment-to-code inputs)

Tokenizer: a new BPE model (Hugging Face tokenizers) trained on our corpus only, 32,000 ids.
We first tried a SentencePiece Unigram model: it kept long identifiers (HashMap) but dropped their
short parts (Map, get, List), so 20% of its tokens were single letters. BPE builds every piece by
merging two smaller ones, so the short common parts are always in the vocabulary.
  * spaces: runs of spaces become one and the ends are stripped (as SentencePiece does), then each
    space becomes "▁" and the text gets a leading "▁"; merges never cross a space;
  * byte fallback: a character outside the learned alphabet becomes its UTF-8 bytes <0x..>,
    never <unk>;
  * ids as in T5: <pad> 0, </s> 1, <unk> 2; then our six tags (kept whole), the 256 bytes and the
    learned pieces up to 31,999; the 100 sentinels take 32,000..32,099 (<extra_id_0> = 32,099);
  * </s> is appended to every input, as in T5.

Output: $DATA/corpus/instances.jsonl.gz (source, repo, text), tokenizer/ (Hugging Face files),
corpus/counts/07_tokenizer.json

Usage: RAYON_NUM_THREADS=4 python src/07_tokenizer.py
"""
import gzip
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DATA", ROOT.parent / "data.nosync"))   # .nosync: keep iCloud out of it
IN, OUT = DATA / "decontaminated", DATA / "corpus"
TOK = ROOT / "tokenizer"
FT = ROOT.parent / "code-review-finetuning-data"
VOCAB = 32000
SPECIAL = ["<pad>", "</s>", "<unk>"]
SYMBOLS = ["<START>", "<END>", "<code>", "</code>", "<technical_language>", "</technical_language>"]
BYTES = [f"<0x{b:02X}>" for b in range(256)]
# T5 numbers its sentinels backwards: <extra_id_99> = 32,000 ... <extra_id_0> = 32,099
SENTINELS = [f"<extra_id_{i}>" for i in range(99, -1, -1)]
PREFIX = {"code-and-comment-to-code": "code&comment2code: ", "code-to-comment": "code2comment: "}
SEED = 680


def one_line(s):
    return " ".join(s.split())


# Glue one cleaned item's separate fields into the single line of text the model trains on.
# Real examples from our data:
#
# A method (fields javadoc + code_c): the Javadoc is wrapped back in /** */ and put first.
#     javadoc  Not supported
#     code_c   public void remove() { /* Not supported */ throw new RuntimeException("..."); }
#     ->       /** Not supported */ public void remove() { /* Not supported */ throw new RuntimeException("..."); }
#
# A commit (field text): the message as it is.
#     ->       Pin release workflow to Maven 3.9.16 for Central publishing
#
# A review thread (fields diff_hunk + comments): the code under review inside <code>...</code>,
# the comments inside <technical_language>...</technical_language>, all on one line.
#     diff_hunk  @@ -0,0 +1,17 @@
#                +<?xml version="1.0" ...?>
#     comments   ["Might want to run this through a SVG optimiser :)", "Fixed in 6d763a6"]
#     ->         <code>@@ -0,0 +1,17 @@ +<?xml version="1.0" ...?></code><technical_language>Might want
#                to run this through a SVG optimiser :) Fixed in 6d763a6</technical_language>
#
# These are the same tags the code&comment2code task uses in its inputs, so pre-training already
# shows the model that layout.
def render(source, r):
    if source == "methods":
        # a */ inside the Javadoc would end the comment early
        doc = one_line(r["javadoc"]).replace("*/", "* /")
        return (f"/** {doc} */ " if doc else "") + r["code_c"]
    if source == "commits":
        return one_line(r["text"])
    comments = " ".join(one_line(c["body"]) for c in r["comments"])
    return f"<code>{one_line(r['diff_hunk'])}</code><technical_language>{comments}</technical_language>"


def pipeline(model):
    """The text handling shared by training and the final tokenizer."""
    t = Tokenizer(model)
    # Step 1, normalize: runs of spaces -> one, strip the ends, then every space -> ▁
    #     "x.close();   // done "  ->  "x.close();▁//▁done"
    # (so double spaces are lost; that is why exact match collapses whitespace)
    t.normalizer = normalizers.Sequence([normalizers.Replace(Regex(" {2,}"), " "), normalizers.Strip(),
                                         normalizers.Replace(" ", "▁")])
    # Step 2, pre-tokenize: cut the text into words at each ▁, adding a ▁ at the very start
    #     "x.close();▁//▁done"  ->  ["▁x.close();", "▁//", "▁done"]
    # BPE then splits each word into pieces; a piece never spans two words.
    t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first", split=True)
    # Decoding (ids -> text): ▁ back to spaces, byte pieces back to characters, drop the leading space
    #     ["▁x", ".close();", "▁//", "▁done"]  ->  "x.close(); // done"
    t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(),
                                   decoders.Strip(" ", 1, 0)])          # drop the leading space
    t.add_special_tokens([AddedToken(s, normalized=False) for s in SPECIAL])
    # Our tags are never split, and are kept when decoding:
    #     "<code>a</code>"  ->  ["<code>", "a", "</code>"]
    t.add_tokens([AddedToken(s, normalized=True, special=False) for s in SYMBOLS])
    return t


def main():
    OUT.mkdir(parents=True, exist_ok=True); TOK.mkdir(exist_ok=True)
    texts, empty = [], {}
    # render every record and save the instance texts for step 8
    with gzip.open(OUT / "instances.jsonl.gz", "wt", encoding="utf-8") as f:
        for source in ("methods", "commits", "reviews"):
            empty[source] = 0
            for line in gzip.open(IN / f"{source}.jsonl.gz", "rt", encoding="utf-8"):
                r = json.loads(line)
                t = render(source, r)
                if t:
                    f.write(json.dumps({"source": source, "repo": r["repo"], "text": t}, ensure_ascii=False) + "\n")
                    texts.append(t)
                else:
                    empty[source] += 1          # e.g. a commit with no message
    print(f"instances rendered: {len(texts)}", flush=True)

    # 1. Learn the BPE merges. BPE starts from single characters and repeatedly merges the most
    #    frequent adjacent pair into a new piece (l + o -> lo, lo + ok -> look, ...), until the
    #    vocabulary is full: 30,739 merges here. Because every piece is built from two smaller ones,
    #    short common parts always exist, so names split at their word parts:
    #        "lookRotate"  ->  ["▁look", "Rotate"]
    #    The vocabulary size given to the trainer leaves room for the 256 byte pieces added below.
    #    Training takes ~10 minutes, so the result is cached in bpe_learned.json (delete it to retrain).
    t = pipeline(models.BPE(unk_token="<unk>"))
    trainer = trainers.BpeTrainer(vocab_size=VOCAB - len(BYTES), special_tokens=SPECIAL + SYMBOLS,
                                  limit_alphabet=1000, show_progress=False)
    learned_path = DATA / "bpe_learned.json"            # the raw training result, kept for re-runs
    if not learned_path.exists():
        t.train_from_iterator(texts, trainer=trainer, length=len(texts))
        learned_path.write_text(t.to_str())
    learned = json.loads(learned_path.read_text())["model"]

    # 2. Final id layout, as T5 expects:
    #        0 <pad>, 1 </s>, 2 <unk>, 3-8 our six tags, 9-264 the 256 bytes <0x00>..<0xFF>,
    #        265-31,999 the learned pieces, 32,000-32,099 the sentinels (added below)
    pieces = SPECIAL + SYMBOLS + BYTES
    pieces += [p for p, _ in sorted(learned["vocab"].items(), key=lambda kv: kv[1]) if p not in SPECIAL + SYMBOLS]
    # every piece must appear once and the total must be exactly 32,000
    dups = sorted(p for p, n in Counter(pieces).items() if n > 1)
    assert len(pieces) == VOCAB and not dups, (len(pieces), dups[:20])
    # Rebuild the BPE model with this id layout and byte fallback on: a character never seen in
    # training becomes its UTF-8 bytes instead of <unk>, so nothing is ever lost:
    #     "漢"  ->  ["▁", "<0xE6>", "<0xBC>", "<0xA2>"]
    merges = [tuple(m.split(" ")) if isinstance(m, str) else tuple(m) for m in learned["merges"]]
    t = pipeline(models.BPE(vocab={p: i for i, p in enumerate(pieces)}, merges=merges,
                            unk_token="<unk>", byte_fallback=True))
    # the 100 sentinels get ids 32,000-32,099
    t.add_special_tokens([AddedToken(s, normalized=False) for s in SENTINELS])
    # Append </s> (id 1) to every input, as T5 does:  "lookRotate" -> [3413, 24889, 1]
    t.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>",
                                                     special_tokens=[("</s>", 1)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="<pad>", eos_token="</s>", unk_token="<unk>",
                                  additional_special_tokens=SENTINELS[::-1])
    # the id layout T5 expects
    assert len(tok) == VOCAB + 100 and tok.convert_tokens_to_ids("<extra_id_0>") == 32099
    assert [tok.convert_tokens_to_ids(s) for s in SPECIAL] == [0, 1, 2]
    tok.save_pretrained(TOK)

    # 3. checks: decoding gives back the text (up to runs of spaces), on the model's real inputs
    #    (task prefix + validation source), the validation targets, and 20,000 corpus instances;
    #    and the share of tokens that are a single letter (the Unigram model's problem)
    check = []
    for task, prefix in PREFIX.items():
        for line in open(FT / task / "val.tsv", encoding="utf-8"):
            src, tgt = line.rstrip("\n").split("\t", 1)
            check += [prefix + src, tgt]
    sample = random.Random(SEED).sample(texts, 20000)
    # Round trip: encode then decode must give back the text (up to whitespace) for all 40,000
    same = sum(tok.decode(tok(x).input_ids, skip_special_tokens=True) == one_line(x) for x in check + sample)
    pieces_sample = [p for x in sample for p in tok.convert_ids_to_tokens(tok(x).input_ids)[:-1]]
    letters = sum(bool(re.fullmatch(r"[A-Za-z]", p)) for p in pieces_sample)
    unk = sum(p == "<unk>" for p in pieces_sample)
    stats = {"instances": len(texts), "empty_removed": empty, "model": "BPE", "vocab": VOCAB,
             "vocab_with_sentinels": len(tok), "merges": len(merges),
             "round_trip_up_to_spaces": f"{same}/{len(check) + len(sample)}",
             "single_letter_token_share_pct": round(100 * letters / len(pieces_sample), 1),
             "unk_tokens_in_sample": unk}
    (ROOT / "corpus/counts/07_tokenizer.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(stats)


if __name__ == "__main__":
    main()
