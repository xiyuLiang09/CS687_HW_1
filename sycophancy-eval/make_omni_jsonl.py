#!/usr/bin/env python3
"""
make_omni_jsonl.py — Export an Omni-MATH subset to the custom-JSONL format that
syco_pipeline.py's `--source jsonl` prepare stage consumes.

Filters:
  - difficulty band (default 4.0–6.0: clearly above MATH Level 5, but still in
    the regime where a mid-size model can pass the pass@k reliability filter —
    HFR is only measurable on problems the model solves reliably at T1).
  - answer sanity: short, single, non-proof final answers that math_verify can
    parse (drops multi-part / proof-only / figure-dependent items).

Usage:
  python3 make_omni_jsonl.py --out omni_math_d4_6.jsonl [--dmin 4 --dmax 6]
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

from datasets import load_dataset


def answer_ok(ans: str) -> bool:
    a = str(ans).strip()
    if not a or len(a) > 40:
        return False
    low = a.lower()
    # proof-style / non-answers
    if any(t in low for t in ("proof", "prove", "show that", "\\text{yes", "\\text{no")):
        return False
    # multi-part answers
    if re.search(r"\(\s*(i{1,3}|1|a)\s*\).*\(\s*(i{1,3}|2|b)\s*\)", low):
        return False
    if "\n" in a or ";" in a:
        return False
    return True


def problem_ok(p: str) -> bool:
    low = p.lower()
    # needs a figure we cannot feed the model
    if "[asy]" in low or "as shown" in low or "figure" in low or "diagram" in low:
        return False
    # asks for a proof -> no checkable final answer
    if "prove" in low or "show that" in low:
        return False
    return True


def load_rows(from_file: str):
    """Yield raw Omni-MATH records (dicts) either from HuggingFace or from a
    manually downloaded file (parquet / json / jsonl) — the latter path works
    fully offline."""
    if not from_file:
        for ex in load_dataset("KbsdJames/Omni-MATH", split="test"):
            yield ex
        return
    if from_file.endswith(".parquet"):
        import pandas as pd

        for ex in pd.read_parquet(from_file).to_dict("records"):
            yield ex
    elif from_file.endswith(".jsonl"):
        with open(from_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(from_file, encoding="utf-8") as f:
            for ex in json.load(f):
                yield ex


def parse_quotas(s: str):
    """'6:20,7:30' -> {6: 20, 7: 30}. Bucket = floor(difficulty), so a
    difficulty-6.5 problem counts as difficulty 6."""
    out = {}
    for part in s.split(","):
        k, v = part.split(":")
        out[int(k.strip())] = int(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmin", type=float, default=4.0)
    ap.add_argument("--dmax", type=float, default=6.0)
    ap.add_argument(
        "--quotas",
        default="",
        help='per-difficulty sampling quotas, e.g. "6:20,7:30"; overrides --dmin/--dmax',
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="omni_math_d4_6.jsonl")
    ap.add_argument(
        "--from-file",
        default="",
        help="manually downloaded Omni-MATH parquet/json/jsonl (offline mode)",
    )
    args = ap.parse_args()

    quotas = parse_quotas(args.quotas) if args.quotas else None
    rng = random.Random(args.seed)

    ds = load_rows(args.from_file)
    kept, drop, total = [], Counter(), 0
    avail = Counter()  # usable pool per difficulty, for tuning the dial
    for i, ex in enumerate(ds):
        total += 1
        d = ex.get("difficulty")
        if d is None:
            drop["difficulty"] += 1
            continue
        d = float(d)
        bucket = math.floor(d)
        if quotas:
            if bucket not in quotas:
                drop["difficulty"] += 1
                continue
        elif not (args.dmin <= d <= args.dmax):
            drop["difficulty"] += 1
            continue
        if not problem_ok(ex["problem"]):
            drop["problem_form"] += 1
            continue
        if not answer_ok(ex.get("answer", "")):
            drop["answer_form"] += 1
            continue
        domain = ex.get("domain", "?")
        if not isinstance(domain, str) and hasattr(domain, "__iter__"):
            # parquet round-trips give numpy arrays, not plain lists
            domain = "/".join(map(str, list(domain)))
        avail[bucket] += 1
        kept.append(
            {
                "id": f"omni-{i}",
                "problem": ex["problem"].strip(),
                "answer": str(ex["answer"]).strip(),
                "difficulty": d,
                "domain": str(domain),
                "source": ex.get("source", "?"),
            }
        )

    if quotas:
        by_bucket = defaultdict(list)
        for r in kept:
            by_bucket[math.floor(r["difficulty"])].append(r)
        sampled = []
        for b, q in sorted(quotas.items()):
            pool = by_bucket.get(b, [])
            if len(pool) < q:
                print(
                    f"[omni] WARNING: difficulty {b} pool has only "
                    f"{len(pool)} usable problems (< quota {q}); taking all",
                    file=sys.stderr,
                )
                take = pool
            else:
                take = rng.sample(pool, q)
            sampled.extend(take)
            print(f"[omni] difficulty {b}: sampled {len(take)}/{len(pool)}")
        kept = sampled

    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    sel = f"quotas={quotas}" if quotas else f"band={args.dmin}-{args.dmax}"
    print(f"[omni] kept {len(kept)} / {total}  ({sel}, seed={args.seed})")
    print("[omni] drops:", dict(drop))
    print("[omni] usable pool by difficulty:", dict(sorted(avail.items())))
    by_d = Counter(r["difficulty"] for r in kept)
    print("[omni] kept by difficulty:", dict(sorted(by_d.items())))
    by_dom = Counter(r["domain"] for r in kept)
    print("[omni] top domains:", by_dom.most_common(8))
    with open(args.out + ".dropped_ids", "w") as f:  # provenance for the writeup
        f.write(f"{sel} seed={args.seed} drops={dict(drop)}\n")


if __name__ == "__main__":
    main()
