"""
trubai_build_failure_tokens.py
SPEC-5BA-v1 §1 — Collect failure_tokens.json

Resolves token IDs for all observed failure tokens from:
  1. v13 streaming diagnostic output (primary — most recent failure mode)
  2. Phase 5b-C diagnostic log (supplement — LaTeX/academic register)

Writes: /checkpoints/data/failure_tokens.json

v13 failure mode: ▁Mr cycling + ▁significantly repetition
Phase 5b-C failure mode: LaTeX/academic junk (Gmina, pgfplots, martingale, etc.)
Both sources included — logit bias should suppress both registers.

Usage:
    modal run trubai_build_failure_tokens.py
"""

import modal
import json
from pathlib import Path

MOSHI_FORK_URL = "git+https://github.com/burak-ozenc/moshi.git#subdirectory=moshi"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git"])
    .pip_install([
        "torch==2.6.0",
        MOSHI_FORK_URL,
        "sentencepiece",
        "huggingface_hub",
    ])
)

app      = modal.App("trubai-build-failure-tokens", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache", create_if_missing=True)
HF_DIR   = Path("/hf-cache")
CKPT_DIR = Path("/checkpoints")

# ── Observed failure tokens from v13 streaming diagnostic ─────────────────────
# Source: document index 17 (lora_v3/best streaming verification)
# Free generation after [B] bridge token, phrase 1 (4.5s phrase)
# Counts are empirical from that diagnostic session.
V13_FAILURE_PIECES = [
    # Primary failure register — proper noun cycling + filler adverb
    ("▁Mr",           234),
    ("▁significantly", 128),
    # Low-frequency fragments appearing in first 4 free-gen tokens
    ("ife",             4),
    ("odia",            2),
    ("cul",             2),
    ("per",             4),   # also appears as suffix fragment
    ("s",               4),   # bare suffix fragment
    ("▁Wellington",     2),
    ("▁geometric",      2),
    ("▁regained",       2),
    ("▁spreading",      2),
]

# ── Phase 5b-C failure tokens (LaTeX/academic register) ───────────────────────
# Source: document index 1 (Phase 5b-C streaming log from trubai_streaming_v4.py)
# Counts are approximate from the logged token stream.
PHASE_5BC_FAILURE_PIECES = [
    ("▁Gmina",           80),
    (")!}",              40),
    ("²).",              35),
    ("▁pgfplots",        30),
    ("martingale",       25),
    ("(-\\",             20),
    ("▁HCl",             18),
    ("▁pointwise",       16),
    ("geom",             14),
    ("pgf",              12),
    ("cientificreports", 10),
    ("pgfplots",         10),
    ("filecontents",     10),
    ("rvert",            10),
    ("Interface",         8),
    ("▁xsi",              8),
    ("▁Lebesgue",         8),
    ("≡",                 7),
    ("▁$|\\",             7),
    ("inputenc",          6),
    ("hline",             5),
    ("▁Kähler",           5),
    ("▁Cerambycidae",     4),
    ("¼",                 4),
    ("addplot",           3),
    (")-(",               3),
    ("++)",               3),
    ("═",                 2),
    ("ffmpeg",            2),
    ("Sqrt",              5),
    ("Interval",          2),
    ("tabular",           5),
    ("HCl",               8),
    ("hydride",           2),
    ("Magento",           2),
]


@app.function(
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=300,
)
def build_failure_tokens():
    import os
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    os.environ["HF_HOME"] = str(HF_DIR)

    # Load tokenizer
    tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    sp = sentencepiece.SentencePieceProcessor(tok_path)
    vocab_size = sp.GetPieceSize()
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    def resolve(piece_count_list, source_label):
        entries = []
        unresolved = []
        for piece, count in piece_count_list:
            # Try to get ID — sp.piece_to_id returns 0 (unk) if not found
            token_id = sp.piece_to_id(piece)
            if token_id == 0 and piece != "▁":
                # Unresolved — unk ID, piece not in vocab
                unresolved.append(piece)
                continue
            if token_id >= vocab_size:
                unresolved.append(piece)
                continue
            # Verify round-trip
            roundtrip = sp.id_to_piece(token_id)
            entries.append({
                "id":      token_id,
                "piece":   piece,
                "count":   count,
                "source":  source_label,
                "verified": roundtrip == piece,
            })
        return entries, unresolved

    v13_entries,  v13_unresolved  = resolve(V13_FAILURE_PIECES,       "v13_streaming")
    bc_entries,   bc_unresolved   = resolve(PHASE_5BC_FAILURE_PIECES,  "phase_5bc_streaming")

    # Merge — deduplicate by ID, keep higher count
    merged = {}
    for e in v13_entries + bc_entries:
        tid = e["id"]
        if tid not in merged or e["count"] > merged[tid]["count"]:
            merged[tid] = e

    all_entries = sorted(merged.values(), key=lambda x: -x["count"])

    # Summary report
    total_occurrences = sum(e["count"] for e in all_entries)
    print()
    print("=" * 60)
    print("FAILURE TOKEN RESOLUTION REPORT")
    print("=" * 60)
    print(f"v13 tokens resolved:      {len(v13_entries)}/{len(V13_FAILURE_PIECES)}")
    print(f"Phase 5b-C tokens resolved:{len(bc_entries)}/{len(PHASE_5BC_FAILURE_PIECES)}")
    print(f"Unique failure tokens:    {len(all_entries)}")
    print(f"Total occurrences:        {total_occurrences}")
    print(f"Threshold (§1):           50 — {'PASS' if total_occurrences >= 50 else 'FAIL'}")
    print()

    if v13_unresolved:
        print(f"v13 unresolved pieces ({len(v13_unresolved)}): {v13_unresolved}")
    if bc_unresolved:
        print(f"5b-C unresolved pieces ({len(bc_unresolved)}): {bc_unresolved}")

    print()
    print("Top 15 failure tokens by count:")
    for e in all_entries[:15]:
        verified = "✓" if e["verified"] else "✗"
        print(f"  id={e['id']:6d}  {e['piece']!r:25}  count={e['count']:4d}  "
              f"src={e['source']:20}  {verified}")

    # Build output
    out = {
        "collected_from": ["v13_streaming_diagnostic", "phase_5bc_diagnostic_log"],
        "vocab_size": vocab_size,
        "total_occurrences": total_occurrences,
        "threshold_50_check": "PASS" if total_occurrences >= 50 else "FAIL",
        "note": (
            "v13 failure mode: ▁Mr cycling (234) + ▁significantly (128). "
            "Phase 5b-C failure mode: LaTeX/academic tokens. "
            "Both included — logit bias suppresses both registers."
        ),
        "token_entries": all_entries,
    }

    # Save to /checkpoints/data/
    out_dir = CKPT_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "failure_tokens.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    vol.commit()
    print()
    print(f"Written: {out_path}")
    print("=" * 60)

    return out


@app.local_entrypoint()
def main():
    result = build_failure_tokens.remote()
    print()
    print(f"failure_tokens.json complete.")
    print(f"Unique tokens: {len(result['token_entries'])}")
    print(f"Total occurrences: {result['total_occurrences']}")
    print(f"Threshold check: {result['threshold_50_check']}")