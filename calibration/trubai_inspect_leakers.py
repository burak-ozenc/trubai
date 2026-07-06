"""
trubai_inspect_leakers.py
SPEC-5BA-v1 — LaTeX leak identification

CPU-only. No GPU, no LM, no MERT.
Loads bias_v2.pt and the tokenizer, reports bias values for all
LATEX_TOKEN_IDS entries. Identifies which entries are weak (likely leakers)
and which are strong (working suppressions).

Also cross-references the token streams observed in the alpha=0.5 and
alpha=1.0 calibration runs to flag confirmed leakers vs candidates.

Usage:
    modal run trubai_inspect_leakers.py
"""

import modal
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

app      = modal.App("trubai-inspect-leakers", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

CKPT_DIR = Path("/checkpoints")
HF_DIR   = Path("/hf-cache")
BIAS_V2  = CKPT_DIR / "logit_bias" / "bias_v2.pt"
BIAS_V1  = CKPT_DIR / "logit_bias" / "bias_v1.pt"

# LATEX_TOKEN_IDS — from calibration script (failure register)
LATEX_TOKEN_IDS = {
    17035, 25671, 16274, 27459, 21445, 28935, 26337, 13603,
    24388, 25128, 30599, 21817, 25888, 29026, 2048, 1117,
}

# Tokens confirmed visible in calibration token streams
# (from truncated display — only first ~12 tokens per phrase visible)
# alpha=0.5 streams: ▁USSR, zio, ▁fiscal, tee, ▁centered, loc, ▁hollow,
#                    ▁forests, ▁travelled, ▁intercept, ▁simultaneously,
#                    eye, irri, ▁Hinduism, ▁lump
# alpha=1.0 streams: ▁intercept, ▁buses, isa, ▁travelers, issa, ssel,
#                    tee, ssel, ▁unpleasant, ▁illustrator, irri,
#                    ▁reunited, ▁Mr  ← only confirmed LATEX_TOKEN_ID visible
CONFIRMED_LEAKERS_FROM_STREAM = {
    2048,   # ▁Mr — visible in alpha=1.0 phrase 3 stream
}
# Note: remaining leaking token IDs are not visible in truncated streams.
# This script resolves them by reporting all LATEX_TOKEN_IDS bias values.


@app.function(
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=300,
    cpu=2,
)
def inspect():
    import torch
    import sentencepiece
    import os
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    os.environ["HF_HOME"] = str(HF_DIR)

    # Load tokenizer (CPU)
    tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    sp = sentencepiece.SentencePieceProcessor(tok_path)

    # Load bias vectors
    bias_v2_data = torch.load(str(BIAS_V2), map_location="cpu", weights_only=True)
    bias_v1_data = torch.load(str(BIAS_V1), map_location="cpu", weights_only=True)
    bias_v2 = bias_v2_data["bias_vector"].float()
    bias_v1 = bias_v1_data["bias_vector"].float()

    print()
    print("=" * 70)
    print("LATEX_TOKEN_IDS — bias_v1 vs bias_v2 entries")
    print("Effective suppression at alpha=0.5 and alpha=1.0 shown.")
    print("=" * 70)
    print()

    rows = []
    for tid in sorted(LATEX_TOKEN_IDS):
        piece     = sp.id_to_piece(tid)
        v1_val    = bias_v1[tid].item()
        v2_val    = bias_v2[tid].item()
        eff_05    = v2_val * 0.5
        eff_10    = v2_val * 1.0
        confirmed = "*** CONFIRMED IN STREAM" if tid in CONFIRMED_LEAKERS_FROM_STREAM else ""
        rows.append((tid, piece, v1_val, v2_val, eff_05, eff_10, confirmed))

    # Sort by v2 value ascending (weakest suppression first — most likely leakers)
    rows.sort(key=lambda r: r[3])

    print(f"  {'id':>6}  {'piece':<25}  {'bias_v1':>9}  {'bias_v2':>9}  "
          f"{'eff@0.5':>8}  {'eff@1.0':>8}  note")
    print("  " + "-" * 85)
    for tid, piece, v1_val, v2_val, eff_05, eff_10, confirmed in rows:
        strength = "WEAK" if abs(v2_val) < 2.0 else ("MOD" if abs(v2_val) < 6.0 else "STRONG")
        print(f"  {tid:>6}  {piece:<25}  {v1_val:>+9.4f}  {v2_val:>+9.4f}  "
              f"{eff_05:>+8.4f}  {eff_10:>+8.4f}  {strength} {confirmed}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    weak    = [(tid, piece, v2_val) for tid, piece, v1_val, v2_val, *_ in rows if abs(v2_val) < 2.0]
    mod     = [(tid, piece, v2_val) for tid, piece, v1_val, v2_val, *_ in rows if 2.0 <= abs(v2_val) < 6.0]
    strong  = [(tid, piece, v2_val) for tid, piece, v1_val, v2_val, *_ in rows if abs(v2_val) >= 6.0]

    print(f"  WEAK entries    (|bias_v2| < 2.0)  — probable leakers:  {len(weak)}")
    for tid, piece, v2_val in weak:
        print(f"    id={tid:6d}  {piece:<25}  bias_v2={v2_val:+.4f}  eff@0.5={v2_val*0.5:+.4f}")

    print(f"  MODERATE entries (2.0 ≤ |bias| < 6.0) — marginal:       {len(mod)}")
    for tid, piece, v2_val in mod:
        print(f"    id={tid:6d}  {piece:<25}  bias_v2={v2_val:+.4f}  eff@0.5={v2_val*0.5:+.4f}")

    print(f"  STRONG entries  (|bias_v2| ≥ 6.0)  — working:          {len(strong)}")
    for tid, piece, v2_val in strong:
        print(f"    id={tid:6d}  {piece:<25}  bias_v2={v2_val:+.4f}  eff@1.0={v2_val*1.0:+.4f}")

    print()
    print("Return leaker list to Muse before any patch to bias_v2.pt.")
    print("=" * 70)

    return {
        "weak":   [(tid, piece, v2_val) for tid, piece, v2_val in weak],
        "mod":    [(tid, piece, v2_val) for tid, piece, v2_val in mod],
        "strong": [(tid, piece, v2_val) for tid, piece, v2_val in strong],
    }


@app.local_entrypoint()
def main():
    inspect.remote()