"""
trubai_verify_rag_tokens.py
SPEC-RAG-v1 §8 step 3 — Token count verification

Runs sp.encode() against all 9 RAG passages on the actual loaded tokenizer.
Reports token count per passage, flags any exceeding RAG_MAX_TOKENS=12.

CPU-only. No GPU, no LM.

Usage:
    modal run trubai_verify_rag_tokens.py
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

app      = modal.App("trubai-verify-rag-tokens", image=image)
hf_cache = modal.Volume.from_name("trubai-hf-cache", create_if_missing=True)
HF_DIR   = Path("/hf-cache")

RAG_MAX_TOKENS = 12

# SPEC-RAG-v1 §2.3 — passage table, keyed by (pitch_bucket, tone_bucket)
RAG_PASSAGES: dict[tuple[str, str], str] = {
    ("LOW",  "LOW"):  "▁from a faster moving column",
    ("LOW",  "MED"):  "▁before the pitch finds center",
    ("LOW",  "HIGH"): "▁at the base, not at the lip",
    ("MED",  "LOW"):  "▁from a steadier column behind",
    ("MED",  "MED"):  "▁thinner toward the phrase end",
    ("MED",  "HIGH"): "▁but the column loses direction",
    ("HIGH", "LOW"):  "▁a narrower aperture to center",
    ("HIGH", "MED"):  "▁slightly at the phrase end",
    ("HIGH", "HIGH"): "▁centered through the full phrase",
}


@app.function(
    volumes={"/hf-cache": hf_cache},
    timeout=300,
    cpu=2,
)
def verify():
    import os
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    os.environ["HF_HOME"] = str(HF_DIR)

    tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    sp = sentencepiece.SentencePieceProcessor(tok_path)
    print(f"Tokenizer loaded. Vocab size: {sp.GetPieceSize()}")

    print()
    print("=" * 70)
    print("SPEC-RAG-v1 §2.3 — RAG PASSAGE TOKEN COUNT VERIFICATION")
    print(f"RAG_MAX_TOKENS = {RAG_MAX_TOKENS}")
    print("=" * 70)
    print()
    print(f"  {'Bucket':<16}  {'Count':>6}  {'Status':<10}  Tokens")
    print("  " + "-" * 70)

    all_pass    = True
    results     = {}

    for (pitch, tone), passage in RAG_PASSAGES.items():
        ids    = sp.encode(passage)
        pieces = [sp.id_to_piece(i) for i in ids]
        count  = len(ids)
        over   = count > RAG_MAX_TOKENS
        status = "✗ OVER 12" if over else "✓"
        if over:
            all_pass = False
        bucket = f"({pitch}, {tone})"
        print(f"  {bucket:<16}  {count:>6}  {status:<10}  {' '.join(pieces)}")
        results[(pitch, tone)] = {
            "passage": passage,
            "count":   count,
            "ids":     ids,
            "pieces":  pieces,
            "over":    over,
        }

    print()
    print("=" * 70)
    if all_pass:
        print(f"  RESULT: ALL PASS — all 9 passages within {RAG_MAX_TOKENS}-token cap")
    else:
        over_list = [(b, r["count"]) for b, r in results.items() if r["over"]]
        print(f"  RESULT: {len(over_list)} PASSAGE(S) EXCEED {RAG_MAX_TOKENS} TOKENS — FLAG TO MUSE")
        for bucket, count in over_list:
            print(f"    {bucket}: {count} tokens")
    print()
    print("  Full ID table for Muse confirmation:")
    print()
    for (pitch, tone), r in results.items():
        bucket = f"({pitch}, {tone})"
        print(f"  {bucket}:")
        print(f"    passage: {r['passage']}")
        print(f"    count:   {r['count']}")
        print(f"    ids:     {r['ids']}")
        print(f"    pieces:  {r['pieces']}")
    print("=" * 70)

    return results


@app.local_entrypoint()
def main():
    verify.remote()