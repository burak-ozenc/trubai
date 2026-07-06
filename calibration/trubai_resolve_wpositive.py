"""
trubai_resolve_wpositive.py
Resolve W_positive vocabulary token IDs from Moshi SentencePiece tokenizer.

Method: sp.encode(' ' + word) — take second element to avoid leading space
artifact. Confirmed method from SPEC-5BC-v1.

Reports IDs to Muse for confirmation before revised §3 derivation runs.

Usage:
    modal run trubai_resolve_wpositive.py
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

app      = modal.App("trubai-resolve-wpositive", image=image)
hf_cache = modal.Volume.from_name("trubai-hf-cache", create_if_missing=True)
HF_DIR   = Path("/hf-cache")

# W_positive vocabulary — 19 tokens per Muse spec
W_POSITIVE_WORDS = [
    "air", "column", "tone", "breath", "aperture",
    "center", "embouchure", "cracked", "flat", "sharp",
    "breathy", "pitch", "partial", "buzz", "focus",
    "pinched", "diffuse", "spreading", "hollow",
]


@app.function(
    volumes={"/hf-cache": hf_cache},
    timeout=300,
)
def resolve_wpositive():
    import os
    import sentencepiece
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    os.environ["HF_HOME"] = str(HF_DIR)

    tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
    sp = sentencepiece.SentencePieceProcessor(tok_path)
    vocab_size = sp.GetPieceSize()
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")
    print()

    results = []
    print("W_positive token ID resolution:")
    print(f"{'word':15} {'piece':20} {'id':8} {'roundtrip':20} {'ok':4}")
    print("-" * 70)

    for word in W_POSITIVE_WORDS:
        # Encode with leading space — standard SentencePiece subword method
        ids = sp.encode(' ' + word)
        if len(ids) >= 2:
            # Take second element — first is the leading space artifact
            token_id = ids[1]
        elif len(ids) == 1:
            # Single-token encoding including the space
            token_id = ids[0]
        else:
            token_id = None

        if token_id is not None:
            piece     = sp.id_to_piece(token_id)
            roundtrip = sp.id_to_piece(token_id)
            expected  = '▁' + word
            ok        = (roundtrip == expected)
        else:
            piece = roundtrip = "UNRESOLVED"
            ok = False

        print(f"  {word:13}  {piece:18}  {str(token_id):6}  {roundtrip:18}  {'✓' if ok else '✗'}")

        results.append({
            "word":      word,
            "piece":     piece,
            "id":        token_id,
            "roundtrip": roundtrip,
            "verified":  ok,
        })

    print()
    # Also verify known IDs from prior session
    known = [
        ("▁air",           1142),
        ("▁Mr",            2048),
        ("▁significantly", 1117),
    ]
    print("Cross-check against known IDs from §3.4:")
    for piece, known_id in known:
        resolved = sp.piece_to_id(piece)
        match = resolved == known_id
        print(f"  {piece:25}  expected={known_id}  resolved={resolved}  {'✓' if match else '✗ MISMATCH'}")

    print()
    # Summary for Muse
    resolved_ids = [r["id"] for r in results if r["verified"]]
    unverified   = [r for r in results if not r["verified"]]
    print(f"Resolved and verified: {len(resolved_ids)}/{len(W_POSITIVE_WORDS)}")
    if unverified:
        print(f"Unverified ({len(unverified)}):")
        for r in unverified:
            ids_full = sp.encode(' ' + r['word'])
            print(f"  {r['word']}: encode result = {ids_full} → "
                  f"{[sp.id_to_piece(i) for i in ids_full]}")

    return results


@app.local_entrypoint()
def main():
    results = resolve_wpositive.remote()
    print()
    print("W_POSITIVE_IDS for revised derivation:")
    ids = [r["id"] for r in results if r["verified"]]
    print(f"W_POSITIVE_IDS = {ids}")
    print()
    print("Paste this output to Muse for confirmation before §3 revised run.")