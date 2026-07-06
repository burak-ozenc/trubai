"""
TRUBAI — SPEC-PREPROCESS-v1: Pair Formation
=============================================
Builds pairs_track_a.json, pairs_track_b.json, pairs_track_c.json
then merges into pairs_expanded.json.

Usage:
    python pair_formation.py \
        --synthesis_log   /path/to/synthesis_log.json \
        --pitch_log       /path/to/pitch_pairs_log.json \
        --crack_dir       /path/to/crack_wavs \
        --output_dir      /path/to/output \
        --skip_track_c    # optional: skip Track C if files not confirmed yet

Track C file naming convention expected:
    crack_low_C4Bb3_take1.wav
    crack_low_Bb3F3_take1.wav
    crack_mid_G4F4_take1.wav
    crack_mid_C5Bb4_take1.wav
    crack_high_F5D5_take1.wav
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ─── Stratification caps (SPEC-PREPROCESS-v1) ─────────────────────────────────
CAP_LOW  = 180
CAP_HIGH = None   # all 77, no cap
CAP_MID  = 150
RANDOM_SEED = 42

# ─── Silence trim config (Track C) ────────────────────────────────────────────
SILENCE_THRESHOLD_DB = -50.0   # dB below peak = silence
MIN_PRE_SILENCE_SEC  = 0.5
MIN_POST_SILENCE_SEC = 0.5

# ─── Label / inner monologue maps ─────────────────────────────────────────────
TRACK_A_TEMPLATES = {
    "slightly_breathy": {
        "label": "slightly_breathy",
        "inner_monologue": "The tone carries a thin breath layer around the core",
    },
    "breathy": {
        "label": "breathy",
        "inner_monologue": "The air splits around the tone, losing center",
    },
}

TRACK_B_SHIFT_MAP = {
    "flat60c":  {"label": "flat",  "inner_monologue": "The pitch drops severely below center"},
    "flat45c":  {"label": "flat",  "inner_monologue": "The pitch drops noticeably below center"},
    "sharp45c": {"label": "sharp", "inner_monologue": "The pitch pushes noticeably above center"},
    "sharp60c": {"label": "sharp", "inner_monologue": "The pitch pushes hard above center"},
}

TRACK_C_REGISTER_MAP = {
    "low": {
        "label": "cracked",
        "inner_monologue": "The note breaks apart as the embouchure loses its seal",
    },
    "mid": {
        "label": "cracked",
        "inner_monologue": "The attack cracks as the air splits between partials",
    },
    "high": {
        "label": "cracked",
        "inner_monologue": "The partial collapses as the aperture loses its seal",
    },
}

# ─── Track C register inference from filename ─────────────────────────────────
CRACK_REGISTER_PATTERN = re.compile(r"cracked_(low|mid|high)_", re.IGNORECASE)


def detect_register(filename: str) -> str | None:
    """Infer register from Track C filename."""
    m = CRACK_REGISTER_PATTERN.search(filename)
    return m.group(1).lower() if m else None


# ─── Silence trim for Track C ─────────────────────────────────────────────────
def trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Trim leading/trailing silence, preserving at least MIN_PRE/POST_SILENCE_SEC.
    Does not trim the crack event itself — only outer silence.
    """
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio

    threshold = peak * (10 ** (SILENCE_THRESHOLD_DB / 20.0))
    above = np.abs(audio) > threshold

    if not above.any():
        return audio

    first = int(np.argmax(above))
    last  = int(len(above) - 1 - np.argmax(above[::-1]))

    pre_samples  = int(MIN_PRE_SILENCE_SEC  * sr)
    post_samples = int(MIN_POST_SILENCE_SEC * sr)

    start = max(0, first - pre_samples)
    end   = min(len(audio), last + post_samples)

    return audio[start:end]


# ─── Track A ──────────────────────────────────────────────────────────────────
def build_track_a(synthesis_log_path: Path, output_dir: Path) -> list[dict]:
    with open(synthesis_log_path) as f:
        log = json.load(f)

    pairs = []
    skipped = 0

    for record in log:
        if record.get("skipped_reason"):
            skipped += 1
            continue

        source_stem = Path(record["source"]).stem

        for tier in ["slightly_breathy", "breathy"]:
            tier_data = record.get(tier)
            if not tier_data or not tier_data.get("passed"):
                continue

            filename = f"{source_stem}__{tier}.wav"
            template = TRACK_A_TEMPLATES[tier]

            pairs.append({
                "file":            filename,
                "label":           template["label"],
                "inner_monologue": template["inner_monologue"],
            })

    print(f"[Track A] {len(pairs)} pairs built ({skipped} source files skipped by HNR gate)")
    return pairs


# ─── Track B ──────────────────────────────────────────────────────────────────
def build_track_b(pitch_log_path: Path, output_dir: Path) -> list[dict]:
    with open(pitch_log_path) as f:
        log = json.load(f)

    # Separate by register
    by_register: dict[str, list[dict]] = {"Low": [], "High": [], "Mid_sustained": []}
    for record in log:
        if not record.get("shifts_written"):
            continue
        reg = record.get("register")
        if reg in by_register:
            by_register[reg].append(record)

    # Apply caps
    random.seed(RANDOM_SEED)

    low_pool  = by_register["Low"]
    high_pool = by_register["High"]
    mid_pool  = by_register["Mid_sustained"]

    low_selected  = low_pool[:CAP_LOW]   # already sorted, take first CAP_LOW
    high_selected = high_pool            # no cap
    mid_selected  = random.sample(mid_pool, min(CAP_MID, len(mid_pool)))

    print(f"[Track B] Pool sizes  — Low: {len(low_pool)}, High: {len(high_pool)}, Mid: {len(mid_pool)}")
    print(f"[Track B] After caps  — Low: {len(low_selected)}, High: {len(high_selected)}, Mid: {len(mid_selected)}")

    pairs = []

    for group_name, group in [("Low", low_selected), ("High", high_selected), ("Mid", mid_selected)]:
        for record in group:
            stem = Path(record["source"]).stem
            for shift_label, shift_cents in [
                ("flat60c", -60), ("flat45c", -45), ("sharp45c", 45), ("sharp60c", 60)
            ]:
                if shift_cents not in record["shifts_written"] and shift_cents not in [abs(s) * (1 if shift_cents > 0 else -1) for s in record["shifts_written"]]:
                    # verify the shift was actually written
                    if shift_cents not in record["shifts_written"]:
                        continue

                filename = f"{stem}__{shift_label}.wav"
                mapping  = TRACK_B_SHIFT_MAP[shift_label]

                pairs.append({
                    "file":            filename,
                    "label":           mapping["label"],
                    "inner_monologue": mapping["inner_monologue"],
                })

    print(f"[Track B] {len(pairs)} pairs built")
    return pairs


# ─── Track C ──────────────────────────────────────────────────────────────────
def build_track_c(crack_dir: Path, output_dir: Path) -> list[dict]:
    wav_files = sorted(crack_dir.glob("*.wav"))
    if not wav_files:
        print(f"[Track C] No WAV files found in {crack_dir}")
        return []

    pairs = []
    skipped = 0

    for wav_path in wav_files:
        register = detect_register(wav_path.name)
        if register is None:
            print(f"[Track C] WARNING: cannot infer register from '{wav_path.name}' — skipping")
            skipped += 1
            continue

        # Trim silence and write trimmed file to output_dir
        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        trimmed = trim_silence(audio, sr)
        out_path = output_dir / wav_path.name
        sf.write(str(out_path), trimmed, sr, subtype="PCM_24")

        template = TRACK_C_REGISTER_MAP[register]
        pairs.append({
            "file":            wav_path.name,
            "label":           template["label"],
            "inner_monologue": template["inner_monologue"],
        })

    print(f"[Track C] {len(pairs)} pairs built ({skipped} skipped — unrecognized filename pattern)")
    return pairs


# ─── Merge ────────────────────────────────────────────────────────────────────
def merge_and_shuffle(pairs_a: list, pairs_b: list, pairs_c: list, output_dir: Path):
    random.seed(RANDOM_SEED)
    all_pairs = pairs_a + pairs_b + pairs_c
    random.shuffle(all_pairs)

    out_path = output_dir / "pairs_expanded.json"
    with open(out_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Count report
    label_counts: dict[str, int] = {}
    for p in all_pairs:
        label_counts[p["label"]] = label_counts.get(p["label"], 0) + 1

    print("\n─── Pair Count Report ───────────────────────────────────────")
    print(f"  Track A : {len(pairs_a)}")
    print(f"  Track B : {len(pairs_b)}")
    print(f"  Track C : {len(pairs_c)}")
    print(f"  Total   : {len(all_pairs)}")
    print("\n  By label:")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:<20} {count}")
    print(f"\n  Output  : {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TRUBAI SPEC-PREPROCESS-v1 — Pair Formation")
    parser.add_argument("--synthesis_log", required=True,  type=Path)
    parser.add_argument("--pitch_log",     required=True,  type=Path)
    parser.add_argument("--crack_dir",     required=False, type=Path, default=None)
    parser.add_argument("--output_dir",    required=True,  type=Path)
    parser.add_argument("--skip_track_c",  action="store_true",
                        help="Skip Track C — use when crack files not yet confirmed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track A
    pairs_a = build_track_a(args.synthesis_log, args.output_dir)
    with open(args.output_dir / "pairs_track_a.json", "w") as f:
        json.dump(pairs_a, f, indent=2)

    # Track B
    pairs_b = build_track_b(args.pitch_log, args.output_dir)
    with open(args.output_dir / "pairs_track_b.json", "w") as f:
        json.dump(pairs_b, f, indent=2)

    # Track C
    if args.skip_track_c:
        print("[Track C] Skipped — run again with --crack_dir once files are confirmed")
        pairs_c = []
    elif args.crack_dir is None:
        print("[Track C] --crack_dir not provided and --skip_track_c not set. Skipping.")
        pairs_c = []
    else:
        pairs_c = build_track_c(args.crack_dir, args.output_dir)
        with open(args.output_dir / "pairs_track_c.json", "w") as f:
            json.dump(pairs_c, f, indent=2)

    # Merge
    merge_and_shuffle(pairs_a, pairs_b, pairs_c, args.output_dir)


if __name__ == "__main__":
    main()