"""
TRUBAI — Track B: Pitch Variant Pairs
======================================
Source: same flat WAV folder as Track A, 24kHz, 1–5 sec
Output: dedicated pitch_pairs/ directory with ±45c / ±60c variants
        + pitch_pairs_log.json

Priority:
    1 — Low register  (median F0 ≤ 233 Hz, Bb3 and below)
    2 — High register (median F0 ≥ 784 Hz, G5 and above)
    3 — Mid sustained (everything else, duration ≥ 3s, F0 std < 15 cents)

Usage:
    python track_b.py --input_dir /path/to/wavs --output_dir /path/to/out
    python track_b.py --input_dir /path/to/wavs --output_dir /path/to/out --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pyrubberband as pyrb
import soundfile as sf


# ─── Register boundaries ──────────────────────────────────────────────────────
F0_LOW_MAX_HZ  = 233.0   # ≤ Bb3
F0_HIGH_MIN_HZ = 784.0   # ≥ G5

# ─── Priority 3 — sustained tone thresholds ───────────────────────────────────
P3_MIN_DURATION_SEC = 2.0
P3_MAX_F0_STD_CENTS = 20.0

# ─── Pitch shifts ─────────────────────────────────────────────────────────────
SHIFTS_CENTS = [-60, -45, 45, 60]
SHIFT_LABELS = {
    -60: "flat60c",
    -45: "flat45c",
    45: "sharp45c",
    60: "sharp60c",
}

SAMPLE_RATE = 24000

# ─── pyin config (trumpet range) ──────────────────────────────────────────────
FMIN_HZ = 58.0    # Bb1 — below lowest practical trumpet note (safety margin)
FMAX_HZ = 1568.0  # G6


def hz_to_cents(f0_array: np.ndarray, reference_hz: float = 440.0) -> np.ndarray:
    """Convert F0 array in Hz to cents relative to reference."""
    with np.errstate(divide="ignore", invalid="ignore"):
        cents = 1200.0 * np.log2(f0_array / reference_hz)
    return cents


def analyze_f0(audio: np.ndarray, sr: int) -> dict:
    """
    Run pyin, return voiced median F0 (Hz), F0 std (cents), voiced ratio.
    Returns None values if no voiced frames found.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    voiced_f0 = f0[voiced_flag]

    if len(voiced_f0) == 0:
        return {"median_f0_hz": None, "f0_std_cents": None, "voiced_ratio": 0.0}

    cents = hz_to_cents(voiced_f0)
    voiced_ratio = voiced_flag.sum() / len(voiced_flag)

    return {
        "median_f0_hz": float(np.median(voiced_f0)),
        "f0_std_cents": float(np.std(cents)),
        "voiced_ratio": float(voiced_ratio),
    }


def classify(median_f0: float, f0_std_cents: float, duration_sec: float) -> tuple:
    """
    Returns (priority, register_label) or (None, None) if file doesn't qualify.
    """
    if median_f0 is None:
        return None, None

    if median_f0 <= F0_LOW_MAX_HZ:
        return 1, "Low"

    if median_f0 >= F0_HIGH_MIN_HZ:
        return 2, "High"

    # Mid register — only keep sustained tones
    if duration_sec >= P3_MIN_DURATION_SEC and f0_std_cents < P3_MAX_F0_STD_CENTS:
        return 3, "Mid_sustained"

    return None, None


def process_file(wav_path: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    Analyze and pitch-shift a single file.
    Returns a log record dict.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration_sec = len(audio) / sr
    f0_info = analyze_f0(audio, sr)

    record = {
        "source": wav_path.name,
        "duration_sec": round(duration_sec, 3),
        "median_f0_hz": round(f0_info["median_f0_hz"], 2) if f0_info["median_f0_hz"] else None,
        "f0_std_cents": round(f0_info["f0_std_cents"], 2) if f0_info["f0_std_cents"] else None,
        "voiced_ratio": round(f0_info["voiced_ratio"], 3),
        "register": None,
        "priority": None,
        "skipped_reason": None,
        "shifts_written": [],
    }

    priority, register = classify(
        f0_info["median_f0_hz"],
        f0_info["f0_std_cents"],
        duration_sec,
    )

    if priority is None:
        record["skipped_reason"] = (
            "unvoiced" if f0_info["median_f0_hz"] is None
            else "mid-register non-sustained"
        )
        return record

    record["register"] = register
    record["priority"] = priority

    if dry_run:
        record["shifts_written"] = SHIFTS_CENTS  # report what would be written
        return record

    stem = wav_path.stem
    for cents in SHIFTS_CENTS:
        semitones = cents / 100.0
        shifted = pyrb.pitch_shift(audio, sr, semitones)
        out_path = output_dir / f"{stem}__{SHIFT_LABELS[cents]}.wav"
        sf.write(str(out_path), shifted, sr, subtype="PCM_24")
        record["shifts_written"].append(cents)

    return record


def main():
    parser = argparse.ArgumentParser(description="TRUBAI Track B — Pitch Variant Pairs")
    parser.add_argument("--input_dir",  required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--dry_run", action="store_true",
                        help="Process first 5 files only, print stats, write nothing")
    args = parser.parse_args()

    wav_files = sorted(args.input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {args.input_dir}")
        sys.exit(1)

    if args.dry_run:
        wav_files = wav_files[:5]
        print(f"[DRY RUN] Processing {len(wav_files)} files — no output will be written\n")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    log = []
    counts = {"Low": 0, "High": 0, "Mid_sustained": 0, "skipped": 0}

    for i, wav_path in enumerate(wav_files):
        print(f"[{i+1}/{len(wav_files)}] {wav_path.name}", end="  ")
        record = process_file(wav_path, args.output_dir, args.dry_run)
        log.append(record)

        if record["skipped_reason"]:
            counts["skipped"] += 1
            dur  = record["duration_sec"]
            std  = record["f0_std_cents"]
            f0   = record["median_f0_hz"]
            std_str = f"std={std:.1f}¢" if std is not None else "std=N/A"
            f0_str  = f"F0={f0:.1f}Hz" if f0 is not None else "F0=N/A"
            print(f"SKIPPED ({record['skipped_reason']}) | {f0_str} {std_str} dur={dur:.2f}s")
            continue

        reg   = record["register"]
        f0    = record["median_f0_hz"]
        std   = record["f0_std_cents"]
        dur   = record["duration_sec"]
        pri   = record["priority"]
        counts[reg] += 1
        print(f"P{pri} {reg} | F0={f0:.1f}Hz std={std:.1f}¢ dur={dur:.2f}s → {len(record['shifts_written'])} shifts")

    total_qualifying = counts["Low"] + counts["High"] + counts["Mid_sustained"]
    print("\n─── Summary ─────────────────────────────────────────────────")
    print(f"  Total processed  : {len(wav_files)}")
    print(f"  P1 Low register  : {counts['Low']}")
    print(f"  P2 High register : {counts['High']}")
    print(f"  P3 Mid sustained : {counts['Mid_sustained']}")
    print(f"  Skipped          : {counts['skipped']}")
    print(f"  Qualifying files : {total_qualifying}")
    print(f"  Expected output  : {total_qualifying * len(SHIFTS_CENTS)} shifted files")
    print(f"  Expected pairs   : {total_qualifying} (source + 4 shifts each)")

    if not args.dry_run:
        log_path = args.output_dir / "pitch_pairs_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Log written to   : {log_path}")


if __name__ == "__main__":
    main()