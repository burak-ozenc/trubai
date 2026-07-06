"""
TRUBAI — Track A: Breathiness Synthesis
========================================
Source: flat WAV folder, 24kHz, 1–5 sec, clean recordings (Group 1)
Output: separate folder with _slightly_breathy and _breathy variants
        + synthesis_log.json

Usage:
    python track_a.py --input_dir /path/to/wavs --output_dir /path/to/out
    python track_a.py --input_dir /path/to/wavs --output_dir /path/to/out --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf
from scipy.signal import butter, sosfilt


# ─── HNR Thresholds ───────────────────────────────────────────────────────────
HNR_SOURCE_MIN = 14.0       # source must be ≥ this to proceed
HNR_SLIGHTLY_MIN = 8.0      # output tier: [8, 14)
HNR_SLIGHTLY_MAX = 14.0
HNR_BREATHY_MAX = 8.0       # output tier: < 8

# ─── Binary search bounds (dB relative to signal RMS) ────────────────────────
# Wide range — algorithm finds the exact level needed per file
MIX_DB_MIN = -30.0   # very quiet noise (almost no effect)
MIX_DB_MAX =   0.0   # noise at full signal RMS (very aggressive)
MIX_SEARCH_TOLERANCE = 0.3  # dB — stop when output HNR is within this of target
MIX_SEARCH_MAX_ITER  = 20

# ─── LPF config ───────────────────────────────────────────────────────────────
LPF_CUTOFF_HZ = 3000
LPF_ORDER     = 6
SAMPLE_RATE   = 24000


def compute_hnr(audio: np.ndarray, sr: int) -> float:
    """Compute Harmonics-to-Noise Ratio via parselmouth (Praat)."""
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    harmonicity = snd.to_harmonicity()
    values = harmonicity.values[harmonicity.values != -200.0]  # -200 = unvoiced marker
    if len(values) == 0:
        return -999.0  # fully unvoiced / silence
    return float(np.mean(values))


def make_shaped_noise(n_samples: int, sr: int) -> np.ndarray:
    """White noise → butter LPF @ 3kHz → RMS-normalized."""
    white = np.random.randn(n_samples)
    sos = butter(LPF_ORDER, LPF_CUTOFF_HZ, btype="low", fs=sr, output="sos")
    filtered = sosfilt(sos, white)
    rms = np.sqrt(np.mean(filtered ** 2))
    if rms < 1e-10:
        return filtered
    return filtered / rms  # RMS = 1.0


def mix_with_noise(signal: np.ndarray, noise: np.ndarray, mix_db: float) -> np.ndarray:
    """
    Mix noise into signal at mix_db relative to signal RMS.
    noise is already RMS-normalized to 1.0.
    """
    signal_rms = np.sqrt(np.mean(signal ** 2))
    scale = signal_rms * (10 ** (mix_db / 20.0))
    mixed = signal + noise * scale
    # Clip to [-1, 1] without hard limiting artifacts
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def find_mix_db(signal: np.ndarray, noise: np.ndarray, sr: int,
                hnr_target_min: float, hnr_target_max: float) -> tuple:
    """
    Binary search for the mix_db that lands output HNR in [hnr_target_min, hnr_target_max).
    Returns (mix_db_found, output_hnr, passed).
    hnr_target_min = -999 means no lower bound (for the breathy tier).
    """
    lo, hi = MIX_DB_MIN, MIX_DB_MAX

    for _ in range(MIX_SEARCH_MAX_ITER):
        mid_db = (lo + hi) / 2.0
        mixed  = mix_with_noise(signal, noise, mid_db)
        hnr    = compute_hnr(mixed, sr)

        if hnr > hnr_target_max:
            # too clean — add more noise (increase mix level toward 0)
            lo = mid_db
        elif hnr_target_min != -999.0 and hnr < hnr_target_min:
            # too noisy — reduce noise
            hi = mid_db
        else:
            # landed in range
            return mid_db, hnr, True

        if (hi - lo) < MIX_SEARCH_TOLERANCE:
            break

    # Final check on best guess
    final_db  = (lo + hi) / 2.0
    mixed     = mix_with_noise(signal, noise, final_db)
    final_hnr = compute_hnr(mixed, sr)
    passed    = (hnr_target_min == -999.0 or final_hnr >= hnr_target_min) and final_hnr < hnr_target_max
    return final_db, final_hnr, passed


def process_file(wav_path: Path, output_dir: Path, dry_run: bool) -> dict:
    """
    Process a single source file.
    Returns a log record dict.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono downmix if stereo

    source_hnr = compute_hnr(audio, sr)

    record = {
        "source": wav_path.name,
        "source_hnr": round(source_hnr, 2),
        "skipped_reason": None,
        "slightly_breathy": None,
        "breathy": None,
    }

    if source_hnr < HNR_SOURCE_MIN:
        record["skipped_reason"] = f"source HNR {source_hnr:.1f} < {HNR_SOURCE_MIN}"
        return record

    noise = make_shaped_noise(len(audio), sr)
    stem = wav_path.stem

    for tier, hnr_min, hnr_max in [
        ("slightly_breathy", HNR_SLIGHTLY_MIN, HNR_SLIGHTLY_MAX),
        ("breathy",          -999.0,           HNR_BREATHY_MAX),
    ]:
        mix_db, output_hnr, passed = find_mix_db(audio, noise, sr, hnr_min, hnr_max)

        record[tier] = {
            "mix_db":     round(mix_db, 2),
            "output_hnr": round(output_hnr, 2),
            "passed":     passed,
        }

        if passed and not dry_run:
            mixed     = mix_with_noise(audio, noise, mix_db)
            out_path  = output_dir / f"{wav_path.stem}__{tier}.wav"
            sf.write(str(out_path), mixed, sr, subtype="PCM_24")

    return record


def main():
    parser = argparse.ArgumentParser(description="TRUBAI Track A — Breathiness Synthesis")
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
    skipped = passed_sb = passed_b = failed_sb = failed_b = 0

    for i, wav_path in enumerate(wav_files):
        print(f"[{i+1}/{len(wav_files)}] {wav_path.name}", end="  ")
        record = process_file(wav_path, args.output_dir, args.dry_run)
        log.append(record)

        if record["skipped_reason"]:
            skipped += 1
            print(f"SKIPPED ({record['skipped_reason']})")
            continue

        sb = record["slightly_breathy"]
        b  = record["breathy"]

        sb_str = f"slightly_breathy HNR={sb['output_hnr']:.1f} mix={sb['mix_db']:.1f}dB {'✓' if sb['passed'] else '✗'}"
        b_str  = f"breathy HNR={b['output_hnr']:.1f} mix={b['mix_db']:.1f}dB {'✓' if b['passed'] else '✗'}"
        print(f"source_HNR={record['source_hnr']:.1f} | {sb_str} | {b_str}")

        if sb["passed"]: passed_sb += 1
        else: failed_sb += 1
        if b["passed"]:  passed_b  += 1
        else: failed_b  += 1

    print("\n─── Summary ───────────────────────────────────")
    print(f"  Total processed : {len(wav_files)}")
    print(f"  Skipped (HNR<14): {skipped}")
    print(f"  slightly_breathy: {passed_sb} passed / {failed_sb} failed")
    print(f"  breathy         : {passed_b} passed  / {failed_b} failed")
    print(f"  Expected output : {passed_sb + passed_b} files")

    if not args.dry_run:
        log_path = args.output_dir / "synthesis_log.json"
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"\n  Log written to  : {log_path}")


if __name__ == "__main__":
    main()