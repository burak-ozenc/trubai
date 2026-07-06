"""
upload_lora_v4_data.py
Uploads training data to the trubai-data Modal volume.

Uploads:
  ./data/output/pairs_train.json  → /data/pairs_train.json
  ./data/output/pairs_eval.json   → /data/pairs_eval.json
  ./data/breathiness/*.wav        → /data/audio/*.wav  (Track A)
  ./data/pitch_pairs/*.wav        → /data/audio/*.wav  (Track B)
  ./data/crack_wavs/*.wav         → /data/audio/*.wav  (Track C)
  ./data/output/*.wav             → /data/audio/*.wav  (cracked output)

All .wav files are flattened into /data/audio/ — filenames are unique
across tracks by construction (they encode source + label).

Usage:
    modal run upload_lora_v4_data.py                    # upload everything
    modal run upload_lora_v4_data.py --pairs-only 1     # JSON only, skip audio
    modal run upload_lora_v4_data.py --audio-only 1     # audio only, skip JSON
"""

import modal
from pathlib import Path

app = modal.App("trubai-upload-v4-data")
data_vol = modal.Volume.from_name("trubai-data", create_if_missing=True)

PAIRS_TRAIN = Path("./data/output/pairs_train.json")
PAIRS_EVAL = Path("./data/output/pairs_eval.json")

# All local audio source directories
AUDIO_SOURCES = [
    Path("./data/breathiness"),  # Track A
    Path("./data/pitch_pairs"),  # Track B
    Path("./data/crack_wavs"),  # Track C source
    Path("./data/output"),  # Track C output (cracked wavs)
]


@app.local_entrypoint()
def main(pairs_only: int = 0, audio_only: int = 0) -> None:
    upload_pairs = not bool(audio_only)
    upload_audio = not bool(pairs_only)

    with data_vol.batch_upload(force=True) as batch:

        if upload_pairs:
            for path, dest in [
                (PAIRS_TRAIN, "/data/pairs_train.json"),
                (PAIRS_EVAL, "/data/pairs_eval.json"),
            ]:
                if not path.exists():
                    print(f"ERROR: {path} not found")
                else:
                    batch.put_file(path, dest)
                    print(f"Queued: {path} → {dest}")

        if upload_audio:
            total = 0
            for src_dir in AUDIO_SOURCES:
                if not src_dir.exists():
                    print(f"WARNING: {src_dir} not found — skipping")
                    continue
                wavs = sorted(src_dir.glob("*.wav"))
                for wav in wavs:
                    batch.put_file(wav, f"/data/audio/{wav.name}")
                print(f"Queued {len(wavs):4d} .wav files from {src_dir}")
                total += len(wavs)
            print(f"Total audio queued: {total} files → /data/audio/")

    print()
    print("Upload complete. Verifying...")
    listed = list(data_vol.listdir("/data"))
    print(f"  /data/ contents: {[e.path for e in listed]}")
    try:
        audio_files = list(data_vol.listdir("/data/audio"))
        print(f"  /data/audio/ file count: {len(audio_files)}")
    except Exception:
        print("  /data/audio/ — could not list")
