"""
lora_v4_split.py
SPEC-LORA-v4 §1 — Stratified dataset split

Reads:  ./data/output/pairs_expanded.json
Writes: ./data/output/pairs_train.json
        ./data/output/pairs_eval.json

Also reads ./data/pitch_pairs/pitch_pairs.json to enrich pairs with
`register` field (absent from pairs_expanded.json, required by §2).
Matches on source filename stem — Track B pair filenames encode the
source file name. Track A pairs carry no register field (breathiness
labels, not pitch-based) and are assigned register=None.

Reports split counts by label before writing. Bring table to Muse.

Usage:
    python lora_v4_split.py
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(99)

PAIRS_PATH       = Path("./data/output/pairs_expanded.json")
PITCH_PAIRS_PATH = Path("./data/pitch_pairs/pitch_pairs_log.json")
TRAIN_OUT       = Path("./data/output/pairs_train.json")
EVAL_OUT        = Path("./data/output/pairs_eval.json")

# §1 eval fractions — exactly as specified
EVAL_FRACTIONS = {
    "slightly_breathy": 0.05,
    "breathy":          0.05,
    "flat":             0.08,
    "sharp":            0.08,
    "cracked":          0.20,
}

# ── Load pairs ────────────────────────────────────────────────────────────────
with open(PAIRS_PATH) as f:
    pairs = json.load(f)
print(f"Loaded {len(pairs)} pairs from {PAIRS_PATH}")

# ── Register enrichment ───────────────────────────────────────────────────────
# Build lookup: source stem → register from pitch_pairs.json
# pitch_pairs "register" values: "Mid_sustained", "Low_sustained" etc.
# Normalise to "Low" / "Mid" / "High" to match §2 spec.

def normalise_register(r) -> str | None:
    if r is None:
        return None
    r_lower = r.lower()
    if r_lower.startswith("high"):
        return "High"
    if r_lower.startswith("low"):
        return "Low"
    if r_lower.startswith("mid"):
        return "Mid"
    return None   # unrecognised value — treat as unresolved

register_lookup: dict[str, str] = {}
if PITCH_PAIRS_PATH.exists():
    with open(PITCH_PAIRS_PATH) as f:
        pitch_pairs = json.load(f)
    for pp in pitch_pairs:
        reg = normalise_register(pp.get("register"))
        if reg is None:
            continue   # skipped file or null register — not usable for lookup
        src = Path(pp["source"]).stem
        register_lookup[src] = reg
    print(f"Register lookup built: {len(register_lookup)} source files")
else:
    print(f"WARNING: {PITCH_PAIRS_PATH} not found — register field will be None for all pairs")

def get_register(pair: dict) -> str | None:
    """
    Match pair filename to pitch_pairs source stem.
    Track B filenames encode the source stem before the shift suffix.
    Track A filenames (breathiness) have no pitch register — return None.
    """
    fname = Path(pair["file"]).stem   # strip .wav
    label = pair["label"]

    # Track A: breathy / slightly_breathy — no register
    if label in ("breathy", "slightly_breathy"):
        return None

    # Track C: cracked — filename may encode low/mid/high
    if label == "cracked":
        f_lower = fname.lower()
        if "high" in f_lower:
            return "High"
        if "low" in f_lower:
            return "Low"
        if "mid" in f_lower:
            return "Mid"
        return None

    # Track B: flat / sharp — source stem is everything before the shift part
    # Filename pattern: <source_stem>__<shift>__<label>
    # Try matching against register_lookup by progressively shorter stems
    parts = fname.split("__")
    for i in range(len(parts), 0, -1):
        candidate = "__".join(parts[:i])
        if candidate in register_lookup:
            return register_lookup[candidate]
        # Also try without numeric suffix
        candidate_nonum = candidate.rsplit("__", 1)[0] if "__" in candidate else candidate
        if candidate_nonum in register_lookup:
            return register_lookup[candidate_nonum]

    return None

# Enrich pairs with register field
n_register_assigned = 0
for pair in pairs:
    reg = get_register(pair)
    pair["register"] = reg
    if reg is not None:
        n_register_assigned += 1

print(f"Register field assigned: {n_register_assigned}/{len(pairs)} pairs "
      f"({len(pairs)-n_register_assigned} Track A/unmatched → None)")

# Register distribution for Track B (flat/sharp)
track_b_pairs = [p for p in pairs if p["label"] in ("flat", "sharp")]
reg_dist = defaultdict(int)
for p in track_b_pairs:
    reg_dist[p.get("register", "None")] += 1
print(f"Track B register distribution: {dict(reg_dist)}")

# ── Stratified split ──────────────────────────────────────────────────────────
by_label = defaultdict(list)
for p in pairs:
    by_label[p["label"]].append(p)

eval_set, train_set = [], []

for label, fraction in EVAL_FRACTIONS.items():
    pool = by_label[label]
    random.shuffle(pool)
    n_eval = max(1, int(len(pool) * fraction))
    eval_set.extend(pool[:n_eval])
    train_set.extend(pool[n_eval:])

random.shuffle(train_set)

# ── Report ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SPEC-LORA-v4 §1 — STRATIFIED SPLIT REPORT — BRING TO MUSE")
print("=" * 60)
print()
print(f"  {'Label':<22}  {'Total':>7}  {'Eval':>6}  {'Train':>7}  {'Eval%':>6}")
print("  " + "-" * 55)

total_total = total_eval = total_train = 0
for label in sorted(EVAL_FRACTIONS.keys()):
    pool     = by_label[label]
    n_total  = len(pool)
    fraction = EVAL_FRACTIONS[label]
    n_eval   = max(1, int(n_total * fraction))
    n_train  = n_total - n_eval
    pct      = n_eval / n_total * 100
    total_total += n_total
    total_eval  += n_eval
    total_train += n_train
    print(f"  {label:<22}  {n_total:>7}  {n_eval:>6}  {n_train:>7}  {pct:>5.1f}%")

print("  " + "-" * 55)
print(f"  {'TOTAL':<22}  {total_total:>7}  {total_eval:>6}  {total_train:>7}  "
      f"{total_eval/total_total*100:>5.1f}%")
print()
print(f"  High register pairs in train set: "
      f"{sum(1 for p in train_set if p.get('register') == 'High')}")
print(f"  High register pairs in eval set:  "
      f"{sum(1 for p in eval_set if p.get('register') == 'High')}")
print()
print(f"  Register field absent (Track A / unmatched): "
      f"{sum(1 for p in train_set + eval_set if p.get('register') is None)}")
print("=" * 60)

# ── Write outputs ─────────────────────────────────────────────────────────────
TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_OUT, "w") as f:
    json.dump(train_set, f, indent=2)
with open(EVAL_OUT, "w") as f:
    json.dump(eval_set, f, indent=2)

print()
print(f"Written: {TRAIN_OUT}  ({len(train_set)} pairs)")
print(f"Written: {EVAL_OUT}  ({len(eval_set)} pairs)")