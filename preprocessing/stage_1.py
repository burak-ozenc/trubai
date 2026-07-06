import numpy as np
from dataclasses import dataclass

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

def f0_to_note_and_cents(f0_hz: float) -> tuple[str | None, float, int]:
    """Returns (note_name_with_octave, cents_deviation, midi_number)."""
    if f0_hz <= 0:
        return None, 0.0, -1
    midi_float = 12 * np.log2(f0_hz / 440.0) + 69
    midi_nearest = int(round(midi_float))
    cents = (midi_float - midi_nearest) * 100.0
    note_name = NOTE_NAMES[midi_nearest % 12]
    octave = midi_nearest // 12 - 1
    return f"{note_name}{octave}", cents, midi_nearest

def midi_to_register(midi: int) -> str:
    """
    low    = below E3  (midi < 52)
    middle = E3–C5    (52 <= midi <= 72)
    upper  = above C5 (midi > 72)
    """
    if midi < 52:
        return "low"
    elif midi <= 72:
        return "middle"
    else:
        return "upper"

def format_pitch_accuracy(label: str, cents: float) -> str:
    """
    Pre-format pitch accuracy as a natural language fragment for the LLM.
    Scholar: fold cents into the string, not a separate numeric field.
    """
    if label == "cracked":
        return "cracked"
    elif label == "in_tune":
        return "in tune"
    elif label == "flat":
        return f"flat by {abs(round(cents))} cents"
    elif label == "sharp":
        return f"sharp by {abs(round(cents))} cents"
    return label


@dataclass
class PhraseObservation:
    """
    Final schema — signed off by Scholar.
    Three primitives only: pitch accuracy, tone quality, consistency.
    Plus register and note for LLM context.
    """
    scale: str                   # e.g. "C major ascending" — context for LLM only
    note: str | None             # most common note in phrase e.g. "G4"
    register: str                # "low" | "middle" | "upper"
    pitch_accuracy: str          # pre-formatted: "flat by 28 cents" / "in tune" / "cracked"
    tone_quality: str            # "open" | "pinched" | "breathy"
    consistency: str             # "steady" | "rushing" | "dragging"

    # ── Raw values for debugging/logging only — not passed to LLM ─────
    _cents_raw: float = 0.0
    _hnr_db_median: float = 0.0
    _flux_variance: float = 0.0

    def to_llm_dict(self) -> dict:
        """Returns only what the LLM should see — no raw numeric fields."""
        return {
            "scale": self.scale,
            "note": self.note,
            "register": self.register,
            "pitch_accuracy": self.pitch_accuracy,
            "tone_quality": self.tone_quality,
            "consistency": self.consistency,
        }


def observe_phrase(
        feature_vectors: list,
        scale_name: str = "unknown scale",
) -> PhraseObservation | None:
    """
    Aggregate FeatureVector list from one phrase → PhraseObservation.
    Returns None if fewer than 5 pitched frames (too short to assess).

    Known limitation (Scholar-acknowledged):
    - consistency uses f0 drift as rushing/dragging proxy
    - will mislabel ascending phrases with intonation drift as "rushing"
    - acceptable for first training run, fix in v2 with onset interval variance
    - flux_variance branch removed per Scholar (measures spectral not temporal)
    """
    # ── Filter to pitched frames only ─────────────────────────────────
    pitched = [
        fv for fv in feature_vectors
        if fv.pitch_salience >= 0.35 and fv.f0_hz > 0
    ]

    if len(pitched) < 5:
        return None

    # ================================================================
    # Primitive 1: Pitch accuracy
    # ================================================================
    cents_list = []
    note_list = []
    midi_list = []

    for fv in pitched:
        note, cents, midi = f0_to_note_and_cents(fv.f0_hz)
        if note:
            cents_list.append(cents)
            note_list.append(note)
            midi_list.append(midi)

    median_cents = float(np.median(cents_list)) if cents_list else 0.0
    most_common_note = max(set(note_list), key=note_list.count) if note_list else None
    median_midi = int(np.median(midi_list)) if midi_list else 60

    # Cracked: sustained low pitch salience (> 30% of frames below threshold)
    low_salience_ratio = sum(
        1 for fv in pitched if fv.pitch_salience < 0.4
    ) / len(pitched)

    if low_salience_ratio > 0.3:
        pitch_label = "cracked"
    elif median_cents < -20:
        pitch_label = "flat"
    elif median_cents > 20:
        pitch_label = "sharp"
    else:
        pitch_label = "in_tune"

    pitch_accuracy_str = format_pitch_accuracy(pitch_label, median_cents)

    # ================================================================
    # Primitive 2: Tone quality
    # ================================================================
    hnr_values = [fv.hnr_db for fv in pitched]
    centroid_values = [fv.spectral_centroid for fv in pitched]
    f0_values = [fv.f0_hz for fv in pitched]

    median_hnr = float(np.median(hnr_values))
    median_centroid = float(np.median(centroid_values))
    median_f0 = float(np.median(f0_values))

    # Brightness ratio: pinched = high centroid relative to f0
    # Normal trumpet: centroid ~3–6x f0
    brightness_ratio = median_centroid / (median_f0 + 1e-6)

    if median_hnr < 8:
        tone_quality = "breathy"
    elif brightness_ratio > 8.0:
        tone_quality = "pinched"
    else:
        tone_quality = "open"

    # ================================================================
    # Primitive 3: Consistency
    # ================================================================
    # f0 drift proxy — Scholar acknowledged limitation for ascending scales
    # flux_variance branch removed per Scholar review
    third = max(1, len(f0_values) // 3)
    f0_start = float(np.mean(f0_values[:third]))
    f0_end = float(np.mean(f0_values[-third:]))
    f0_drift_cents = np.log2(f0_end / (f0_start + 1e-6)) * 1200

    if f0_drift_cents > 15:
        consistency = "rushing"
    elif f0_drift_cents < -15:
        consistency = "dragging"
    else:
        consistency = "steady"

    flux_values = [fv.spectral_flux for fv in pitched]

    return PhraseObservation(
        scale=scale_name,
        note=most_common_note,
        register=midi_to_register(median_midi),
        pitch_accuracy=pitch_accuracy_str,
        tone_quality=tone_quality,
        consistency=consistency,
        _cents_raw=round(median_cents, 1),
        _hnr_db_median=round(median_hnr, 1),
        _flux_variance=round(float(np.var(flux_values)), 6),
    )