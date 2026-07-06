"""
phrase_conditioner.py
PhraseConditioner — extended with RAG queue per SPEC-RAG-v1 §3

Queues (in drain order):
  1. _queue        — primary forced tokens [F]   (SPEC-5A-v1, unchanged)
  2. _bridge_queue — bridge token          [B]   (SPEC-5BC-v1, unchanged)
  3. _rag_queue    — RAG passage tokens    [R]   (SPEC-RAG-v1, new)

[D]-window (SPEC-5BA-v1) opens only after is_active → False,
which requires all three queues to be empty. See §3.4.
"""

from __future__ import annotations

import sentencepiece
import torch
from collections import deque
from typing import NamedTuple


# ── PhraseFeatures ─────────────────────────────────────────────────────────────

class PhraseFeatures(NamedTuple):
    pitch_accuracy: float   # normalised autocorrelation [0, 1]
    tone_quality:   float   # clamp((hnr_db - 8.0) / 6.0, 0.0, 1.0)


def phrase_features_from_vectors(fvs) -> PhraseFeatures | None:
    """
    Convert a list of FeatureVectors (from trublib.FeatureExtractor) to
    PhraseFeatures. Returns None if insufficient pitched frames.
    """
    pitched = [fv for fv in fvs if fv.pitch_salience > 0.3]
    if len(pitched) < 5:
        return None
    pitch_acc  = sum(fv.pitch_salience for fv in pitched) / len(pitched)
    hnr_values = [fv.hnr_db for fv in pitched if hasattr(fv, "hnr_db")]
    if not hnr_values:
        return None
    hnr_mean  = sum(hnr_values) / len(hnr_values)
    tone_qual = max(0.0, min(1.0, (hnr_mean - 8.0) / 6.0))
    return PhraseFeatures(pitch_accuracy=pitch_acc, tone_quality=tone_qual)


# ── Bucket helper ──────────────────────────────────────────────────────────────

def _bucket(value: float) -> str:
    if value < 0.33:
        return "LOW"
    if value < 0.67:
        return "MED"
    return "HIGH"


# ── Conditioning table (SPEC-5A-v1, unchanged) ─────────────────────────────────

CONDITIONING_TABLE: dict[tuple[str, str], str] = {
    ("LOW",  "LOW"):  "Your air and tone need support",
    ("LOW",  "MED"):  "The air column wants support",
    ("LOW",  "HIGH"): "Your air support needs opening",
    ("MED",  "LOW"):  "The tone center wants support",
    ("MED",  "MED"):  "Notice how the air feels",
    ("MED",  "HIGH"): "The tone is nearly flowing",
    ("HIGH", "LOW"):  "Good pitch, the tone needs",
    ("HIGH", "MED"):  "Your tone center is spreading",
    ("HIGH", "HIGH"): "The column and tone sounds",
}


# ── Bridge token IDs (SPEC-5BC-v1, unchanged) ──────────────────────────────────
# Bridge token is the first token of the RAG passage context word — the
# word that semantically links the prefix to the continuation.

BRIDGE_TOKEN_IDS: dict[tuple[str, str], int] = {
    ("LOW",  "LOW"):  2368,    # ▁column
    ("LOW",  "MED"):  6396,    # ▁pitch
    ("LOW",  "HIGH"): 16252,   # ▁aperture
    ("MED",  "LOW"):  9064,    # ▁tone
    ("MED",  "MED"):  1142,    # ▁air
    ("MED",  "HIGH"): 1142,    # ▁air
    ("HIGH", "LOW"):  9064,    # ▁tone
    ("HIGH", "MED"):  13369,   # ▁spreading
    ("HIGH", "HIGH"): 2368,    # ▁column
}


# ── RAG passages (SPEC-RAG-v1 §2.3, Muse-approved) ────────────────────────────
# Token counts verified against loaded tokenizer (§8 step 3):
#   (LOW,LOW)=6  (LOW,MED)=6  (LOW,HIGH)=9  (MED,LOW)=8
#   (MED,MED)=6  (MED,HIGH)=6  (HIGH,LOW)=6  (HIGH,MED)=6  (HIGH,HIGH)=6
# Leading ▁ (id=260) retained — grammatical join after bridge token.
# All counts within RAG_MAX_TOKENS=12 cap.

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


# ── PhraseConditioner ──────────────────────────────────────────────────────────

class PhraseConditioner:
    MAX_TOKENS     = 8    # primary queue cap — unchanged
    RAG_MAX_TOKENS = 12   # RAG queue cap — SPEC-RAG-v1 §3.1

    def __init__(self, tokenizer_path: str):
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(tokenizer_path)
        self._queue:        deque[int] = deque()   # primary [F]
        self._bridge_queue: deque[int] = deque()   # bridge  [B]
        self._rag_queue:    deque[int] = deque()   # RAG     [R]

    def prime(self, features: PhraseFeatures) -> None:
        pitch_bucket = _bucket(features.pitch_accuracy)
        tone_bucket  = _bucket(features.tone_quality)
        bucket_pair  = (pitch_bucket, tone_bucket)

        # Primary queue — unchanged
        text = CONDITIONING_TABLE[bucket_pair]
        ids: list[int] = self._sp.encode(text)
        ids = ids[:self.MAX_TOKENS]
        self._queue.clear()
        self._queue.extend(ids)

        # Bridge queue — unchanged
        self._bridge_queue.clear()
        bridge_id = BRIDGE_TOKEN_IDS.get(bucket_pair)
        if bridge_id is not None:
            self._bridge_queue.append(bridge_id)

        # RAG queue — SPEC-RAG-v1 §3.2
        self._rag_queue.clear()
        rag_text = RAG_PASSAGES.get(bucket_pair)
        if rag_text is not None:
            rag_ids = self._sp.encode(rag_text)
            if len(rag_ids) > self.RAG_MAX_TOKENS:
                print(f"WARNING: RAG passage for {bucket_pair} has {len(rag_ids)} tokens "
                      f"(cap={self.RAG_MAX_TOKENS}) — truncating. Report to Muse.")
                rag_ids = rag_ids[:self.RAG_MAX_TOKENS]
            self._rag_queue.extend(rag_ids)
            # DEBUG — confirm rag_ids loaded correctly, remove after verification
            pieces = [self._sp.id_to_piece(i) for i in rag_ids]
            print(f"  [RAG DEBUG] bucket={bucket_pair}  rag_ids={rag_ids}  pieces={pieces}")

    def next_token(
            self, device: torch.device
    ) -> tuple[torch.Tensor | None, str]:
        """
        Drain order: _queue [F] → _bridge_queue [B] → _rag_queue [R] → None ''.
        [D]-window activation fires when this returns (None, '') — i.e. all
        three queues empty. See SPEC-RAG-v1 §3.4.
        """
        if self._queue:
            token_id = self._queue.popleft()
            return torch.tensor([token_id], dtype=torch.long, device=device), '[F]'
        if self._bridge_queue:
            token_id = self._bridge_queue.popleft()
            return torch.tensor([token_id], dtype=torch.long, device=device), '[B]'
        if self._rag_queue:
            token_id = self._rag_queue.popleft()
            return torch.tensor([token_id], dtype=torch.long, device=device), '[R]'
        return None, ''

    @property
    def is_active(self) -> bool:
        """
        True while any queue has tokens remaining.
        [D]-window must not open until this returns False — SPEC-RAG-v1 §3.4.
        """
        return bool(self._queue) or bool(self._bridge_queue) or bool(self._rag_queue)