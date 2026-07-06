"""
Option 2a — Retrieval-Based Logit Bias
=======================================
No training required. Uses cosine similarity over 22 pre-computed MERT embeddings
to build a vocabulary frequency vector, applied as a decaying logit bias via
LMGen's on_text_logits_hook.

Injection path: on_text_logits_hook fires after CUDA graph, before sample_token.
text_logits shape: [B=1, 1, 1, 32000]. In-place .add_() modifies what sample_token sees.

Scholar's logging requirements are all implemented:
  - cosine similarities at phrase boundary
  - which training pairs are nearest neighbors
  - top-10 logit bias tokens being applied
"""

import json
import math
import numpy as np
import torch
import io
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants (must match training / streaming setup)
# ─────────────────────────────────────────────────────────────────────────────
MOSHI_SR     = 24000
MERT_DIM     = 768
MOSHI_DIM    = 4096
FRAME        = 1920
RMS_THRESH   = 0.01
SILENCE_SECS = 0.6

BIAS_SCALE_INIT  = 3.0
BIAS_SCALE_DECAY = 0.7
SIM_THRESHOLD_OK = 0.6   # above this → retrieval is working
SIM_THRESHOLD_WARN = 0.3 # below this → flag as unreliable, log warning

TOP_K_NEIGHBORS = 3       # number of nearest neighbors to aggregate


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval index: built once at session start from Modal volume data
# ─────────────────────────────────────────────────────────────────────────────

class RetrievalIndex:
    """
    Loads 22 training pairs from JSONL + pre-computed embeddings from Modal volume.
    Provides cosine-similarity nearest-neighbor lookup and bias vector construction.
    """

    def __init__(
            self,
            jsonl_path: str,
            embeddings_dir: str,
            text_tokenizer,         # sentencepiece.SentencePieceProcessor
            device: str = "cuda",
    ):
        self.text_tokenizer = text_tokenizer
        self.device = device
        self.pairs = []

        # Load JSONL — embedding_paths need remapping from /kaggle to /checkpoints
        with open(jsonl_path) as f:
            for line in f:
                record = json.loads(line.strip())
                self.pairs.append(record)

        vocab_size = text_tokenizer.get_piece_size()  # 32000

        # Build embedding matrix [N, 768] and vocab frequency matrix [N, 32000]
        N = len(self.pairs)
        self.embeddings = torch.zeros(N, MERT_DIM, dtype=torch.float32)
        self.vocab_freq  = torch.zeros(N, vocab_size, dtype=torch.float32)

        for i, record in enumerate(self.pairs):
            # Remap path: replace whatever prefix to embeddings_dir
            orig_path = Path(record["embedding_path"])
            emb_path  = Path(embeddings_dir) / orig_path.name
            emb = torch.load(str(emb_path), map_location="cpu", weights_only=False)
            # Shape: [1, 1, 768] → flatten to [768]
            self.embeddings[i] = emb.view(-1)

            # Tokenize inner_monologue → count token frequencies
            monologue = record["inner_monologue"]
            token_ids = text_tokenizer.encode(monologue)
            for tid in token_ids:
                if 0 <= tid < vocab_size:
                    self.vocab_freq[i, tid] += 1.0
            # Normalize to frequency (not count) so longer monologues don't dominate
            total = self.vocab_freq[i].sum()
            if total > 0:
                self.vocab_freq[i] /= total

        # L2-normalize embeddings for cosine similarity via dot product
        norms = self.embeddings.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.embeddings_normed = self.embeddings / norms

        self.embeddings_normed = self.embeddings_normed.to(device)
        self.vocab_freq         = self.vocab_freq.to(device)

        print(f"[RetrievalIndex] Loaded {N} training pairs.")
        print(f"[RetrievalIndex] Embedding matrix: {self.embeddings_normed.shape}")
        print(f"[RetrievalIndex] Vocab freq matrix: {self.vocab_freq.shape}")

    def query(
            self,
            query_embedding: torch.Tensor,  # [768] float32 on any device
            top_k: int = TOP_K_NEIGHBORS,
    ) -> dict:
        """
        Returns:
            {
                "bias_vector": Tensor [32000] on self.device,
                "similarities": list of floats (top_k),
                "neighbor_indices": list of ints (top_k),
                "neighbor_labels": list of dicts (tone_quality, note, register),
                "top10_tokens": list of (piece_str, weight) tuples,
                "mean_similarity": float,
            }
        """
        # Normalize query
        q = query_embedding.float().to(self.device).view(-1)
        q = q / q.norm().clamp(min=1e-8)

        # Cosine similarities [N]
        sims = (self.embeddings_normed @ q)  # dot product of normed vectors

        # Top-k
        top_sims, top_idxs = torch.topk(sims, k=min(top_k, len(self.pairs)))
        top_sims  = top_sims.cpu().tolist()
        top_idxs  = top_idxs.cpu().tolist()

        # Weighted aggregate: weight each neighbor's vocab_freq by its similarity
        # Use ReLU on similarity so negative sims don't subtract
        sim_weights = torch.tensor(
            [max(0.0, s) for s in top_sims],
            dtype=torch.float32, device=self.device,
        )
        weight_sum = sim_weights.sum().clamp(min=1e-8)

        bias_vector = torch.zeros(self.vocab_freq.shape[1], device=self.device)
        for rank, (idx, w) in enumerate(zip(top_idxs, sim_weights.tolist())):
            bias_vector += (w / weight_sum) * self.vocab_freq[idx]

        # Top-10 tokens in the bias vector (for logging)
        top10_vals, top10_ids = torch.topk(bias_vector, k=10)
        top10_tokens = [
            (self.text_tokenizer.id_to_piece(int(tid)), float(v))
            for tid, v in zip(top10_ids.cpu().tolist(), top10_vals.cpu().tolist())
        ]

        # Neighbor labels for logging
        neighbor_labels = [
            self.pairs[idx].get("observation", {}) for idx in top_idxs
        ]

        mean_sim = float(sum(top_sims) / len(top_sims)) if top_sims else 0.0

        return {
            "bias_vector":      bias_vector,
            "similarities":     top_sims,
            "neighbor_indices": top_idxs,
            "neighbor_labels":  neighbor_labels,
            "top10_tokens":     top10_tokens,
            "mean_similarity":  mean_sim,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Logit bias hook state (one per streaming session)
# ─────────────────────────────────────────────────────────────────────────────

class LogitBiasState:
    """
    Mutable state bag shared between the streaming loop (which writes to it)
    and the on_text_logits_hook closure (which reads from it).

    The hook runs inside lm_gen.step() — we can't pass arguments, so we use
    a shared object updated at phrase boundaries.
    """

    def __init__(self):
        self.bias_vector: Optional[torch.Tensor] = None  # [32000] on cuda
        self.token_index: int = 0      # steps since last phrase boundary
        self.active: bool = False      # True only after first phrase fires
        self.phrase_retrieval_logs: list = []

    def update_for_phrase(self, bias_vector: torch.Tensor, retrieval_log: dict):
        """Called at each phrase boundary."""
        self.bias_vector  = bias_vector
        self.token_index  = 0
        self.active       = True
        self.phrase_retrieval_logs.append(retrieval_log)

    def current_scale(self) -> float:
        return BIAS_SCALE_INIT * (BIAS_SCALE_DECAY ** self.token_index)

    def tick(self):
        """Increment token counter. Called inside the hook."""
        self.token_index += 1


# ─────────────────────────────────────────────────────────────────────────────
# The hook itself — passed to LMGen(on_text_logits_hook=...)
# ─────────────────────────────────────────────────────────────────────────────

def make_logit_bias_hook(bias_state: LogitBiasState):
    """
    Returns a closure that modifies text_logits in-place before sample_token.

    text_logits shape: [B=1, 1, 1, 32000]
    Modification: text_logits[:, 0, 0, :] += scale * bias_vector
    scale = 3.0 * (0.7 ** token_index) — decays exponentially per token step.
    """

    def hook(text_logits: torch.Tensor) -> None:
        if not bias_state.active or bias_state.bias_vector is None:
            return

        scale = bias_state.current_scale()
        if scale < 0.05:
            # Below this threshold the effect is negligible — skip for perf
            bias_state.tick()
            return

        # In-place add: broadcasts [32000] across [1, 1, 1, 32000]
        bias = bias_state.bias_vector.to(text_logits.dtype)
        text_logits[:, 0, 0, :].add_(scale * bias)
        bias_state.tick()

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Main session function — replaces _run_session_core in handover doc
# ─────────────────────────────────────────────────────────────────────────────

def run_session_with_retrieval_bias(
        self,                      # Modal class instance (has all model attrs)
        audio_bytes: bytes,
        retrieval_index: "RetrievalIndex",
        temp: float      = 0.8,
        temp_text: float = 0.7,
) -> dict:
    """
    Drop-in replacement for _run_session_core that adds Option 2a retrieval bias.

    Additional return keys vs baseline:
      "retrieval_logs": per-phrase retrieval diagnostics
          {
            "phrase_idx": int,
            "mean_similarity": float,
            "similarities": [float, ...],   # top-3 neighbor similarities
            "neighbor_labels": [dict, ...], # tone_quality, note, register
            "top10_bias_tokens": [(piece, weight), ...],
            "reliability": "ok" | "warn" | "fail",
          }
    """
    import torch, torchaudio, time, io
    from moshi.models import LMGen
    from moshi.conditioners import TensorCondition, ConditionAttributes

    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)
    if sr != MOSHI_SR:
        waveform = torchaudio.functional.resample(waveform, sr, MOSHI_SR)
    audio = waveform.mean(0).numpy().astype(np.float32)
    print(f"Audio: {len(audio)/MOSHI_SR:.1f}s | temp={temp} | mode=retrieval_bias")

    # ── Bias state + hook ────────────────────────────────────────────────────
    bias_state = LogitBiasState()
    logit_hook = make_logit_bias_hook(bias_state)

    # ── Session state ────────────────────────────────────────────────────────
    logs, text_buffer, phrase_tokens = [], [], {}
    current_phrase = -1
    phrase_idx     = 0

    acc_buffer, acc_silent, acc_active, acc_start_t = [], 0, False, 0.0
    silence_frames = int(SILENCE_SECS * MOSHI_SR / FRAME)
    NOTE_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

    def get_rms(f): return float(np.sqrt(np.mean(f ** 2)))
    def is_trumpet(fvs):
        return any(fv.f0_hz > 0 and 1400.0 <= fv.spectral_centroid <= 5500.0
                   for fv in fvs)

    # ── LMGen with null init and hook ────────────────────────────────────────
    init_ct = self._make_null_condition_tensors()
    lm_gen  = LMGen(
        lm_model=self.merged_model,
        condition_tensors=init_ct,
        temp=temp,
        temp_text=temp_text,
        on_text_logits_hook=logit_hook,   # ← Option 2a injection point
    )

    with torch.no_grad(), lm_gen.streaming(1), self.mimi.streaming(1):

        def process_frame(frame):
            nonlocal acc_buffer, acc_silent, acc_active, acc_start_t
            nonlocal phrase_idx, current_phrase

            rms = get_rms(frame)
            self.fe.reset()
            fvs, local_fm = [], self.FrameManager()
            for i in range(0, len(frame) - 512, 512):
                for f in local_fm.push(frame[i:i+512]):
                    fvs.append(self.fe.extract(f))
            is_trp = is_trumpet(fvs) if fvs else False

            phrase_fired = None
            if not acc_active:
                if rms > RMS_THRESH and is_trp:
                    acc_active  = True
                    acc_silent  = 0
                    acc_start_t = time.perf_counter()
                    acc_buffer  = [frame.copy()]
            else:
                acc_buffer.append(frame.copy())
                if rms < RMS_THRESH:
                    acc_silent += 1
                    if acc_silent >= silence_frames:
                        phrase_fired = (
                            np.concatenate(acc_buffer),
                            time.perf_counter() - acc_start_t,
                        )
                        acc_buffer, acc_silent, acc_active = [], 0, False
                else:
                    acc_silent = 0

            if phrase_fired is not None and phrase_idx < 20:
                phrase_audio, phrase_duration = phrase_fired
                boundary_time = time.perf_counter()
                print(f"\n[Phrase {phrase_idx+1}] {phrase_duration:.2f}s")

                # ── MERT embedding ───────────────────────────────────────────
                embedding = self._get_phrase_embedding(phrase_audio)  # [1,1,768]
                mert_latency = (time.perf_counter() - boundary_time) * 1000.0

                # ── Condition sum update (same as baseline) ──────────────────
                new_cond_sum, _ = self._update_condition_sum(phrase_audio)
                lm_gen._streaming_state.condition_sum = new_cond_sum

                # ── Retrieval lookup ─────────────────────────────────────────
                query_emb = embedding.view(-1)  # [768]
                retrieval = retrieval_index.query(query_emb, top_k=TOP_K_NEIGHBORS)

                mean_sim = retrieval["mean_similarity"]
                if mean_sim >= SIM_THRESHOLD_OK:
                    reliability = "ok"
                elif mean_sim >= SIM_THRESHOLD_WARN:
                    reliability = "warn"
                else:
                    reliability = "fail"
                    print(f"  ⚠ Low similarity ({mean_sim:.3f}) — bias may not be useful")

                print(f"  Retrieval: mean_sim={mean_sim:.3f} [{reliability}]")
                for rank, (sim, label) in enumerate(
                        zip(retrieval["similarities"], retrieval["neighbor_labels"])
                ):
                    print(f"    NN{rank+1}: sim={sim:.3f}  {label.get('tone_quality','?')} "
                          f"{label.get('note','?')} {label.get('register','?')}")
                print(f"  Top-10 bias: {retrieval['top10_tokens'][:5]}")  # first 5 for readability

                # ── Update bias state → hook will apply on next step() calls ─
                retrieval_log = {
                    "phrase_idx":       phrase_idx,
                    "mean_similarity":  round(mean_sim, 4),
                    "similarities":     [round(s, 4) for s in retrieval["similarities"]],
                    "neighbor_indices": retrieval["neighbor_indices"],
                    "neighbor_labels":  retrieval["neighbor_labels"],
                    "top10_bias_tokens": retrieval["top10_tokens"],
                    "reliability":      reliability,
                }
                bias_state.update_for_phrase(retrieval["bias_vector"], retrieval_log)

                # ── Pitch / note analysis (same as baseline) ─────────────────
                fvs_p, local_fm2 = [], self.FrameManager()
                self.fe.reset()
                for i in range(0, len(phrase_audio) - 512, 512):
                    for f in local_fm2.push(phrase_audio[i:i+512]):
                        fvs_p.append(self.fe.extract(f))
                pitched = [fv for fv in fvs_p
                           if fv.f0_hz > 0 and fv.pitch_salience >= 0.35]
                if pitched:
                    midis       = [int(round(12 * np.log2(fv.f0_hz/440)+69))
                                   for fv in pitched]
                    median_midi = int(np.median(midis))
                    top_note    = f"{NOTE_NAMES[median_midi%12]}{median_midi//12-1}"
                    register    = ("low"   if median_midi < 52 else
                                   "upper" if median_midi > 72 else "middle")
                else:
                    top_note, register = None, "unknown"

                total_latency = (time.perf_counter() - boundary_time) * 1000.0

                logs.append({
                    "phrase_idx":         phrase_idx,
                    "mert_latency_ms":    round(mert_latency, 1),
                    "total_latency_ms":   round(total_latency, 1),
                    "phrase_duration_s":  round(phrase_duration, 2),
                    "top_note":           top_note,
                    "register":           register,
                    "retrieval_sim":      round(mean_sim, 4),
                    "retrieval_reliability": reliability,
                })
                print(f"  MERT: {mert_latency:.0f}ms | Total: {total_latency:.0f}ms")
                print(f"  Note: {top_note} / {register}")
                current_phrase = phrase_idx
                phrase_idx    += 1

            # ── MIMI encode + LMGen step ─────────────────────────────────────
            chunk = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
            codes = self.mimi.encode(chunk.unsqueeze(0))
            if codes.shape[-1] > 0:
                for t in range(codes.shape[-1]):
                    result = lm_gen.step(codes[:, :, t:t+1])
                    # Hook fires inside step() — bias applied before sampling
                    if result is not None:
                        tok_id = int(result[0, 0].item())
                        if tok_id > 3:
                            piece = self.text_tokenizer.id_to_piece(tok_id)
                            text_buffer.append(piece)
                            bucket = current_phrase if current_phrase >= 0 else "pre"
                            phrase_tokens.setdefault(bucket, []).append(piece)

        for start in range(0, len(audio) - FRAME, FRAME):
            process_frame(audio[start : start + FRAME])

    return {
        "logs":           logs,
        "text_tokens":    text_buffer,
        "phrase_tokens":  {str(k): v for k, v in phrase_tokens.items()},
        "retrieval_logs": bias_state.phrase_retrieval_logs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Modal integration helper — instantiate index at container load time
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH      = "/checkpoints/trubai_training_pairs.jsonl"
EMBEDDINGS_DIR  = "/checkpoints/embeddings"

def build_retrieval_index(text_tokenizer, device="cuda") -> "RetrievalIndex":
    """
    Call this once in @modal.enter() after text_tokenizer is loaded.
    Stores result on self.retrieval_index.
    """
    return RetrievalIndex(
        jsonl_path=JSONL_PATH,
        embeddings_dir=EMBEDDINGS_DIR,
        text_tokenizer=text_tokenizer,
        device=device,
    )