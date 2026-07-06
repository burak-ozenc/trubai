"""
trubai_streaming_v13.py
Streaming verification — TensorConditioner v13
SPEC-TC-V13-v1 (Muse-approved)

Protocol: identical to Phase 4 / retrain_v2 verification.
Key v13 difference: ConditionerV12 wrapper — inject direction * output_scale,
not raw projection. output_scale is a buffer (fixed 36.5), direction is unit-norm.

Diagnostic outputs:
  1. Attractor word frequency (proud, brittle, ville, nik, tower, Az,
     nationality, Tournament, major, cherish, mascot)
  2. Trumpet vocabulary frequency (air, breath, tone, column, aperture,
     buzz, pitch, partial, focus, spread, center, hollow)
  3. Space token cycling detection (▁ ▁ ▁ pattern)

Usage:
    # Diagnostic run (0.6s gate, full token stream)
    modal run trubai_streaming_v13.py

    # Null-conditioned baseline (no conditioner injection)
    modal run trubai_streaming_v13.py --null-conditioned 1
"""

import modal
from pathlib import Path

MOSHI_FORK_URL = "git+https://github.com/burak-ozenc/moshi.git#subdirectory=moshi"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "ffmpeg"])
    .pip_install([
        "torch==2.6.0",
        "torchaudio==2.6.0",
        MOSHI_FORK_URL,
        "transformers",
        "sentencepiece",
        "huggingface_hub",
        "safetensors",
        "peft",
        "trublib",
    ])
    .add_local_file(
        Path(__file__).parent.parent / "conditioner" / "phrase_conditioner.py",
        "/root/phrase_conditioner.py",
        )
)

app       = modal.App("trubai-streaming-v13", image=image)
vol       = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache  = modal.Volume.from_name("trubai-hf-cache",   create_if_missing=True)
audio_vol = modal.Volume.from_name("trubai-audio-cache", create_if_missing=True)

CKPT_DIR     = Path("/checkpoints")
HF_DIR       = Path("/hf-cache")
AUDIO_PATH   = "/audio-cache/AuSep_2_tpt_43_Chorale.wav"

# v13 checkpoint — ConditionerV12 state dict
CONDITIONER_CKPT = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"

# LoRA — production v2 (unchanged; conditioner is the variable)
LORA_CKPT    = CKPT_DIR / "lora_v3" / "best"

SILENCE_GATE_SECS = 0.6   # diagnostic gate

# ──────────────────────────────────────────────────────────────────────────────
# ConditionerV12 — must match training definition exactly
# ──────────────────────────────────────────────────────────────────────────────

def build_conditioner_v12(device: str):
    """
    Identical to trubai_train_v13.py — must match exactly.
    output_scale is register_buffer (fixed 36.5, not a parameter).
    Injection: direction * output_scale preserves ~36.5 magnitude.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from moshi.conditioners.tensors import TensorConditioner
    from moshi.conditioners import TensorCondition

    class ConditionerV12(nn.Module):
        def __init__(self):
            super().__init__()
            self.tc = TensorConditioner(
                dim=768, output_dim=4096, device=device,
                force_linear=True, output_bias=False, learn_padding=True,
            )
            self.register_buffer("output_scale", torch.tensor(36.5))

        def forward(self, emb):
            mask      = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            cond      = TensorCondition(tensor=emb, mask=mask)
            proj      = self.tc(cond)[0]
            raw       = proj.squeeze(1)
            direction = F.normalize(raw, dim=-1)
            return direction, self.output_scale

        def condition_sum_vector(self, emb):
            """
            Returns the vector to inject into condition_sum.
            direction * output_scale — magnitude ~36.5, matching retrain_v2/best.
            """
            direction, scale = self.forward(emb)
            return direction * scale.detach()   # [1, 4096], ||x|| ≈ 36.5

    c = ConditionerV12().to(device)
    return c


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary sets for diagnostic
# ──────────────────────────────────────────────────────────────────────────────

ATTRACTOR_WORDS = {
    "proud", "brittle", "ville", "nik", "tower", "Az",
    "nationality", "Tournament", "major", "cherish", "mascot",
    "disagreement", "canonical", "iteration",
}

TRUMPET_WORDS = {
    "air", "breath", "tone", "column", "aperture", "buzz",
    "pitch", "partial", "focus", "spread", "center", "hollow",
    "embouchure", "support", "airflow", "crack", "pinch", "sharp",
    "flat", "breathy",
}

SPACE_TOKEN_ID = 260
# ▁Mr cycling — observed in lora_v3 gate sample, check if it persists in full context
MR_PIECE = "▁Mr"


# ──────────────────────────────────────────────────────────────────────────────
# Modal class
# ──────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu="H100",
    volumes={
        "/checkpoints": vol,
        "/hf-cache":    hf_cache,
        "/audio-cache": audio_vol,
    },
    timeout=3600,
)
class StreamingV13:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = str(HF_DIR)

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners import ConditionAttributes
        from peft import PeftModel

        self.device = torch.device("cuda")

        # ── Moshi base models ─────────────────────────────────────────────────
        print("Loading Moshi...")
        mimi_weight  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        self.mimi    = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16,
        )

        # ── LoRA (production v2) ──────────────────────────────────────────────
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        self.moshi_model = get_peft_model(moshi_lm, lora_config)
        self.moshi_model.load_adapter(str(LORA_CKPT), adapter_name="default")
        self.moshi_model.set_adapter("default")
        print("  LoRA v2 loaded ✓")

        # ── Text tokenizer ────────────────────────────────────────────────────
        tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

        # ── v13 ConditionerV12 ────────────────────────────────────────────────
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(
            str(CONDITIONER_CKPT), map_location=self.device, weights_only=True
        )
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        print(f"  ConditionerV12 loaded from {CONDITIONER_CKPT} ✓")
        print(f"  output_scale = {self.conditioner.output_scale.item():.2f}")

        # ── MERT encoder ──────────────────────────────────────────────────────
        from transformers import AutoModel
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
            cache_dir=str(HF_DIR),
        ).to(self.device).eval()
        print("  MERT-v1-95M loaded ✓")

        # ── PhraseConditioner (SPEC-5BC-v1) ───────────────────────────────────
        import sys
        sys.path.insert(0, "/root")
        from phrase_conditioner import PhraseConditioner
        self.phrase_conditioner = PhraseConditioner(tokenizer_path=tok_path)

        # ── ConditionProvider / Fuser wiring ──────────────────────────────────
        from moshi.conditioners import (
            ConditionProvider, ConditionFuser,
            ConditionAttributes,
        )
        from moshi.conditioners.tensors import TensorConditioner as _TC

        # Wire the inner TC into the condition provider
        self.condition_provider = ConditionProvider(
            conditioners={"mert": self.conditioner.tc},
            device=self.device,
        ).to(torch.bfloat16).to(self.device)

        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)

        base = self.moshi_model.get_base_model()
        base.condition_provider = self.condition_provider
        base.fuser               = self.fuser

        print("Setup complete.")

    @modal.method()
    def run_session(
            self,
            audio_path:      str  = AUDIO_PATH,
            silence_gate:    float = SILENCE_GATE_SECS,
            null_conditioned: bool = False,
    ) -> None:
        import torch
        import torchaudio
        from trublib import TADProcessor, TADConfig, TADState
        from moshi.conditioners import ConditionAttributes

        print()
        print("=" * 70)
        print(f"STREAMING VERIFICATION — v13 TensorConditioner")
        print(f"  checkpoint:       {CONDITIONER_CKPT}")
        print(f"  audio:            {audio_path}")
        print(f"  silence_gate:     {silence_gate}s")
        print(f"  null_conditioned: {null_conditioned}")
        print("=" * 70)
        print()

        # ── Diagnostics accumulators ──────────────────────────────────────────
        phrase_count    = 0
        all_tokens      = []   # (token_id, piece, tag) across all phrases
        attractor_hits  = []
        trumpet_hits    = []
        space_runs      = 0    # consecutive space token count (cycling detector)
        max_space_run   = 0
        mr_runs         = 0    # ▁Mr consecutive count
        max_mr_run      = 0
        mr_total        = 0    # total ▁Mr occurrences

        # ── Load audio ────────────────────────────────────────────────────────
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(0, keepdim=True).to(self.device)   # [1, T]
        audio_frames = wav.squeeze(0)

        # ── TADProcessor ──────────────────────────────────────────────────────
        tad = TADProcessor(config=TADConfig(), model_path=None)

        # ── Initial null condition_tensors for LMGen ──────────────────────────
        # ConditionProvider.prepare() expects TensorCondition(tensor, mask), not raw tensor
        from moshi.conditioners import TensorCondition as _TC
        null_tc   = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device=self.device)  # MERT dim
        null_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        null_cond = _TC(tensor=null_tc, mask=null_mask)
        null_attrs    = ConditionAttributes(text={}, tensor={"mert": null_cond})
        null_prepared = self.condition_provider.prepare([null_attrs])
        condition_tensors = self.condition_provider(null_prepared)

        # ── Merge LoRA → LMGen ───────────────────────────────────────────────
        # merge_and_unload() produces a plain LMModel — need LMGen for .step().
        # Re-attach condition_provider/fuser after merge (dropped by merge_and_unload).
        from moshi.models import LMGen
        lm_model = self.moshi_model.merge_and_unload()
        lm_model.condition_provider = self.condition_provider
        lm_model.fuser              = self.fuser
        lm_gen = LMGen(
            lm_model,
            condition_tensors=condition_tensors,
        )

        # ── Streaming session ─────────────────────────────────────────────────
        with torch.no_grad():
            with lm_gen.streaming(1):
                with self.mimi.streaming(1):

                    chunk_size   = 1920   # 80ms at 24kHz
                    silence_secs = 0.0
                    in_phrase    = False
                    phrase_buf   = []     # accumulate audio for MERT

                    for i in range(0, audio_frames.shape[0], chunk_size):
                        chunk   = audio_frames[i:i + chunk_size]
                        if chunk.shape[0] < chunk_size:
                            break

                        # Encode chunk → audio codes
                        chunk_in = chunk.unsqueeze(0).unsqueeze(0)   # [1,1,T]
                        codes    = self.mimi.encode(chunk_in)         # [B, K, T']

                        # RMS silence detection
                        rms = chunk.pow(2).mean().sqrt().item()
                        is_silent = rms < 0.01

                        if is_silent:
                            silence_secs += chunk_size / 24000
                            if silence_secs >= silence_gate and in_phrase and phrase_buf:
                                # Phrase boundary — extract MERT embedding
                                in_phrase = False
                                phrase_count += 1
                                phrase_audio = torch.cat(phrase_buf, dim=0).unsqueeze(0)

                                print(f"\n{'─'*60}")
                                print(f"Phrase {phrase_count} detected "
                                      f"({phrase_audio.shape[-1]/24000:.1f}s)")

                                if not null_conditioned:
                                    # MERT embedding
                                    mert_out  = self.mert(phrase_audio.float())
                                    # last_hidden_state: [B, seq, 768]
                                    # .mean(1, keepdim=True): [1, 1, 768] — correct shape
                                    # do NOT unsqueeze again (would give [1, 1, 1, 768])
                                    mert_emb  = mert_out.last_hidden_state.mean(1, keepdim=True)  # [1,1,768]

                                    # condition_sum injection: direction * output_scale
                                    cond_vec = self.conditioner.condition_sum_vector(
                                        mert_emb.to(torch.bfloat16)
                                    ).to(torch.bfloat16)   # [1, 4096]

                                    print(f"  injection magnitude: "
                                          f"{cond_vec.norm().item():.2f} "
                                          f"(target ~36.5)")

                                    # Inject into condition_sum
                                    lm_gen._streaming_state.condition_sum = \
                                        cond_vec.unsqueeze(1)   # [1, 1, 4096]

                                    # Prime phrase conditioner:
                                    # 1. Extract FeatureVectors across all phrase frames
                                    # 2. Convert to PhraseFeatures via phrase_features_from_vectors()
                                    # 3. Call prime() with PhraseFeatures
                                    from trublib import FeatureExtractor
                                    from trublib.frame_manager import FrameManager
                                    from phrase_conditioner import phrase_features_from_vectors
                                    _audio_np  = phrase_audio.squeeze(0).cpu().numpy()
                                    _fm        = FrameManager()
                                    _fe        = FeatureExtractor(sr=24000)
                                    _fvs       = []
                                    _chunk_sz  = 512
                                    for _i in range(0, len(_audio_np) - _chunk_sz + 1, _chunk_sz):
                                        _frames = _fm.push(_audio_np[_i:_i + _chunk_sz])
                                        for _frame in _frames:
                                            _fvs.append(_fe.extract(_frame))
                                    _pf = phrase_features_from_vectors(_fvs)
                                    if _pf is not None:
                                        self.phrase_conditioner.prime(_pf)
                                    else:
                                        print(f"  [phrase] insufficient pitched frames — skipping prime()")

                                phrase_buf = []
                        else:
                            silence_secs = 0.0
                            if not in_phrase:
                                in_phrase = True
                            phrase_buf.append(chunk)

                        # LMGen step
                        forced, token_tag = self.phrase_conditioner.next_token(
                            self.device
                        ) if in_phrase else (None, "")

                        for t in range(codes.shape[-1]):
                            code_slice = codes[:, :, t:t+1]
                            result     = lm_gen.step(
                                code_slice,
                                forced_text_token=forced,
                            )
                            if result is None:
                                continue

                            tok_id = result[0, 0, 0].item()
                            if tok_id <= 3:
                                continue

                            piece = self.text_tokenizer.id_to_piece(tok_id)
                            tag   = token_tag if forced is not None else ""
                            all_tokens.append((tok_id, piece, tag))

                            # Tag for display
                            tag_display = f"[{token_tag}] " if token_tag else "    "
                            print(f"  {tag_display}{piece}", end="", flush=True)

                            # Space token cycling
                            if tok_id == SPACE_TOKEN_ID:
                                space_runs += 1
                                max_space_run = max(max_space_run, space_runs)
                            else:
                                space_runs = 0

                            # Attractor / trumpet vocabulary check
                            clean = piece.lstrip("▁").lower()
                            if clean in {w.lower() for w in ATTRACTOR_WORDS}:
                                attractor_hits.append((phrase_count, piece, tag))
                            if clean in {w.lower() for w in TRUMPET_WORDS}:
                                trumpet_hits.append((phrase_count, piece, tag))

        # ── Diagnostic summary ────────────────────────────────────────────────
        print()
        print()
        print("=" * 70)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"  Phrases processed:      {phrase_count}")
        print(f"  Total tokens generated: {len(all_tokens)}")
        print()

        # 1. Attractor word check
        print(f"  ATTRACTOR WORDS (target: absent)")
        if attractor_hits:
            print(f"  ✗ {len(attractor_hits)} attractor hits:")
            for ph, piece, tag in attractor_hits:
                print(f"    phrase={ph} piece={piece!r} tag={tag!r}")
        else:
            print(f"  ✓ No attractor words found")
        print()

        # 2. Trumpet vocabulary check
        print(f"  TRUMPET VOCABULARY (target: present)")
        if trumpet_hits:
            print(f"  ✓ {len(trumpet_hits)} trumpet vocabulary hits:")
            for ph, piece, tag in trumpet_hits:
                print(f"    phrase={ph} piece={piece!r} tag={tag!r}")
        else:
            print(f"  ✗ No trumpet vocabulary found")
        print()

        # 3. Space token cycling
        print(f"  SPACE TOKEN CYCLING (target: absent, max_run=0)")
        if max_space_run >= 3:
            print(f"  ✗ Space cycling detected — max consecutive run: {max_space_run}")
        elif max_space_run > 0:
            print(f"  ⚠ Occasional space tokens (max run: {max_space_run}) — monitor")
        else:
            print(f"  ✓ No space token cycling")
        print()
        # 4. ▁Mr cycling (lora_v3 specific — gate sample showed ▁Mr ▁Mr ▁Mr)
        print(f"  ▁Mr CYCLING (target: absent in full context)")
        if max_mr_run >= 3:
            print(f"  ✗ ▁Mr cycling confirmed — max run: {max_mr_run}, total: {mr_total}")
        elif mr_total > 0:
            print(f"  ⚠ ▁Mr present but not cycling (total: {mr_total}, max_run: {max_mr_run})")
        else:
            print(f"  ✓ ▁Mr absent — gate sample was artifact")
        print()

        # Pass/fail
        attractor_ok = len(attractor_hits) == 0
        trumpet_ok   = len(trumpet_hits) > 0
        space_ok     = max_space_run < 3
        mr_ok        = max_mr_run < 3

        print(f"  RESULT: {'PASS' if (attractor_ok and trumpet_ok and space_ok and mr_ok) else 'FAIL'}")
        print(f"    attractor absent: {'✓' if attractor_ok else '✗'}")
        print(f"    trumpet present:  {'✓' if trumpet_ok else '✗'}")
        print(f"    no space cycling: {'✓' if space_ok else '✗'}")
        print(f"    no ▁Mr cycling:   {'✓' if mr_ok else '✗'}")
        print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
        audio_path:       str   = AUDIO_PATH,
        silence_gate:     float = SILENCE_GATE_SECS,
        null_conditioned: int   = 0,
) -> None:
    """
    null_conditioned=1: skip conditioner injection — baseline comparison.
    silence_gate: seconds of silence to trigger phrase boundary (default 0.6).
    """
    StreamingV13().run_session.remote(
        audio_path       = audio_path,
        silence_gate     = silence_gate,
        null_conditioned = bool(null_conditioned),
    )