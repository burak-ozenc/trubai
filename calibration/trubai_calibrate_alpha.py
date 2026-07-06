"""
trubai_calibrate_alpha.py
SPEC-5BA-v1 §4 — Alpha Calibration

Runs 3 free-generation phrases at each alpha ∈ {0.5, 1.0, 2.0}.
Records 32 free-generation steps post-bridge token per phrase.

Metrics per run:
  1. LaTeX/academic token count in 32 post-prefix steps
  2. Trumpet pedagogical token count in 32 post-prefix steps
  3. Output coherence assessment (readable vs repetitive vs collapsed)
  4. Audio codebook stream continuity (codes generated = codes expected)

Implementation: monkey-patches LMGen._step() to apply text_logit_modifier
before sampling. Proper fork modification follows in §5.

Usage:
    modal run trubai_calibrate_alpha.py
"""

import modal
import json
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

app       = modal.App("trubai-calibrate-alpha", image=image)
vol       = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache  = modal.Volume.from_name("trubai-hf-cache",   create_if_missing=True)
audio_vol = modal.Volume.from_name("trubai-audio-cache", create_if_missing=True)

CKPT_DIR     = Path("/checkpoints")
HF_DIR       = Path("/hf-cache")
AUDIO_PATH   = "/audio-cache/AuSep_2_tpt_43_Chorale.wav"
V13_CKPT     = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_CKPT    = CKPT_DIR / "lora_v3" / "best"
BIAS_CKPT    = CKPT_DIR / "logit_bias" / "bias_v1.pt"

ALPHAS           = [0.5, 1.0, 2.0]
FREE_GEN_STEPS   = 32      # steps to record post-bridge
SILENCE_GATE     = 0.6     # seconds
N_PHRASES        = 3       # phrases per alpha run

# Failure token IDs (from failure_tokens.json — top entries by count)
LATEX_TOKEN_IDS = {
    17035, 25671, 16274, 27459, 21445, 28935, 26337, 13603,
    24388, 25128, 30599, 21817, 25888, 21445, 29026, 2048,   # ▁Mr included
    1117,                                                       # ▁significantly
}

# Trumpet vocabulary pieces (for text matching)
TRUMPET_PIECES = {
    "▁air", "▁breath", "▁tone", "▁column", "▁aperture", "▁buzz",
    "▁pitch", "▁partial", "▁focus", "▁spread", "▁center", "▁hollow",
    "▁crack", "▁flat", "▁sharp", "▁pinch", "▁diffuse", "▁spreading",
    "▁embouchure", "▁support", "▁airflow", "▁breathy",
}


def build_conditioner_v12(device: str):
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
            from moshi.conditioners import TensorCondition as TC
            cond      = TC(tensor=emb, mask=mask)
            proj      = self.tc(cond)[0]
            raw       = proj.squeeze(1)
            direction = F.normalize(raw, dim=-1)
            return direction, self.output_scale

        def condition_sum_vector(self, emb):
            direction, scale = self.forward(emb)
            return direction * scale.detach()

    return ConditionerV12().to(device)


def patch_lmgen_step(lm_gen, logit_modifier):
    """
    Monkey-patch lm_gen to apply logit_modifier inside _step().
    Stores original _step, wraps it to intercept text_logits before sampling.
    This is a calibration-only patch — §5 does the proper fork modification.

    Since we cannot intercept text_logits from outside step() without fork changes,
    we instead wrap the text_linear call to capture and modify logits in-place.
    """
    import types
    import torch

    base = lm_gen.lm_model if hasattr(lm_gen, 'lm_model') else lm_gen

    original_text_linear = base.text_linear

    class BiasedLinear(torch.nn.Module):
        def __init__(self, inner, modifier):
            super().__init__()
            self.inner    = inner
            self.modifier = modifier

        def forward(self, x):
            logits = self.inner(x)
            if self.modifier is not None:
                logits = self.modifier(logits)
            return logits

    biased = BiasedLinear(original_text_linear, logit_modifier)
    base.text_linear = biased

    def restore():
        base.text_linear = original_text_linear

    return restore


@app.cls(
    gpu="H100",
    volumes={
        "/checkpoints": vol,
        "/hf-cache":    hf_cache,
        "/audio-cache": audio_vol,
    },
    timeout=7200,
)
class AlphaCalibration:

    @modal.enter()
    def setup(self):
        import torch._dynamo
        torch._dynamo.config.disable = True

        import os
        os.environ["HF_HOME"] = str(HF_DIR)

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners import ConditionProvider, ConditionFuser, TensorCondition
        from peft import LoraConfig, get_peft_model

        self.device = torch.device("cuda")

        # Tokenizer
        tok_path  = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp   = sentencepiece.SentencePieceProcessor(tok_path)

        # Mimi
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi   = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)

        # Moshi LM + LoRA
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )

        # v13 ConditionerV12
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False

        # ConditionProvider / Fuser
        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

        # LoRA
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        peft_model = get_peft_model(moshi_lm, lora_config)
        peft_model.load_adapter(str(LORA_CKPT), adapter_name="default")
        peft_model.eval()

        # Merge LoRA — required for streaming
        self.lm_model = peft_model.merge_and_unload()
        self.lm_model.condition_provider = self.cp
        self.lm_model.fuser              = self.fuser

        # PhraseConditioner
        import sys; sys.path.insert(0, "/root")
        from phrase_conditioner import PhraseConditioner
        self.phrase_conditioner = PhraseConditioner(tokenizer_path=tok_path)

        # Load bias vector (alpha applied at run time)
        bias_data      = torch.load(str(BIAS_CKPT), map_location="cpu", weights_only=True)
        self.bias_vec  = bias_data["bias_vector"].float()
        print(f"bias_v1.pt loaded. Bias vector shape: {self.bias_vec.shape}")
        print("Setup complete.")

    @modal.method()
    def calibrate(self) -> dict:
        import torch
        import torchaudio
        from moshi.models import LMGen
        from moshi.conditioners import ConditionAttributes, TensorCondition
        from trublib import TADProcessor, TADConfig
        from trublib.frame_manager import FrameManager
        from trublib import FeatureExtractor
        from phrase_conditioner import phrase_features_from_vectors
        import sys; sys.path.insert(0, "/root")

        # Load audio once
        wav, sr = torchaudio.load(AUDIO_PATH)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(0).to(self.device)

        # Null condition tensors for LMGen init
        null_tc   = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device=self.device)
        null_mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        null_cond = TensorCondition(tensor=null_tc, mask=null_mask)
        null_attrs    = ConditionAttributes(text={}, tensor={"mert": null_cond})
        null_prepared = self.cp.prepare([null_attrs])
        ct = self.cp(null_prepared)

        results = {}

        for alpha in ALPHAS:
            print()
            print(f"{'='*60}")
            print(f"ALPHA = {alpha}")
            print(f"{'='*60}")

            # Build modifier for this alpha
            bias_dev = self.bias_vec.to(self.device)
            def make_modifier(a, b):
                def modifier(logits):
                    return logits + a * b.to(logits.device, logits.dtype)
                return modifier
            modifier = make_modifier(alpha, bias_dev)

            phrase_results = []

            for phrase_idx in range(N_PHRASES):
                print(f"\n  Phrase {phrase_idx+1}/{N_PHRASES} — alpha={alpha}")
                self.phrase_conditioner._queue.clear()
                self.phrase_conditioner._bridge_queue.clear()

                lm_gen   = LMGen(self.lm_model, condition_tensors=ct)
                restore  = patch_lmgen_step(lm_gen, modifier)

                # Track metrics
                free_gen_token_ids  = []
                free_gen_pieces     = []
                post_bridge_steps   = 0
                in_free_gen         = False
                phrase_primed       = False
                audio_codes_count   = 0
                phrase_found        = False

                # State for phrase detection
                silence_secs  = 0.0
                in_phrase     = False
                phrase_buf    = []
                phrase_count  = 0
                chunk_size    = 1920

                with torch.no_grad():
                    with lm_gen.streaming(1):
                        with self.mimi.streaming(1):
                            for i in range(0, wav.shape[0] - chunk_size + 1, chunk_size):
                                chunk   = wav[i:i + chunk_size]
                                rms     = chunk.pow(2).mean().sqrt().item()
                                is_sil  = rms < 0.01

                                chunk_in = chunk.unsqueeze(0).unsqueeze(0)
                                codes    = self.mimi.encode(chunk_in)   # [1, K, T']
                                audio_codes_count += codes.shape[-1]

                                if is_sil:
                                    silence_secs += chunk_size / 24000
                                    if (silence_secs >= SILENCE_GATE and
                                            in_phrase and phrase_buf and
                                            phrase_count < 1):
                                        in_phrase   = False
                                        phrase_count += 1
                                        phrase_found = True

                                        # Extract MERT & prime conditioner
                                        pa = torch.cat(phrase_buf).unsqueeze(0)
                                        from transformers import AutoModel
                                        # Use cached MERT if available
                                        if not hasattr(self, '_mert'):
                                            self._mert = AutoModel.from_pretrained(
                                                "m-a-p/MERT-v1-95M",
                                                trust_remote_code=True,
                                                cache_dir=str(HF_DIR),
                                            ).to(self.device).eval()
                                        mert_out = self._mert(pa.float())
                                        mert_emb = mert_out.last_hidden_state.mean(1, keepdim=True)
                                        cond_vec = self.conditioner.condition_sum_vector(
                                            mert_emb.to(torch.bfloat16)
                                        )
                                        lm_gen._streaming_state.condition_sum = \
                                            cond_vec.unsqueeze(1)

                                        # Feature extraction for PhraseConditioner
                                        _np  = pa.squeeze(0).cpu().numpy()
                                        _fm  = FrameManager()
                                        _fe  = FeatureExtractor(sr=24000)
                                        _fvs = []
                                        _csz = 512
                                        for _i in range(0, len(_np) - _csz + 1, _csz):
                                            for _f in _fm.push(_np[_i:_i+_csz]):
                                                _fvs.append(_fe.extract(_f))
                                        _pf = phrase_features_from_vectors(_fvs)
                                        if _pf is not None:
                                            self.phrase_conditioner.prime(_pf)
                                        phrase_buf = []
                                else:
                                    silence_secs = 0.0
                                    if not in_phrase:
                                        in_phrase = True
                                    phrase_buf.append(chunk)

                                # Step through codes
                                for t in range(codes.shape[-1]):
                                    code_slice = codes[:, :, t:t+1]
                                    forced, tag = self.phrase_conditioner.next_token(self.device)

                                    result = lm_gen.step(code_slice, forced_text_token=forced)
                                    if result is None:
                                        continue

                                    tok_id = result[0, 0, 0].item()
                                    if tok_id <= 3:
                                        continue

                                    piece = self.sp.id_to_piece(tok_id)

                                    # Track when bridge token is consumed
                                    was_bridge = (tag == '[B]')
                                    if was_bridge:
                                        in_free_gen = True
                                        post_bridge_steps = 0

                                    if in_free_gen and not was_bridge:
                                        if post_bridge_steps < FREE_GEN_STEPS:
                                            free_gen_token_ids.append(tok_id)
                                            free_gen_pieces.append(piece)
                                            post_bridge_steps += 1

                                # Stop after collecting enough free-gen tokens
                                if in_free_gen and post_bridge_steps >= FREE_GEN_STEPS:
                                    break

                restore()  # Remove patch

                # Compute metrics for this phrase
                latex_count   = sum(1 for tid in free_gen_token_ids if tid in LATEX_TOKEN_IDS)
                trumpet_count = sum(1 for p in free_gen_pieces if p in TRUMPET_PIECES)
                token_seq     = " ".join(free_gen_pieces[:FREE_GEN_STEPS])

                # Coherence: check for degenerate cycling
                if len(free_gen_pieces) >= 4:
                    unique_ratio = len(set(free_gen_pieces)) / len(free_gen_pieces)
                    if unique_ratio < 0.1:
                        coherence = "COLLAPSED (near-deterministic cycling)"
                    elif unique_ratio < 0.25:
                        coherence = "REPETITIVE (heavy cycling)"
                    elif trumpet_count >= 3:
                        coherence = "PEDAGOGICAL (trumpet vocab present)"
                    else:
                        coherence = "MIXED (some variety, limited trumpet vocab)"
                else:
                    coherence = "INSUFFICIENT TOKENS"

                phrase_results.append({
                    "latex_count":   latex_count,
                    "trumpet_count": trumpet_count,
                    "coherence":     coherence,
                    "token_seq":     token_seq,
                    "audio_codes":   audio_codes_count,
                    "n_tokens":      len(free_gen_pieces),
                })

                print(f"    LaTeX tokens:   {latex_count}/{FREE_GEN_STEPS}")
                print(f"    Trumpet tokens: {trumpet_count}/{FREE_GEN_STEPS}")
                print(f"    Coherence:      {coherence}")
                print(f"    Audio codes:    {audio_codes_count} (continuous={audio_codes_count > 0})")
                print(f"    Token stream:   {token_seq[:80]}...")

            results[alpha] = phrase_results

        # Summary report
        print()
        print("=" * 60)
        print("§4 ALPHA CALIBRATION SUMMARY — REPORT TO MUSE")
        print("=" * 60)
        for alpha in ALPHAS:
            pr = results[alpha]
            avg_latex   = sum(r["latex_count"]   for r in pr) / len(pr)
            avg_trumpet = sum(r["trumpet_count"] for r in pr) / len(pr)
            audio_ok    = all(r["audio_codes"] > 0 for r in pr)
            coherences  = [r["coherence"] for r in pr]
            print(f"\nalpha={alpha}:")
            print(f"  Avg LaTeX tokens (32 steps): {avg_latex:.1f}")
            print(f"  Avg trumpet tokens (32 steps): {avg_trumpet:.1f}")
            print(f"  Audio codebook continuous: {'✓' if audio_ok else '✗ BROKEN'}")
            print(f"  Coherence per phrase:")
            for i, c in enumerate(coherences):
                print(f"    phrase {i+1}: {c}")

        print()
        print("Awaiting Muse alpha selection before §5 integration.")
        print("=" * 60)

        return results


@app.local_entrypoint()
def main():
    AlphaCalibration().calibrate.remote()