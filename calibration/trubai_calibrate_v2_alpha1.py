"""
trubai_calibrate_v2.py
SPEC-5BA-v1 §4 rerun — bias_v2.pt (suppression-only vector)

Muse instruction: zero out all 17 W_POSITIVE_IDS entries in bias_v1.pt.
LoRA owns positive attraction. LogitBiasVector owns negative suppression only.

Steps:
  1. Load bias_v1.pt, zero W_POSITIVE_IDS, save bias_v2.pt
  2. Run §4 calibration at alpha=1.0 ONLY
  3. Report 4 metrics + direct comparison to lora_v3/best streaming baseline

Fix notes:
  - ▁spreading (id=13369): in W_POSITIVE_IDS AND failure tokens → was -4.7155 in v1
  - Some failure tokens have p_ped > p_fail → small positive values in v1 from
    the negative-side formula. These are NOT in W_POSITIVE_IDS and are left as-is.
    The "zero positives" assertion was incorrect and is removed.
  - Only assertions retained:
      1. Every W_POSITIVE_IDS entry is exactly 0.0 in v2
      2. Every non-W_POSITIVE entry is byte-identical to v1

Do not proceed to §5 without Muse confirmation.
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

app       = modal.App("trubai-calibrate-v2-alpha1", image=image)
vol       = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache  = modal.Volume.from_name("trubai-hf-cache",   create_if_missing=True)
audio_vol = modal.Volume.from_name("trubai-audio-cache", create_if_missing=True)

CKPT_DIR   = Path("/checkpoints")
HF_DIR     = Path("/hf-cache")
AUDIO_PATH = "/audio-cache/AuSep_2_tpt_43_Chorale.wav"
V13_CKPT   = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_CKPT  = CKPT_DIR / "lora_v3" / "best"
BIAS_V1    = CKPT_DIR / "logit_bias" / "bias_v1.pt"
BIAS_V2    = CKPT_DIR / "logit_bias" / "bias_v2.pt"

ALPHA          = 1.0
FREE_GEN_STEPS = 32
SILENCE_GATE   = 0.6
N_PHRASES      = 3

BASELINE_TRUMPET_HITS    = 124
BASELINE_TOTAL_TOKENS    = 562
BASELINE_TRUMPET_DENSITY = BASELINE_TRUMPET_HITS / BASELINE_TOTAL_TOKENS

W_POSITIVE_IDS = [
    1142,   # ▁air
    2368,   # ▁column
    9064,   # ▁tone
    8735,   # ▁breath
    16252,  # ▁aperture
    1611,   # ▁center
    6615,   # ▁crack
    3077,   # ▁flat
    6064,   # ▁sharp
    6396,   # ▁pitch
    4107,   # ▁partial
    21938,  # ▁buzz
    1563,   # ▁focus
    24657,  # ▁pinch
    11984,  # ▁diffuse
    13369,  # ▁spreading  ← was negative in bias_v1 (overlap with failure tokens)
    19664,  # ▁hollow
]

W_POSITIVE_SET = set(W_POSITIVE_IDS)

LATEX_TOKEN_IDS = {
    17035, 25671, 16274, 27459, 21445, 28935, 26337, 13603,
    24388, 25128, 30599, 21817, 25888, 29026, 2048, 1117,
}

TRUMPET_PIECES = {
    "▁air", "▁breath", "▁tone", "▁column", "▁aperture", "▁buzz",
    "▁pitch", "▁partial", "▁focus", "▁spread", "▁center", "▁hollow",
    "▁embouchure", "▁support", "▁airflow", "▁crack", "▁pinch", "▁sharp",
    "▁flat", "▁breathy", "▁spreading", "▁diffuse",
}


def build_conditioner_v12(device: str):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from moshi.conditioners.tensors import TensorConditioner

    class ConditionerV12(nn.Module):
        def __init__(self):
            super().__init__()
            self.tc = TensorConditioner(
                dim=768, output_dim=4096, device=device,
                force_linear=True, output_bias=False, learn_padding=True,
            )
            self.register_buffer("output_scale", torch.tensor(36.5))

        def forward(self, emb):
            mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            from moshi.conditioners import TensorCondition as TC
            cond = TC(tensor=emb, mask=mask)
            proj = self.tc(cond)[0]
            raw  = proj.squeeze(1)
            direction = F.normalize(raw, dim=-1)
            return direction, self.output_scale

        def condition_sum_vector(self, emb):
            direction, scale = self.forward(emb)
            return direction * scale.detach()

    return ConditionerV12().to(device)


def patch_lmgen_step(lm_gen, logit_modifier):
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

    base.text_linear = BiasedLinear(original_text_linear, logit_modifier)

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
class CalibrateV2:

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

        # ── Step 1: Produce bias_v2.pt ────────────────────────────────────────
        print("Producing bias_v2.pt...")
        bias_data = torch.load(str(BIAS_V1), map_location="cpu", weights_only=True)
        bias_v1   = bias_data["bias_vector"].float()

        bias_v2 = bias_v1.clone()
        zeroed  = []
        for tid in W_POSITIVE_IDS:
            original_val = bias_v2[tid].item()
            bias_v2[tid] = 0.0
            zeroed.append((tid, original_val))

        torch.save({"bias_vector": bias_v2, "alpha": ALPHA}, str(BIAS_V2))
        vol.commit()

        print(f"bias_v2.pt saved: {BIAS_V2}")
        print(f"Zeroed {len(zeroed)} W_POSITIVE entries:")
        for tid, val in zeroed:
            note = " <- was negative (failure token overlap)" if val < 0 else ""
            print(f"  id={tid:6d}  was={val:+.4f}  now=0.0{note}")

        # Verification — two invariants only:
        # 1. Every W_POSITIVE_IDS entry is exactly 0.0 in v2
        # 2. Every non-W_POSITIVE entry is unchanged from v1
        #
        # NOTE: bias_v1 has positive values outside W_POSITIVE_IDS (failure tokens
        # where p_ped > p_fail yield positive log-ratio). These are left as-is.
        for tid in W_POSITIVE_IDS:
            assert bias_v2[tid].item() == 0.0, f"W_POSITIVE id={tid} not zeroed"

        for i in range(bias_v1.shape[0]):
            if i not in W_POSITIVE_SET:
                assert bias_v2[i].item() == bias_v1[i].item(), (
                    f"Non-W_POSITIVE entry id={i} unexpectedly changed"
                )

        # Diagnostic: count residual positives outside W_POSITIVE_IDS
        residual_pos = [(i, bias_v2[i].item()) for i in range(bias_v2.shape[0])
                        if bias_v2[i].item() > 0 and i not in W_POSITIVE_SET]
        n_neg_v2 = (bias_v2 < 0).sum().item()

        print(f"Verification:")
        print(f"  All W_POSITIVE entries = 0.0 ✓")
        print(f"  Non-W_POSITIVE entries unchanged ✓")
        print(f"  Negative entries in bias_v2: {n_neg_v2}")
        print(f"  Residual positive entries (non-W_POSITIVE, small log-ratio): "
              f"{len(residual_pos)}")
        if residual_pos:
            print(f"  Residual positive sample (top 5 by value):")
            for tid, val in sorted(residual_pos, key=lambda x: -x[1])[:5]:
                print(f"    id={tid:6d}  val={val:+.4f}")

        # ── Tokenizer ─────────────────────────────────────────────────────────
        tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp  = sentencepiece.SentencePieceProcessor(tok_path)

        # ── Mimi ──────────────────────────────────────────────────────────────
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi   = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)

        # ── Moshi LM + LoRA ───────────────────────────────────────────────────
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )

        # ── ConditionerV12 ────────────────────────────────────────────────────
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False

        # ── ConditionProvider / Fuser ─────────────────────────────────────────
        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

        # ── LoRA ──────────────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        peft_model = get_peft_model(moshi_lm, lora_config)
        peft_model.load_adapter(str(LORA_CKPT), adapter_name="default")
        peft_model.eval()

        self.lm_model = peft_model.merge_and_unload()
        self.lm_model.condition_provider = self.cp
        self.lm_model.fuser              = self.fuser

        # ── PhraseConditioner ─────────────────────────────────────────────────
        import sys; sys.path.insert(0, "/root")
        from phrase_conditioner import PhraseConditioner
        self.phrase_conditioner = PhraseConditioner(tokenizer_path=tok_path)

        self._mert    = None
        self.bias_vec = bias_v2.to(self.device)
        print("Setup complete. bias_v2.pt active (suppression-only, alpha=1.0).")

    @modal.method()
    def calibrate(self) -> dict:
        import torch
        import torchaudio
        from moshi.models import LMGen
        from moshi.conditioners import ConditionAttributes, TensorCondition
        from trublib.frame_manager import FrameManager
        from trublib import FeatureExtractor
        from phrase_conditioner import phrase_features_from_vectors
        import sys; sys.path.insert(0, "/root")

        wav, sr = torchaudio.load(AUDIO_PATH)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(0).to(self.device)

        null_tc       = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device=self.device)
        null_mask     = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        null_cond     = TensorCondition(tensor=null_tc, mask=null_mask)
        null_attrs    = ConditionAttributes(text={}, tensor={"mert": null_cond})
        null_prepared = self.cp.prepare([null_attrs])
        ct            = self.cp(null_prepared)

        bias_dev = self.bias_vec
        def modifier(logits):
            return logits + ALPHA * bias_dev.to(logits.device, logits.dtype)

        phrase_results  = []
        all_free_pieces = []

        for phrase_idx in range(N_PHRASES):
            print(f"\n  Phrase {phrase_idx+1}/{N_PHRASES} — alpha={ALPHA} bias_v2 (suppression-only)")
            self.phrase_conditioner._queue.clear()
            self.phrase_conditioner._bridge_queue.clear()

            lm_gen  = LMGen(self.lm_model, condition_tensors=ct)
            restore = patch_lmgen_step(lm_gen, modifier)

            free_gen_token_ids = []
            free_gen_pieces    = []
            post_bridge_steps  = 0
            in_free_gen        = False
            audio_codes_count  = 0
            silence_secs       = 0.0
            in_phrase          = False
            phrase_buf         = []
            phrase_count       = 0
            chunk_size         = 1920

            with torch.no_grad():
                with lm_gen.streaming(1):
                    with self.mimi.streaming(1):
                        for i in range(0, wav.shape[0] - chunk_size + 1, chunk_size):
                            chunk  = wav[i:i + chunk_size]
                            rms    = chunk.pow(2).mean().sqrt().item()
                            is_sil = rms < 0.01

                            chunk_in = chunk.unsqueeze(0).unsqueeze(0)
                            codes    = self.mimi.encode(chunk_in)
                            audio_codes_count += codes.shape[-1]

                            if is_sil:
                                silence_secs += chunk_size / 24000
                                if (silence_secs >= SILENCE_GATE and
                                        in_phrase and phrase_buf and
                                        phrase_count < 1):
                                    in_phrase    = False
                                    phrase_count += 1

                                    pa = torch.cat(phrase_buf).unsqueeze(0)

                                    if self._mert is None:
                                        from transformers import AutoModel
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

                                was_bridge = (tag == '[B]')
                                if was_bridge:
                                    in_free_gen       = True
                                    post_bridge_steps = 0

                                if in_free_gen and not was_bridge:
                                    if post_bridge_steps < FREE_GEN_STEPS:
                                        free_gen_token_ids.append(tok_id)
                                        free_gen_pieces.append(piece)
                                        post_bridge_steps += 1

                            if in_free_gen and post_bridge_steps >= FREE_GEN_STEPS:
                                break

            restore()
            all_free_pieces.extend(free_gen_pieces)

            latex_count   = sum(1 for tid in free_gen_token_ids if tid in LATEX_TOKEN_IDS)
            trumpet_count = sum(1 for p in free_gen_pieces if p in TRUMPET_PIECES)
            token_seq     = " ".join(free_gen_pieces[:FREE_GEN_STEPS])

            if len(free_gen_pieces) >= 4:
                unique_ratio = len(set(free_gen_pieces)) / len(free_gen_pieces)
                if unique_ratio < 0.1:
                    coherence = "COLLAPSED"
                elif unique_ratio < 0.25:
                    coherence = "REPETITIVE (heavy cycling)"
                elif trumpet_count >= 3:
                    coherence = "PEDAGOGICAL (trumpet vocab present)"
                else:
                    coherence = "MIXED"
            else:
                coherence = "INSUFFICIENT TOKENS"

            phrase_results.append({
                "latex_count":   latex_count,
                "trumpet_count": trumpet_count,
                "coherence":     coherence,
                "token_seq":     token_seq,
                "audio_codes":   audio_codes_count,
                "unique_ratio":  round(len(set(free_gen_pieces)) / max(len(free_gen_pieces), 1), 3),
            })

            print(f"    LaTeX tokens:   {latex_count}/{FREE_GEN_STEPS}")
            print(f"    Trumpet tokens: {trumpet_count}/{FREE_GEN_STEPS}")
            print(f"    Unique ratio:   {phrase_results[-1]['unique_ratio']:.3f}")
            print(f"    Coherence:      {coherence}")
            print(f"    Audio codes:    {audio_codes_count} (continuous={audio_codes_count > 0})")
            print(f"    Token stream:   {token_seq[:100]}...")

        # ── Summary + baseline comparison ─────────────────────────────────────
        avg_latex   = sum(r["latex_count"]   for r in phrase_results) / len(phrase_results)
        avg_trumpet = sum(r["trumpet_count"] for r in phrase_results) / len(phrase_results)
        audio_ok    = all(r["audio_codes"] > 0 for r in phrase_results)
        avg_unique  = sum(r["unique_ratio"]  for r in phrase_results) / len(phrase_results)

        total_free    = len(all_free_pieces)
        trumpet_total = sum(1 for p in all_free_pieces if p in TRUMPET_PIECES)
        density_v2    = trumpet_total / max(total_free, 1)

        print()
        print("=" * 60)
        print("§4 RERUN — bias_v2.pt (suppression-only) alpha=1.0")
        print("=" * 60)
        print(f"  Avg LaTeX tokens (32 steps):   {avg_latex:.1f}  (target: 0)")
        print(f"  Avg trumpet tokens (32 steps):  {avg_trumpet:.1f}  (LoRA-driven)")
        print(f"  Avg unique-token ratio:         {avg_unique:.3f}")
        print(f"  Audio codebook continuous:      {'✓' if audio_ok else '✗ BROKEN'}")
        print()
        print("  Coherence per phrase:")
        for i, r in enumerate(phrase_results):
            print(f"    phrase {i+1}: {r['coherence']}  (unique_ratio={r['unique_ratio']:.3f})")
        print()
        print("  BASELINE COMPARISON (lora_v3/best streaming, no bias):")
        print(f"    Trumpet density:  {BASELINE_TRUMPET_DENSITY:.2%}  "
              f"({BASELINE_TRUMPET_HITS}/{BASELINE_TOTAL_TOKENS} tokens)")
        print(f"    Attractor hits: 0  |  Space cycling: absent  |  Mr cycling: absent")
        print()
        print(f"  bias_v2 trumpet density:        {density_v2:.2%}  "
              f"({trumpet_total}/{total_free} free-gen tokens)")
        print(f"  Delta vs baseline:              {density_v2 - BASELINE_TRUMPET_DENSITY:+.2%}")
        print()
        print("  GATE CRITERIA:")
        print(f"    LaTeX eliminated:   {'✓' if avg_latex == 0.0 else '✗'}  ({avg_latex:.1f})")
        print(f"    Audio continuous:   {'✓' if audio_ok else '✗'}")
        print(f"    Unique ratio:       {avg_unique:.3f}")
        print()
        print("  Return to Muse. Do not proceed to §5 without confirmation.")
        print("=" * 60)

        return {
            "avg_latex":        avg_latex,
            "avg_trumpet":      avg_trumpet,
            "avg_unique_ratio": avg_unique,
            "audio_ok":         audio_ok,
            "trumpet_density":  density_v2,
            "phrase_results":   phrase_results,
        }


@app.local_entrypoint()
def main():
    CalibrateV2().calibrate.remote()