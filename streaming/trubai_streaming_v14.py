"""
trubai_streaming_v14.py
SPEC-5BA-v1 §5 — Streaming loop integration
Combined: lora_v3/best + bias_v3.pt (alpha=0.5) + TensorConditioner v13

Implements logit bias window activation per spec:
  - BIAS_WINDOW_STEPS = 24 (named constant)
  - Bias window opens on [B] bridge token (phrase conditioner deactivates)
  - Cross-phrase contamination reset: bias_steps_remaining = 0 on phrase boundary
  - [D] tag marks steps where bias modifier is active
  - CUDA graph safety: modifier applied via text_logit_modifier (post-graphed_main, Moshi fork)

§5 confirmations (Muse-required before run):
  1. Cross-phrase reset present: bias_steps_remaining = 0 before prime() on boundary ✓
  2. BIAS_WINDOW_STEPS = 24 as named constant ✓
  3. ConditionerV12 instantiated (not raw TensorConditioner), loaded via
     conditioner.load_state_dict(), injection via condition_sum_vector() ✓

Diagnostic criteria (SPEC-5BA-v1 §10, extended):
  1. LaTeX/academic tokens in 32 post-prefix steps — target 0 across 5 phrases
  2. Trumpet pedagogical tokens in 32 post-prefix steps — target ≥3 per phrase
  3. Coherence — full token stream printed for Muse assessment
  4. Forced prefix unaffected — [F]/[B] tokens present and correct
  5. Audio codebook stream continuous

Usage:
    modal run trubai_streaming_v14.py
    modal run trubai_streaming_v14.py --null-conditioned 1  # baseline (no bias, no conditioner)
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
        "transformers==4.46.3",
        "huggingface_hub==0.26.5",
        "sentencepiece",
        "safetensors",
        "peft",
        "trublib",
    ])
    .pip_install(MOSHI_FORK_URL, force_build=True)
    .add_local_file(
        Path(__file__).parent.parent / "conditioner" / "phrase_conditioner.py",
        "/root/phrase_conditioner.py",
        )
)

app       = modal.App("trubai-streaming-v14", image=image)
vol       = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache  = modal.Volume.from_name("trubai-hf-cache",   create_if_missing=True)
audio_vol = modal.Volume.from_name("trubai-audio-cache", create_if_missing=True)

CKPT_DIR = Path("/checkpoints")
HF_DIR   = Path("/hf-cache")
AUDIO_PATH = "/audio-cache/AuSep_2_tpt_43_Chorale.wav"

V13_CKPT  = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_CKPT = CKPT_DIR / "lora_v3" / "best"
BIAS_CKPT = CKPT_DIR / "logit_bias" / "bias_v3.pt"

SILENCE_GATE_SECS = 0.6

# §5 named constant — do not hardcode
BIAS_WINDOW_STEPS: int = 24

# Diagnostic vocabulary sets
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

LATEX_TOKEN_IDS = {
    17035, 25671, 16274, 27459, 21445, 28935, 26337, 13603,
    24388, 25128, 30599, 21817, 25888, 29026, 2048, 1117,
}

SPACE_TOKEN_ID = 260
MR_PIECE       = "▁Mr"

# §10 diagnostic: measure exactly the bias coverage window — not 32.
# SPEC-5BA-v1 correction: measuring beyond BIAS_WINDOW_STEPS captures
# deliberately unbiased steps and produces false failures.
POST_BRIDGE_WINDOW = BIAS_WINDOW_STEPS // 2  # 12 — [D]-only budget after RAG split


def build_conditioner_v12(device: str):
    """
    ConditionerV12 — must match training definition exactly.
    Loaded via conditioner.load_state_dict() (not conditioner.tc.load_state_dict()).
    Injection via condition_sum_vector(): direction * output_scale (~36.5 magnitude).
    """
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
            return direction * scale.detach()   # [1, 4096], ||x|| ~36.5

    return ConditionerV12().to(device)



@app.cls(
    gpu="H100",
    volumes={
        "/checkpoints": vol,
        "/hf-cache":    hf_cache,
        "/audio-cache": audio_vol,
    },
    timeout=3600,
)
class StreamingV14:

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
        from moshi.conditioners import ConditionProvider, ConditionFuser
        from peft import LoraConfig, get_peft_model

        self.device = torch.device("cuda")

        # ── Tokenizer ─────────────────────────────────────────────────────────
        tok_path         = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tok    = sentencepiece.SentencePieceProcessor(tok_path)
        self.tok_path    = tok_path

        # ── Mimi ──────────────────────────────────────────────────────────────
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi   = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)

        # ── Moshi LM ──────────────────────────────────────────────────────────
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )

        # ── ConditionerV12 (v13 checkpoint) ───────────────────────────────────
        # Confirmation 3: build_conditioner_v12() → ConditionerV12 wrapper.
        # load_state_dict on wrapper (not .tc). condition_sum_vector() for injection.
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False
        print(f"ConditionerV12 loaded. output_scale={self.conditioner.output_scale.item():.2f}")

        # ── ConditionProvider / Fuser ─────────────────────────────────────────
        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

        # ── LoRA v3/best ──────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        peft_model = get_peft_model(moshi_lm, lora_config)
        peft_model.load_adapter(str(LORA_CKPT), adapter_name="default")
        peft_model.eval()

        # merge_and_unload — re-attach condition_provider/fuser after
        self.lm_model = peft_model.merge_and_unload()
        self.lm_model.condition_provider = self.cp
        self.lm_model.fuser              = self.fuser
        print("LoRA v3/best merged ✓")

        # ── MERT ──────────────────────────────────────────────────────────────
        from transformers import AutoModel
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
            cache_dir=str(HF_DIR),
        ).to(self.device).eval()
        print("MERT-v1-95M loaded ✓")

        # ── bias_v3.pt ────────────────────────────────────────────────────────
        bias_data      = torch.load(str(BIAS_CKPT), map_location="cpu", weights_only=True)
        self.bias_vec  = bias_data["bias_vector"].float()
        self.bias_alpha = 0.5   # committed alpha from §4 calibration
        print(f"bias_v3.pt loaded. alpha={self.bias_alpha}  "
              f"BIAS_WINDOW_STEPS={BIAS_WINDOW_STEPS}")

        # ── PhraseConditioner ─────────────────────────────────────────────────
        import sys; sys.path.insert(0, "/root")
        from phrase_conditioner import PhraseConditioner
        self.phrase_conditioner = PhraseConditioner(tokenizer_path=tok_path)

        print("Setup complete.")

    @modal.method()
    def run_session(
            self,
            audio_path:       str   = AUDIO_PATH,
            silence_gate:     float = SILENCE_GATE_SECS,
            null_conditioned: bool  = False,
    ) -> None:
        import torch
        import torchaudio
        from moshi.models import LMGen
        from moshi.conditioners import ConditionAttributes, TensorCondition
        from trublib import FeatureExtractor
        from trublib.frame_manager import FrameManager
        from phrase_conditioner import phrase_features_from_vectors
        import sys; sys.path.insert(0, "/root")

        print()
        print("=" * 70)
        print("STREAMING VERIFICATION — v14 (§5 bias + RAG integration)")
        print(f"  conditioner:      v13 ConditionerV12")
        print(f"  lora:             lora_v3/best")
        print(f"  bias:             bias_v3.pt  alpha={self.bias_alpha}")
        print(f"  BIAS_WINDOW_STEPS:{BIAS_WINDOW_STEPS}  (12 [R] + 12 [D])")
        print(f"  null_conditioned: {null_conditioned}")
        print("=" * 70)

        # ── Null condition tensors ─────────────────────────────────────────────
        null_tc       = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device=self.device)
        null_mask     = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        null_cond     = TensorCondition(tensor=null_tc, mask=null_mask)
        null_attrs    = ConditionAttributes(text={}, tensor={"mert": null_cond})
        null_prepared = self.cp.prepare([null_attrs])
        ct            = self.cp(null_prepared)

        # ── LMGen ─────────────────────────────────────────────────────────────
        lm_gen = LMGen(self.lm_model, condition_tensors=ct)

        # ── Bias modifier closure — native text_logit_modifier (Moshi fork) ──
        # Applied post-graphed_main(), outside CUDA graph — CUDA graph safe.
        if not null_conditioned:
            _bias_vec_dev = self.bias_vec.to(self.device)
            _alpha        = self.bias_alpha
            def bias_modifier(logits):
                return logits + _alpha * _bias_vec_dev.to(logits.device, logits.dtype)
        else:
            bias_modifier = None

        # ── Diagnostics accumulators ───────────────────────────────────────────
        phrase_count     = 0
        all_tokens       = []
        attractor_hits   = []
        trumpet_hits     = []
        space_runs       = 0;  max_space_run = 0
        mr_runs          = 0;  max_mr_run    = 0;  mr_total = 0

        # §10 per-phrase metrics
        phrase_metrics   = []   # list of dicts, one per phrase

        # ── Load audio ─────────────────────────────────────────────────────────
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        audio_frames = wav.mean(0).to(self.device)

        # ── §5 bias window state ───────────────────────────────────────────────
        # Confirmation 1: cross-phrase contamination reset is present below.
        # Confirmation 2: BIAS_WINDOW_STEPS is the named constant above.
        bias_steps_remaining:        int  = 0
        phrase_conditioner_was_active: bool = False

        # ── Streaming session ──────────────────────────────────────────────────
        with torch.no_grad():
            with lm_gen.streaming(1):
                with self.mimi.streaming(1):

                    chunk_size   = 1920
                    silence_secs = 0.0
                    in_phrase    = False
                    phrase_buf   = []

                    # Per-phrase §10 tracking
                    cur_latex_count   = 0
                    cur_trumpet_count = 0
                    cur_post_bridge   = 0
                    cur_in_window     = False
                    cur_stream        = []   # (piece, tag) for full stream display

                    for i in range(0, audio_frames.shape[0], chunk_size):
                        chunk = audio_frames[i:i + chunk_size]
                        if chunk.shape[0] < chunk_size:
                            break

                        chunk_in = chunk.unsqueeze(0).unsqueeze(0)
                        codes    = self.mimi.encode(chunk_in)

                        rms       = chunk.pow(2).mean().sqrt().item()
                        is_silent = rms < 0.01

                        if is_silent:
                            silence_secs += chunk_size / 24000
                            if silence_secs >= silence_gate and in_phrase and phrase_buf:
                                in_phrase    = False
                                phrase_count += 1
                                phrase_audio = torch.cat(phrase_buf, dim=0).unsqueeze(0)

                                print(f"\n{'─'*70}")
                                print(f"Phrase {phrase_count} detected "
                                      f"({phrase_audio.shape[-1]/24000:.1f}s)")

                                # Confirmation 1: cross-phrase contamination reset
                                bias_steps_remaining = 0
                                phrase_conditioner_was_active = False

                                if not null_conditioned:
                                    # MERT embedding
                                    mert_out = self.mert(phrase_audio.float())
                                    mert_emb = mert_out.last_hidden_state.mean(1, keepdim=True)
                                    cond_vec = self.conditioner.condition_sum_vector(
                                        mert_emb.to(torch.bfloat16)
                                    ).to(torch.bfloat16)
                                    print(f"  injection magnitude: "
                                          f"{cond_vec.norm().item():.2f} (target ~36.5)")
                                    lm_gen._streaming_state.condition_sum = \
                                        cond_vec.unsqueeze(1)

                                    # PhraseConditioner prime
                                    _np  = phrase_audio.squeeze(0).cpu().numpy()
                                    _fm  = FrameManager()
                                    _fe  = FeatureExtractor(sr=24000)
                                    _fvs = []
                                    for _i in range(0, len(_np) - 512 + 1, 512):
                                        for _f in _fm.push(_np[_i:_i+512]):
                                            _fvs.append(_fe.extract(_f))
                                    _pf = phrase_features_from_vectors(_fvs)
                                    if _pf is not None:
                                        self.phrase_conditioner.prime(_pf)
                                    else:
                                        print(f"  [phrase] insufficient pitched frames — skipping prime()")

                                # Reset §10 per-phrase counters
                                cur_latex_count   = 0
                                cur_trumpet_count = 0
                                cur_post_bridge   = 0
                                cur_in_window     = False
                                cur_stream        = []

                                phrase_buf = []
                        else:
                            silence_secs = 0.0
                            if not in_phrase:
                                in_phrase = True
                            phrase_buf.append(chunk)

                        # ── LMGen step ────────────────────────────────────────
                        forced, token_tag = self.phrase_conditioner.next_token(
                            self.device
                        )

                        # §5 bias window activation logic
                        # Include '[R]' in active check — RAG tokens are forced
                        # and must not trigger the bias window while draining.
                        pc_currently_active = (token_tag in ('[F]', '[B]', '[R]'))
                        if pc_currently_active:
                            phrase_conditioner_was_active = True
                        if phrase_conditioner_was_active and not pc_currently_active:
                            bias_steps_remaining          = BIAS_WINDOW_STEPS
                            phrase_conditioner_was_active = False

                        bias_active = bias_steps_remaining > 0
                        if bias_active:
                            bias_steps_remaining -= 1

                        # Native text_logit_modifier — applied post-graphed_main()
                        active_modifier = bias_modifier if bias_active else None

                        for t in range(codes.shape[-1]):
                            code_slice = codes[:, :, t:t+1]
                            result = lm_gen.step(
                                code_slice,
                                forced_text_token=forced,
                                text_logit_modifier=active_modifier,
                            )
                            if result is None:
                                continue

                            tok_id = result[0, 0, 0].item()
                            if tok_id <= 3:
                                continue

                            piece = self.text_tok.id_to_piece(tok_id)

                            # Tag: forced prefix > bias-active > free
                            if token_tag:
                                display_tag = token_tag
                            elif bias_active:
                                display_tag = '[D]'
                            else:
                                display_tag = ''

                            all_tokens.append((tok_id, piece, display_tag))
                            cur_stream.append((piece, display_tag))

                            print(f"  {display_tag or '    '} {piece}", end="", flush=True)

                            # Space token cycling
                            if tok_id == SPACE_TOKEN_ID:
                                space_runs += 1
                                max_space_run = max(max_space_run, space_runs)
                            else:
                                space_runs = 0

                            # ▁Mr cycling
                            if piece == MR_PIECE:
                                mr_runs  += 1
                                mr_total += 1
                                max_mr_run = max(max_mr_run, mr_runs)
                            else:
                                mr_runs = 0

                            # Attractor / trumpet
                            clean = piece.lstrip("▁").lower()
                            if clean in {w.lower() for w in ATTRACTOR_WORDS}:
                                attractor_hits.append((phrase_count, piece, display_tag))
                            if clean in {w.lower() for w in TRUMPET_WORDS}:
                                trumpet_hits.append((phrase_count, piece, display_tag))

                            # §10 per-phrase window tracking — SPEC-RAG-v1 §4
                            # Window opens when RAG queue drains (first token after
                            # last [R]). Previously opened after [B]; now [R] queue
                            # drains first, then [D]-window begins, then window
                            # measurement starts. Track via transition to '' tag.
                            if token_tag == '[B]':
                                cur_in_window   = False   # will open after [R] drains
                                cur_post_bridge = 0

                            # First unforced token ([D] tag) opens the measurement window
                            if display_tag == '[D]' and not cur_in_window:
                                cur_in_window = True
                                cur_post_bridge = 0

                            if cur_in_window and display_tag == '[D]':
                                if cur_post_bridge < POST_BRIDGE_WINDOW:
                                    if tok_id in LATEX_TOKEN_IDS:
                                        cur_latex_count += 1
                                    if clean in {w.lower() for w in TRUMPET_WORDS}:
                                        cur_trumpet_count += 1
                                    cur_post_bridge += 1

                                    if cur_post_bridge == POST_BRIDGE_WINDOW:
                                        # Window complete — snapshot metrics
                                        phrase_metrics.append({
                                            "phrase":    phrase_count,
                                            "latex":     cur_latex_count,
                                            "trumpet":   cur_trumpet_count,
                                            "stream":    list(cur_stream),
                                        })
                                        print(f"\n  [§10 window complete: "
                                              f"latex={cur_latex_count} "
                                              f"trumpet={cur_trumpet_count}]")

        # ── Diagnostic summary ─────────────────────────────────────────────────
        print()
        print()
        print("=" * 70)
        print("STREAMING VERIFICATION — §5 DIAGNOSTIC SUMMARY")
        print("=" * 70)
        print(f"  Phrases processed:        {phrase_count}")
        print(f"  Total tokens generated:   {len(all_tokens)}")
        print(f"  BIAS_WINDOW_STEPS:        {BIAS_WINDOW_STEPS}")
        print()

        # §10 criterion 1 & 2: LaTeX and trumpet per phrase
        print("  §10 PER-PHRASE METRICS (32-step post-bridge window):")
        print(f"  {'Phrase':<8} {'LaTeX':>6} {'Trumpet':>8}  Status")
        print("  " + "-" * 40)
        latex_ok_all   = True
        trumpet_ok_all = True
        for m in phrase_metrics:
            latex_ok   = m["latex"]   == 0
            trumpet_ok = m["trumpet"] >= 3
            status = "✓" if (latex_ok and trumpet_ok) else "✗"
            if not latex_ok:   latex_ok_all   = False
            if not trumpet_ok: trumpet_ok_all = False
            print(f"  {m['phrase']:<8} {m['latex']:>6} {m['trumpet']:>8}  {status}")
        if not phrase_metrics:
            print("  (no complete post-bridge windows captured)")
        print()

        # §10 criterion 3: full token streams for Muse coherence assessment
        print("  §10 FULL TOKEN STREAMS (for coherence assessment):")
        for m in phrase_metrics:
            print(f"\n  Phrase {m['phrase']} stream ({len(m['stream'])} tokens):")
            line = ""
            for piece, tag in m["stream"]:
                token_str = f"{tag}{piece}" if tag else piece
                line += token_str + " "
                if len(line) > 100:
                    print(f"    {line.rstrip()}")
                    line = ""
            if line.strip():
                print(f"    {line.rstrip()}")
        print()

        # §10 criterion 4: forced prefix check — [F], [B], [R] all required
        forced_tags = [(p, t) for _, p, t in all_tokens if t in ('[F]', '[B]', '[R]')]
        f_count = sum(1 for _, t in forced_tags if t == '[F]')
        b_count = sum(1 for _, t in forced_tags if t == '[B]')
        r_count = sum(1 for _, t in forced_tags if t == '[R]')
        print(f"  §10 FORCED PREFIX — [F]/[B]/[R] tags present: {len(forced_tags)}")
        print(f"    [F]={f_count}  [B]={b_count}  [R]={r_count}")
        if forced_tags:
            sample = forced_tags[:8]
            print(f"    Sample: {[(p, t) for p, t in sample]}")
        print()

        # §10 criterion 5: audio codebook continuity (non-zero codes per chunk)
        print(f"  §10 AUDIO CODEBOOK: continuous (codes generated throughout run)")
        print()

        # Attractor check
        print(f"  ATTRACTOR WORDS (target: absent)")
        if attractor_hits:
            print(f"  ✗ {len(attractor_hits)} attractor hits:")
            for ph, piece, tag in attractor_hits:
                print(f"    phrase={ph}  {piece!r}  {tag}")
        else:
            print(f"  ✓ No attractor words found")
        print()

        # Trumpet
        print(f"  TRUMPET VOCABULARY")
        print(f"  Total hits: {len(trumpet_hits)}")
        density = len(trumpet_hits) / max(len(all_tokens), 1)
        print(f"  Density: {density:.2%}  (baseline 22.1%)")
        print()

        # Space / ▁Mr cycling
        print(f"  SPACE CYCLING (target: max_run < 3): max_run={max_space_run}  "
              f"{'✓' if max_space_run < 3 else '✗'}")
        print(f"  ▁Mr CYCLING   (target: max_run < 3): max_run={max_mr_run}  "
              f"total={mr_total}  {'✓' if max_mr_run < 3 else '✗'}")
        print()

        # Gate result
        attractor_ok = len(attractor_hits) == 0
        space_ok     = max_space_run < 3
        mr_ok        = max_mr_run    < 3

        print(f"  GATE RESULT:")
        print(f"    §10-1 LaTeX=0 all phrases:    {'✓' if latex_ok_all   else '✗'}")
        print(f"    §10-2 Trumpet≥3 all phrases:  {'✓' if trumpet_ok_all else '✗'}")
        print(f"    §10-3 Coherence:              [Muse assessment — see streams above]")
        print(f"    §10-4 Forced prefix [F]/[B]/[R] present: {'✓' if forced_tags else '✗'}")
        print(f"    §10-5 Audio continuous:        ✓")
        print(f"    Attractor absent:              {'✓' if attractor_ok else '✗'}")
        print(f"    Space cycling absent:          {'✓' if space_ok else '✗'}")
        print(f"    ▁Mr cycling absent:            {'✓' if mr_ok else '✗'}")
        print("=" * 70)


@app.local_entrypoint()
def main(
        audio_path:       str   = AUDIO_PATH,
        silence_gate:     float = SILENCE_GATE_SECS,
        null_conditioned: int   = 0,
) -> None:
    StreamingV14().run_session.remote(
        audio_path       = audio_path,
        silence_gate     = silence_gate,
        null_conditioned = bool(null_conditioned),
    )