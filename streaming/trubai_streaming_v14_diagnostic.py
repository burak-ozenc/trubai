"""
trubai_streaming_v14_diagnostic.py
SPEC-DIAGNOSTIC-SPOKEN-v1 — Phase 5 Spoken Output Diagnostic

Diagnostic copy of trubai_streaming_v14.py. DO NOT use as production stack.
trubai_streaming_v14.py is unchanged.

What this does:
  - Injects rich inner monologue scaffold at session start (before any phrase boundary)
  - Tags scaffold tokens [S] in diagnostic output
  - Records 30 seconds of Moshi spoken audio output → diagnostic_output.wav
  - Dumps full token stream → diagnostic_token_stream.txt

Report to Muse:
  1. diagnostic_token_stream.txt
  2. diagnostic_output.wav
  3. Manual transcription of spoken audio
  4. Any crashes or unexpected behavior

Muse assesses — Builder 2 (Lituus) reports raw results only.
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
        "soundfile",
    ])
    .pip_install(MOSHI_FORK_URL, force_build=True)
    .add_local_file(
        Path(__file__).parent.parent / "conditioner" / "phrase_conditioner.py",
        "/root/phrase_conditioner.py",
        )
)

app       = modal.App("trubai-streaming-v14-diagnostic", image=image)
vol       = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache  = modal.Volume.from_name("trubai-hf-cache",   create_if_missing=True)
audio_vol = modal.Volume.from_name("trubai-audio-cache", create_if_missing=True)

CKPT_DIR   = Path("/checkpoints")
HF_DIR     = Path("/hf-cache")
AUDIO_PATH = "/audio-cache/AuSep_2_tpt_43_Chorale.wav"

V13_CKPT  = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_CKPT = CKPT_DIR / "lora_v3" / "best"
BIAS_CKPT = CKPT_DIR / "logit_bias" / "bias_v3.pt"

SILENCE_GATE_SECS = 0.6
BIAS_WINDOW_STEPS: int = 24

# ── [DIAGNOSTIC] Scaffold definition (SPEC-DIAGNOSTIC-SPOKEN-v1) ──────────────
SCAFFOLD_TEXT = (
    "Student has played three phrases. "
    "First phrase: pitch low, tone low. "
    "Second phrase: pitch low, tone medium. "
    "Third phrase: pitch medium, tone low. "
    "Current observation: the air column loses support before the pitch stabilizes."
)
DIAGNOSTIC_DURATION_SEC = 30
DIAGNOSTIC_SAMPLE_RATE  = 24000
DIAGNOSTIC_AUDIO_OUT    = "/audio-cache/diagnostic_output.wav"
DIAGNOSTIC_TOKEN_OUT    = "/audio-cache/diagnostic_token_stream.txt"
# ─────────────────────────────────────────────────────────────────────────────

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
POST_BRIDGE_WINDOW = BIAS_WINDOW_STEPS // 2


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


@app.cls(
    gpu="H100",
    volumes={
        "/checkpoints": vol,
        "/hf-cache":    hf_cache,
        "/audio-cache": audio_vol,
    },
    timeout=3600,
)
class StreamingV14Diagnostic:

    @modal.enter()
    def setup(self):
        # Identical to v14 setup — no changes
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

        tok_path         = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tok    = sentencepiece.SentencePieceProcessor(tok_path)
        self.tok_path    = tok_path

        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.mimi   = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)

        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )

        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False

        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

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

        from transformers import AutoModel
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
            cache_dir=str(HF_DIR),
        ).to(self.device).eval()

        bias_data       = torch.load(str(BIAS_CKPT), map_location="cpu", weights_only=True)
        self.bias_vec   = bias_data["bias_vector"].float()
        self.bias_alpha = 0.5

        import sys; sys.path.insert(0, "/root")
        from phrase_conditioner import PhraseConditioner
        self.phrase_conditioner = PhraseConditioner(tokenizer_path=tok_path)

        print("Setup complete.")

    @modal.method()
    def run_diagnostic(
            self,
            audio_path:   str   = AUDIO_PATH,
            silence_gate: float = SILENCE_GATE_SECS,
    ) -> None:
        import torch
        import torchaudio
        import soundfile as sf
        import numpy as np
        from moshi.models import LMGen
        from moshi.conditioners import ConditionAttributes, TensorCondition
        from trublib import FeatureExtractor
        from trublib.frame_manager import FrameManager
        from phrase_conditioner import phrase_features_from_vectors
        import sys; sys.path.insert(0, "/root")

        print()
        print("=" * 70)
        print("DIAGNOSTIC RUN — SPEC-DIAGNOSTIC-SPOKEN-v1")
        print(f"  Scaffold: {SCAFFOLD_TEXT[:60]}...")
        print(f"  Recording: {DIAGNOSTIC_DURATION_SEC}s of spoken output")
        print("=" * 70)

        # ── Condition tensors (identical to v14) ──────────────────────────────
        null_tc       = torch.zeros(1, 1, 768, dtype=torch.bfloat16, device=self.device)
        null_mask     = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        null_cond     = TensorCondition(tensor=null_tc, mask=null_mask)
        null_attrs    = ConditionAttributes(text={}, tensor={"mert": null_cond})
        null_prepared = self.cp.prepare([null_attrs])
        ct            = self.cp(null_prepared)

        lm_gen = LMGen(self.lm_model, condition_tensors=ct)

        _bias_vec_dev = self.bias_vec.to(self.device)
        _alpha        = self.bias_alpha
        def bias_modifier(logits):
            return logits + _alpha * _bias_vec_dev.to(logits.device, logits.dtype)

        # ── [DIAGNOSTIC] Token stream log ─────────────────────────────────────
        token_log_lines = []   # (tag, tok_id, piece)

        # ── [DIAGNOSTIC] Audio capture state ──────────────────────────────────
        diag_audio_chunks        = []
        diag_samples_remaining   = DIAGNOSTIC_DURATION_SEC * DIAGNOSTIC_SAMPLE_RATE
        diag_recording           = True   # starts True — record from first step
        diag_recording_complete  = False

        # ── Load audio ────────────────────────────────────────────────────────
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        audio_frames = wav.mean(0).to(self.device)

        # ── v14 diagnostics accumulators ─────────────────────────────────────
        phrase_count   = 0
        all_tokens     = []
        attractor_hits = []
        trumpet_hits   = []
        space_runs = 0; max_space_run = 0
        mr_runs    = 0; max_mr_run    = 0; mr_total = 0
        phrase_metrics = []

        bias_steps_remaining         = 0
        phrase_conditioner_was_active = False

        with torch.no_grad():
            with lm_gen.streaming(1):
                with self.mimi.streaming(1):

                    # ── [DIAGNOSTIC] Scaffold injection ───────────────────────
                    # Inject before any phrase boundary fires.
                    # Tokenize scaffold, force each token through lm_gen.step()
                    # using a zero dummy code slice. Tag all [S].
                    #
                    # Dummy codes: shape [1, 8, 1], zeros (silent frame).
                    # lm_gen.step() accepts forced_text_token — forces the token
                    # into the inner monologue channel exactly as [F]/[R] tokens.
                    # text_logit_modifier is None during scaffold (no bias).

                    scaffold_token_ids = self.text_tok.encode(SCAFFOLD_TEXT)
                    dummy_codes = torch.zeros(
                        1, 8, 1, dtype=torch.long, device=self.device
                    )

                    print(f"\n[DIAGNOSTIC] Injecting scaffold — "
                          f"{len(scaffold_token_ids)} tokens")

                    for s_tok in scaffold_token_ids:
                        s_tok_tensor = torch.tensor([s_tok], device=self.device)
                        result = lm_gen.step(
                            dummy_codes,
                            forced_text_token=s_tok_tensor,
                            text_logit_modifier=None,
                        )
                        if result is None:
                            token_log_lines.append(f"[S] {s_tok} (step returned None)")
                            continue

                        # Capture generated text token (inner monologue channel)
                        gen_tok_id = result[0, 0, 0].item()
                        piece      = self.text_tok.id_to_piece(gen_tok_id)
                        token_log_lines.append(f"[S] {s_tok} → gen={gen_tok_id} {piece!r}")
                        print(f"  [S] forced={s_tok} → {piece}", flush=True)

                        # Audio capture from scaffold steps
                        if diag_recording and not diag_recording_complete:
                            # Decode audio codebooks from result: result shape [B, K+1, T]
                            # Row 0 = text, rows 1: = audio codebooks
                            audio_codes = result[:, 1:, :]   # [1, 8, 1]
                            if audio_codes.shape[1] > 0:
                                try:
                                    audio_out = self.mimi.decode(audio_codes)
                                    chunk_np  = audio_out.squeeze().cpu().float().numpy()
                                    diag_audio_chunks.append(chunk_np)
                                    diag_samples_remaining -= len(chunk_np)
                                    if diag_samples_remaining <= 0:
                                        diag_recording_complete = True
                                        diag_recording          = False
                                except Exception as e:
                                    print(f"  [DIAGNOSTIC] audio decode error: {e}")

                    print(f"[DIAGNOSTIC] Scaffold injection complete\n")
                    # ── [END DIAGNOSTIC scaffold injection] ───────────────────

                    # ── Main streaming loop (v14, unchanged) ──────────────────
                    chunk_size   = 1920
                    silence_secs = 0.0
                    in_phrase    = False
                    phrase_buf   = []

                    cur_latex_count   = 0
                    cur_trumpet_count = 0
                    cur_post_bridge   = 0
                    cur_in_window     = False
                    cur_stream        = []

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

                                bias_steps_remaining          = 0
                                phrase_conditioner_was_active = False

                                mert_out = self.mert(phrase_audio.float())
                                mert_emb = mert_out.last_hidden_state.mean(1, keepdim=True)
                                cond_vec = self.conditioner.condition_sum_vector(
                                    mert_emb.to(torch.bfloat16)
                                ).to(torch.bfloat16)
                                lm_gen._streaming_state.condition_sum = \
                                    cond_vec.unsqueeze(1)

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

                                cur_latex_count   = 0
                                cur_trumpet_count = 0
                                cur_post_bridge   = 0
                                cur_in_window     = False
                                cur_stream        = []
                                phrase_buf        = []
                        else:
                            silence_secs = 0.0
                            if not in_phrase:
                                in_phrase = True
                            phrase_buf.append(chunk)

                        forced, token_tag = self.phrase_conditioner.next_token(
                            self.device
                        )

                        pc_currently_active = (token_tag in ('[F]', '[B]', '[R]'))
                        if pc_currently_active:
                            phrase_conditioner_was_active = True
                        if phrase_conditioner_was_active and not pc_currently_active:
                            bias_steps_remaining          = BIAS_WINDOW_STEPS
                            phrase_conditioner_was_active = False

                        bias_active = bias_steps_remaining > 0
                        if bias_active:
                            bias_steps_remaining -= 1

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

                            if token_tag:
                                display_tag = token_tag
                            elif bias_active:
                                display_tag = '[D]'
                            else:
                                display_tag = ''

                            all_tokens.append((tok_id, piece, display_tag))
                            cur_stream.append((piece, display_tag))

                            # [DIAGNOSTIC] log every token from main loop
                            token_log_lines.append(
                                f"{display_tag or '[ ]'} {tok_id} {piece!r}"
                            )

                            print(f"  {display_tag or '    '} {piece}", end="", flush=True)

                            if tok_id == SPACE_TOKEN_ID:
                                space_runs += 1
                                max_space_run = max(max_space_run, space_runs)
                            else:
                                space_runs = 0

                            if piece == MR_PIECE:
                                mr_runs  += 1; mr_total += 1
                                max_mr_run = max(max_mr_run, mr_runs)
                            else:
                                mr_runs = 0

                            clean = piece.lstrip("▁").lower()
                            if clean in {w.lower() for w in ATTRACTOR_WORDS}:
                                attractor_hits.append((phrase_count, piece, display_tag))
                            if clean in {w.lower() for w in TRUMPET_WORDS}:
                                trumpet_hits.append((phrase_count, piece, display_tag))

                            if token_tag == '[B]':
                                cur_in_window   = False
                                cur_post_bridge = 0

                            if display_tag == '[D]' and not cur_in_window:
                                cur_in_window   = True
                                cur_post_bridge = 0

                            if cur_in_window and display_tag == '[D]':
                                if cur_post_bridge < POST_BRIDGE_WINDOW:
                                    if tok_id in LATEX_TOKEN_IDS:
                                        cur_latex_count += 1
                                    if clean in {w.lower() for w in TRUMPET_WORDS}:
                                        cur_trumpet_count += 1
                                    cur_post_bridge += 1

                                    if cur_post_bridge == POST_BRIDGE_WINDOW:
                                        phrase_metrics.append({
                                            "phrase":  phrase_count,
                                            "latex":   cur_latex_count,
                                            "trumpet": cur_trumpet_count,
                                            "stream":  list(cur_stream),
                                        })

                            # [DIAGNOSTIC] Capture audio output from main loop
                            if diag_recording and not diag_recording_complete:
                                audio_codes = result[:, 1:, :]
                                if audio_codes.shape[1] > 0:
                                    try:
                                        audio_out = self.mimi.decode(audio_codes)
                                        chunk_np  = audio_out.squeeze().cpu().float().numpy()
                                        diag_audio_chunks.append(chunk_np)
                                        diag_samples_remaining -= len(chunk_np)
                                        if diag_samples_remaining <= 0:
                                            diag_recording_complete = True
                                            diag_recording          = False
                                            print(f"\n[DIAGNOSTIC] 30s audio capture complete")
                                    except Exception as e:
                                        print(f"  [DIAGNOSTIC] audio decode error: {e}")

        # ── [DIAGNOSTIC] Write outputs ─────────────────────────────────────────
        # Token stream
        with open(DIAGNOSTIC_TOKEN_OUT, "w") as f:
            f.write(f"SPEC-DIAGNOSTIC-SPOKEN-v1 — Token Stream\n")
            f.write(f"Scaffold: {SCAFFOLD_TEXT}\n")
            f.write("─" * 70 + "\n")
            f.write("Tag      tok_id  piece\n")
            f.write("─" * 70 + "\n")
            for line in token_log_lines:
                f.write(line + "\n")
        print(f"\n[DIAGNOSTIC] Token stream → {DIAGNOSTIC_TOKEN_OUT}")

        # Audio
        if diag_audio_chunks:
            full_audio = np.concatenate(diag_audio_chunks)
            sf.write(DIAGNOSTIC_AUDIO_OUT, full_audio, DIAGNOSTIC_SAMPLE_RATE)
            duration = len(full_audio) / DIAGNOSTIC_SAMPLE_RATE
            print(f"[DIAGNOSTIC] Audio ({duration:.1f}s) → {DIAGNOSTIC_AUDIO_OUT}")
        else:
            print("[DIAGNOSTIC] WARNING: No audio chunks captured.")

        # Commit volume so files persist after container exits
        audio_vol.commit()
        print("[DIAGNOSTIC] Volume committed — files available in trubai-audio-cache")

        # ── v14 diagnostic summary (unchanged) ────────────────────────────────
        print()
        print("=" * 70)
        print("v14 DIAGNOSTIC SUMMARY (standard)")
        print("=" * 70)
        print(f"  Phrases processed:      {phrase_count}")
        print(f"  Total tokens generated: {len(all_tokens)}")
        forced_tags = [(p, t) for _, p, t in all_tokens if t in ('[F]', '[B]', '[R]')]
        f_count = sum(1 for _, t in forced_tags if t == '[F]')
        b_count = sum(1 for _, t in forced_tags if t == '[B]')
        r_count = sum(1 for _, t in forced_tags if t == '[R]')
        print(f"  Forced [F]={f_count} [B]={b_count} [R]={r_count}")
        print(f"  Attractor hits: {len(attractor_hits)}")
        print(f"  Trumpet hits:   {len(trumpet_hits)}")
        print()
        print("  [DIAGNOSTIC OUTPUTS]")
        print(f"    Token stream : {DIAGNOSTIC_TOKEN_OUT}")
        print(f"    Audio output : {DIAGNOSTIC_AUDIO_OUT}")
        print(f"    Scaffold [S] tokens: {len(scaffold_token_ids)}")
        print()
        print("  Transcribe diagnostic_output.wav manually.")
        print("  Report token stream + transcription + crashes to Muse.")
        print("=" * 70)


@app.local_entrypoint()
def main(audio_path: str = AUDIO_PATH) -> None:
    StreamingV14Diagnostic().run_diagnostic.remote(audio_path=audio_path)