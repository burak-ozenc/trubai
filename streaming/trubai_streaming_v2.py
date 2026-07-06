# TRUB.AI — Streaming Session v2
# Pure architecture: TensorConditioner (retrain_v2) + LoRA (lora_v2).
# Retrieval hook removed — confirmed non-load-bearing (Task 3).
# SILENCE_SECS = 1.5 — prevents phrase boundary on breath gaps (Task 2).
# ─────────────────────────────────────────────────────────────────────────────

import modal
import numpy as np
import json
import typing as tp
from dataclasses import dataclass
from pathlib import Path

app = modal.App("trubai-streaming")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg"])
    .pip_install([
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "moshi",
        "transformers",
        "sentencepiece",
        "peft",
        "huggingface_hub",
        "librosa", "soundfile",
        "nnAudio",
        "trublib",
    ])
)

volume   = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MOSHI_SR     = 24000
MERT_DIM     = 768
FRAME        = 1920
RMS_THRESH   = 0.01
SILENCE_SECS = 1.5    # Task 2: was 0.6 — prevents phrase boundary on breath gaps
# silence_frames = int(1.5 * 24000 / 1920) = 18 frames = 1.44s

NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

CONDITIONER_CKPT = "/checkpoints/retrain_v2/best/tensor_conditioner.pt"
LORA_CKPT        = "/checkpoints/lora_v2/best"


# ─────────────────────────────────────────────────────────────────────────────
# Phrase observation helpers (pure numpy, not in trublib)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhraseObservation:
    scale:          str
    note:           tp.Optional[str]
    register:       str
    pitch_accuracy: str
    tone_quality:   str
    _cents_raw:     float = 0.0
    _hnr_db_median: float = 0.0

    def to_llm_dict(self) -> dict:
        return {
            "scale":          self.scale,
            "note":           self.note,
            "register":       self.register,
            "pitch_accuracy": self.pitch_accuracy,
            "tone_quality":   self.tone_quality,
        }


def f0_to_note_and_cents(f0_hz: float) -> tp.Tuple[tp.Optional[str], float, int]:
    if f0_hz <= 0:
        return None, 0.0, -1
    midi_float   = 12 * np.log2(f0_hz / 440.0) + 69
    midi_nearest = int(round(midi_float))
    cents        = (midi_float - midi_nearest) * 100.0
    return f"{NOTE_NAMES[midi_nearest % 12]}{midi_nearest // 12 - 1}", cents, midi_nearest


def midi_to_register(midi: int) -> str:
    return "low" if midi < 52 else "upper" if midi > 72 else "middle"


def observe_phrase(
        feature_vectors: list,
        scale_name:    str   = "unknown scale",
        min_salience:  float = 0.35,
) -> tp.Optional[PhraseObservation]:
    pitched = [fv for fv in feature_vectors
               if fv.f0_hz > 0 and fv.pitch_salience >= min_salience]
    if len(pitched) < 5:
        return None

    hnr_values      = [fv.hnr_db for fv in pitched]
    centroid_values = [fv.spectral_centroid for fv in pitched]
    f0_values       = [fv.f0_hz for fv in pitched]
    median_hnr      = float(np.median(hnr_values))
    median_centroid = float(np.median(centroid_values))
    median_f0       = float(np.median(f0_values))
    brightness_ratio = median_centroid / (median_f0 + 1e-6)

    cents_list, note_list, midi_list = [], [], []
    for fv in pitched:
        note, cents, midi = f0_to_note_and_cents(fv.f0_hz)
        if note:
            cents_list.append(cents)
            note_list.append(note)
            midi_list.append(midi)

    median_cents     = float(np.median(cents_list)) if cents_list else 0.0
    most_common_note = max(set(note_list), key=note_list.count) if note_list else None
    median_midi      = int(np.median(midi_list)) if midi_list else 60

    pitch_label = "flat" if median_cents < -20 else "sharp" if median_cents > 20 else "in_tune"
    pitch_str   = f"{pitch_label} ({median_cents:+.0f}¢)"

    if median_hnr >= 14:
        tone = "open"
    elif median_hnr >= 8:
        tone = "slightly_breathy"
    else:
        tone = "breathy"
    if brightness_ratio > 8.0:
        tone = "pinched"

    return PhraseObservation(
        scale=scale_name, note=most_common_note,
        register=midi_to_register(median_midi),
        pitch_accuracy=pitch_str, tone_quality=tone,
        _cents_raw=round(median_cents, 1), _hnr_db_median=round(median_hnr, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Modal class
# ─────────────────────────────────────────────────────────────────────────────

@app.cls(
    image=image,
    gpu="H100",
    volumes={
        "/checkpoints": volume,
        "/hf-cache":    hf_cache,
    },
    timeout=600,
)
class TrubAI:

    @modal.enter()
    def load_models(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"  # MUST BE FIRST

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners.tensors import TensorConditioner
        from moshi.conditioners import ConditionFuser, ConditionProvider
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoProcessor
        from trublib import FeatureExtractor
        from trublib.frame_manager import FrameManager

        self.device = "cuda"

        # 1. Base Moshi
        mimi_weight  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)
        moshi_model = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16,
        )

        # 2. Wire fuser
        self.tensor_conditioner = TensorConditioner(
            dim=768, output_dim=4096, device=self.device,
            force_linear=True, output_bias=False, learn_padding=True,
        ).to(self.device)

        self.condition_provider = ConditionProvider(
            conditioners={"mert": self.tensor_conditioner}, device=self.device,
        ).to(torch.bfloat16).to(self.device)

        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)

        moshi_model.condition_provider = self.condition_provider
        moshi_model.fuser              = self.fuser

        # 3. TensorConditioner — retrain_v2 (contrastive + norm penalty, epoch 104)
        state = torch.load(CONDITIONER_CKPT, map_location=self.device, weights_only=False)
        self.tensor_conditioner.load_state_dict(state)
        self.tensor_conditioner.to(torch.bfloat16)

        # 4. LoRA v2 — trained from scratch against corrected TensorConditioner
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        moshi_model = get_peft_model(moshi_model, lora_config)
        base = moshi_model.get_base_model()
        base.condition_provider = self.condition_provider
        base.fuser              = self.fuser
        self.condition_provider.to(torch.bfloat16)
        moshi_model.load_adapter(LORA_CKPT, adapter_name="default")
        self.merged_model = moshi_model.merge_and_unload()
        self.merged_model.condition_provider = self.condition_provider
        self.merged_model.fuser              = self.fuser
        self.merged_model.eval()

        # 5. MERT (persistent on H100)
        self.mert_processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True,
        )
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True,
        ).to(self.device).eval()

        # 6. Text tokenizer
        text_tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(text_tok_path)

        # 7. trublib
        self.FeatureExtractor = FeatureExtractor
        self.FrameManager     = FrameManager
        self.fe               = FeatureExtractor(sr=24000)

        # 8. MERT warmup
        dummy = np.zeros(24000, dtype=np.float32)
        self._get_phrase_embedding(dummy)

        print("All models loaded ✓")
        print(f"  TensorConditioner: {CONDITIONER_CKPT}")
        print(f"  LoRA:              {LORA_CKPT}")
        print(f"  SILENCE_SECS:      {SILENCE_SECS}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_phrase_embedding(self, phrase_audio: np.ndarray):
        import torch
        inputs = self.mert_processor(
            phrase_audio, sampling_rate=MOSHI_SR, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        return outputs.last_hidden_state.max(dim=1, keepdim=True).values.float().cpu()

    def _update_condition_sum(self, phrase_audio: np.ndarray):
        import torch, time
        from moshi.conditioners import TensorCondition, ConditionAttributes
        t0        = time.perf_counter()
        embedding = self._get_phrase_embedding(phrase_audio)
        mask      = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        tc        = TensorCondition(
            tensor=embedding.to(device=self.device, dtype=torch.bfloat16), mask=mask,
        )
        attrs    = ConditionAttributes(text={}, tensor={"mert": tc})
        prepared = self.condition_provider.prepare([attrs])
        cond_t   = self.condition_provider(prepared)
        cond_sum = self.merged_model.fuser.get_sum(cond_t)
        return cond_sum, (time.perf_counter() - t0) * 1000.0

    def _make_null_condition_tensors(self):
        import torch
        from moshi.conditioners import TensorCondition, ConditionAttributes
        null_emb = torch.zeros(1, 1, MERT_DIM, device=self.device, dtype=torch.bfloat16)
        mask     = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        tc       = TensorCondition(tensor=null_emb, mask=mask)
        attrs    = ConditionAttributes(text={}, tensor={"mert": tc})
        prepared = self.condition_provider.prepare([attrs])
        return self.condition_provider(prepared)

    # ── Main session ──────────────────────────────────────────────────────────

    @modal.method()
    def run_session(self, audio_bytes: bytes, temp: float = 0.8, temp_text: float = 0.7) -> dict:
        import torch, torchaudio, time, io
        from moshi.models import LMGen

        buf = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buf)
        if sr != MOSHI_SR:
            waveform = torchaudio.functional.resample(waveform, sr, MOSHI_SR)
        audio = waveform.mean(0).numpy().astype(np.float32)
        print(f"Audio: {len(audio)/MOSHI_SR:.1f}s | temp={temp} | SILENCE_SECS={SILENCE_SECS}")

        logs, text_buffer, phrase_tokens = [], [], {}
        current_phrase = -1
        phrase_idx     = 0

        acc_buffer, acc_silent, acc_active, acc_start_t = [], 0, False, 0.0
        silence_frames = int(SILENCE_SECS * MOSHI_SR / FRAME)
        print(f"silence_frames={silence_frames} ({silence_frames * FRAME / MOSHI_SR:.2f}s)")

        def get_rms(f):
            return float(np.sqrt(np.mean(f ** 2)))

        def is_trumpet(fvs):
            return any(
                fv.f0_hz > 0 and 1400.0 <= fv.spectral_centroid <= 5500.0
                for fv in fvs
            )

        init_ct = self._make_null_condition_tensors()
        lm_gen  = LMGen(
            lm_model=self.merged_model,
            condition_tensors=init_ct,
            temp=temp,
            temp_text=temp_text,
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

                    new_cond_sum, mert_latency = self._update_condition_sum(phrase_audio)
                    lm_gen._streaming_state.condition_sum = new_cond_sum
                    total_latency = (time.perf_counter() - boundary_time) * 1000.0

                    fvs_p, local_fm2 = [], self.FrameManager()
                    self.fe.reset()
                    for i in range(0, len(phrase_audio) - 512, 512):
                        for f in local_fm2.push(phrase_audio[i:i+512]):
                            fvs_p.append(self.fe.extract(f))
                    obs = observe_phrase(fvs_p)

                    if obs is not None:
                        top_note = obs.note
                        register = obs.register
                        print(f"  Obs: note={obs.note} reg={obs.register} "
                              f"pitch={obs.pitch_accuracy} tone={obs.tone_quality}")
                    else:
                        pitched = [fv for fv in fvs_p
                                   if fv.f0_hz > 0 and fv.pitch_salience >= 0.35]
                        if pitched:
                            midis       = [int(round(12 * np.log2(fv.f0_hz/440)+69))
                                           for fv in pitched]
                            median_midi = int(np.median(midis))
                            top_note    = f"{NOTE_NAMES[median_midi%12]}{median_midi//12-1}"
                            register    = midi_to_register(median_midi)
                        else:
                            top_note, register = None, "unknown"
                        print(f"  Note: {top_note} / {register} (obs=None)")

                    print(f"  MERT: {mert_latency:.0f}ms | Total: {total_latency:.0f}ms")
                    logs.append({
                        "phrase_idx":        phrase_idx,
                        "mert_latency_ms":   round(mert_latency, 1),
                        "total_latency_ms":  round(total_latency, 1),
                        "phrase_duration_s": round(phrase_duration, 2),
                        "top_note":          top_note,
                        "register":          register,
                        "observation":       obs.to_llm_dict() if obs else None,
                    })
                    current_phrase  = phrase_idx
                    phrase_idx     += 1

                chunk = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
                codes = self.mimi.encode(chunk.unsqueeze(0))
                if codes.shape[-1] > 0:
                    for t in range(codes.shape[-1]):
                        result = lm_gen.step(codes[:, :, t:t+1])
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
            "logs":          logs,
            "text_tokens":   text_buffer,
            "phrase_tokens": {str(k): v for k, v in phrase_tokens.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    test_file = "AuSep_2_tpt_43_Chorale.wav"
    with open(test_file, "rb") as f:
        audio_bytes = f.read()

    result = TrubAI().run_session.remote(audio_bytes, temp=0.8, temp_text=0.7)

    print("\n" + "="*60)
    print("SESSION RESULTS")
    print("="*60)

    print("\n── Phrase logs ──")
    for log in result["logs"]:
        print(json.dumps(log, indent=2))

    print("\n── Phrase tokens (inner monologue windows) ──")
    for bucket, tokens in result["phrase_tokens"].items():
        label = f"Phrase {int(bucket)+1}" if bucket != "pre" else "pre-update"
        print(f"  {label}: {' '.join(tokens)}")

    print("\n── Full token stream ──")
    print(" ".join(result["text_tokens"]))