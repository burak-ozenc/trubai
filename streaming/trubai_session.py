# trubai_session.py
import modal
import numpy as np
from pathlib import Path
from dataclasses import dataclass

app = modal.App("trubai-streaming")

# ── Image ─────────────────────────────────────────────────────────────
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

# ── Volume for checkpoints + test audio ───────────────────────────────
volume = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache", create_if_missing=True)

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


NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
MOSHI_VOCAB_SIZE = 32000



@dataclass
class PhraseObservation:
    scale: str
    note: str | None
    register: str
    pitch_accuracy: str
    tone_quality: str
    _cents_raw: float = 0.0
    _hnr_db_median: float = 0.0

    def to_llm_dict(self) -> dict:
        return {
            "scale": self.scale,
            "note": self.note,
            "register": self.register,
            "pitch_accuracy": self.pitch_accuracy,
            "tone_quality": self.tone_quality,
        }


@app.cls(
    gpu="H100",
    image=image,
    volumes={
        "/checkpoints": volume,
        "/hf-cache": hf_cache,  # ← persistent HF cache
    },
    timeout=600,
)
class TrubAISession:

    @modal.enter()
    def load_models(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"  # ← must be first line
        import torch
        import torchaudio
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders, LMGen
        from moshi.conditioners.tensors import TensorConditioner
        from moshi.conditioners import (
            TensorCondition, ConditionFuser,
            ConditionProvider, ConditionAttributes
        )
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel, AutoProcessor

        self.device = "cuda"
        self.torch = torch

        print("Loading Moshi...")
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)

        self.mimi = loaders.get_mimi(mimi_weight, device=self.device)
        self.mimi.set_num_codebooks(8)

        moshi_model = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )
        print(f"Moshi loaded ✓  VRAM: {self._vram():.2f} GB free")

        # H100 — no context patch needed (80GB VRAM)

        # Wire fuser
        self.tensor_conditioner = TensorConditioner(
            dim=MERT_DIM, output_dim=MOSHI_DIM,
            device=self.device, force_linear=True,
            output_bias=False, learn_padding=True,
        ).to(self.device)

        self.condition_provider = ConditionProvider(
            conditioners={"mert": self.tensor_conditioner},
            device=self.device,
        ).to(torch.bfloat16).to(self.device)

        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)

        moshi_model.condition_provider = self.condition_provider
        moshi_model.fuser = self.fuser

        # Load checkpoint
        BEST_CKPT = Path("/checkpoints/epoch_047")
        state = torch.load(
            BEST_CKPT / "tensor_conditioner.pt",
            map_location=self.device,
            weights_only=False,  # ← explicit, our own checkpoint
        )
        self.tensor_conditioner.load_state_dict(state)
        self.tensor_conditioner.to(torch.bfloat16)
        print("TensorConditioner loaded ✓")

        # Apply LoRA then merge
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        moshi_model = get_peft_model(moshi_model, lora_config)
        base = moshi_model.get_base_model()
        base.condition_provider = self.condition_provider
        base.fuser = self.fuser
        self.condition_provider.to(torch.bfloat16)

        moshi_model.load_adapter(
            str(BEST_CKPT / "lora_adapter"), adapter_name="default"
        )
        print("LoRA loaded ✓")

        self.merged_model = moshi_model.merge_and_unload()
        self.merged_model.condition_provider = self.condition_provider
        self.merged_model.fuser = self.fuser
        self.merged_model.eval()
        print(f"LoRA merged ✓  VRAM: {self._vram():.2f} GB free")

        # Load MERT once and keep it resident — H100 has headroom
        print("Loading MERT (persistent)...")
        self.mert_processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        ).to(self.device).eval()
        print(f"MERT loaded ✓  VRAM: {self._vram():.2f} GB free")

        print("Warming up MERT...")
        dummy = np.zeros(MOSHI_SR, dtype=np.float32)  # 1 second of silence
        self._get_phrase_embedding(dummy)
        print("MERT warm ✓")

        # Text tokenizer
        text_tok_path = hf_hub_download(
            loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME
        )
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(text_tok_path)

        actual_vocab = self.text_tokenizer.get_piece_size()
        print(f"Moshi vocab size: {actual_vocab}")

        print(f"Moshi emb is: {type(self.merged_model.emb)}")
        print(f"Number of embeddings: {len(self.merged_model.emb)}")
        for i, e in enumerate(self.merged_model.emb):
            print(f"  emb[{i}] shape: {e.weight.shape}")
        self.moshi_emb_size = self.merged_model.emb[0].weight.shape[0]
        print(f"Text embedding (codebook 0) size: {self.moshi_emb_size}")

        # In load_models, after text_tokenizer is loaded:
        self.the_token_id = self.text_tokenizer.piece_to_id("▁The")
        print(f"▁The token id: {self.the_token_id}")
        # Verify it round-trips correctly
        assert self.text_tokenizer.id_to_piece(self.the_token_id) == "▁The", "token mismatch"

       

        # trublib
        from trublib import FeatureExtractor, FrameManager
        self.FeatureExtractor = FeatureExtractor
        self.FrameManager = FrameManager
        self.fe = FeatureExtractor(sr=MOSHI_SR)

        # Use only short common words — verify each id is in-bounds before keeping it
        candidate_prefixes = {
            "breathy":           "The air",
            "slightly_breathy":  "The sound",
            "open":              "The sound",
            "cracked":           "The note",
        }
        
        self.prefix_ids = {}
        for tone, text in candidate_prefixes.items():
            ids = self.text_tokenizer.encode(text)
            valid = [i for i in ids if 3 < i < self.moshi_emb_size]
            decoded = [self.text_tokenizer.id_to_piece(i) for i in valid]
            self.prefix_ids[tone] = valid
            print(f"  prefix [{tone}]: {decoded} → ids {valid}")

        # Also import observe_phrase from trublib
        from trublib import  FrameManager, FeatureExtractor
        # self.observe_phrase = observe_phrase

        print("All models loaded ✓")

    def _vram(self):
        t = self.torch
        return (
                t.cuda.get_device_properties(0).total_memory
                - t.cuda.memory_allocated()
        ) / 1e9

    def f0_to_note_and_cents(self, f0_hz: float) -> tuple[str | None, float, int]:
        if f0_hz <= 0:
            return None, 0.0, -1
        midi_float = 12 * np.log2(f0_hz / 440.0) + 69
        midi_nearest = int(round(midi_float))
        cents = (midi_float - midi_nearest) * 100.0
        note_name = NOTE_NAMES[midi_nearest % 12]
        octave = midi_nearest // 12 - 1
        return f"{note_name}{octave}", cents, midi_nearest


    def midi_to_register(self, midi: int) -> str:
        if midi < 52:
            return "low"
        elif midi <= 72:
            return "middle"
        else:
            return "upper"

    def format_pitch_accuracy(self, label: str, cents: float) -> str:
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

    def observe_phrase(
            self,
            feature_vectors: list,
            scale_name: str = "unknown scale",
            min_salience: float = 0.35,
            force_cracked: bool = False,
    ) -> PhraseObservation | None:
        all_voiced = [fv for fv in feature_vectors if fv.f0_hz > 0]
        pitched = [fv for fv in all_voiced if fv.pitch_salience >= min_salience]

        if len(pitched) < 5:
            return None

        cents_list, note_list, midi_list = [], [], []
        for fv in pitched:
            note, cents, midi = self.f0_to_note_and_cents(fv.f0_hz)
            if note:
                cents_list.append(cents)
                note_list.append(note)
                midi_list.append(midi)

        if force_cracked:
            hnr_values = [fv.hnr_db for fv in pitched]
            median_hnr = float(np.median(hnr_values))
            if median_hnr >= 14:
                tone_quality = "open"
            elif median_hnr >= 8:
                tone_quality = "slightly_breathy"
            else:
                tone_quality = "breathy"
            f0_values = [fv.f0_hz for fv in pitched]
            median_f0 = float(np.median(f0_values))
            note, _, midi = self.f0_to_note_and_cents(median_f0)
            return PhraseObservation(
                scale=scale_name, note=note,
                register=self.midi_to_register(midi),
                pitch_accuracy="cracked", tone_quality=tone_quality,
                _cents_raw=0.0, _hnr_db_median=round(median_hnr, 1),
            )

        median_cents = float(np.median(cents_list)) if cents_list else 0.0
        most_common_note = max(set(note_list), key=note_list.count) if note_list else None
        median_midi = int(np.median(midi_list)) if midi_list else 60

        if median_cents < -20:
            pitch_label = "flat"
        elif median_cents > 20:
            pitch_label = "sharp"
        else:
            pitch_label = "in_tune"

        pitch_accuracy_str = self.format_pitch_accuracy(pitch_label, median_cents)

        hnr_values = [fv.hnr_db for fv in pitched]
        centroid_values = [fv.spectral_centroid for fv in pitched]
        f0_values = [fv.f0_hz for fv in pitched]
        median_hnr = float(np.median(hnr_values))
        median_centroid = float(np.median(centroid_values))
        median_f0 = float(np.median(f0_values))
        brightness_ratio = median_centroid / (median_f0 + 1e-6)

        if median_hnr >= 14:
            tone_quality = "open"
        elif median_hnr >= 8:
            tone_quality = "slightly_breathy"
        else:
            tone_quality = "breathy"

        if brightness_ratio > 8.0:
            tone_quality = "pinched"

        return PhraseObservation(
            scale=scale_name, note=most_common_note,
            register=self.midi_to_register(median_midi),
            pitch_accuracy=pitch_accuracy_str, tone_quality=tone_quality,
            _cents_raw=round(median_cents, 1), _hnr_db_median=round(median_hnr, 1),
        )

    def _get_phrase_embedding(self, phrase_audio: np.ndarray):
        """MERT is persistent — no load/unload. Pure inference."""
        import torch
        audio_tensor = torch.from_numpy(phrase_audio).float()
        inputs = self.mert_processor(
            audio_tensor.numpy(),
            sampling_rate=MOSHI_SR,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mert_model(**inputs, output_hidden_states=True)
        embedding = outputs.last_hidden_state.max(dim=1, keepdim=True).values
        return embedding.float().cpu()

    def _update_condition_sum(self, phrase_audio: np.ndarray):
        import torch, time
        t0 = time.perf_counter()
        embedding = self._get_phrase_embedding(phrase_audio)
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        from moshi.conditioners import TensorCondition, ConditionAttributes
        tensor_cond = TensorCondition(
            tensor=embedding.to(device=self.device, dtype=torch.bfloat16),
            mask=mask
        )
        attrs = ConditionAttributes(text={}, tensor={"mert": tensor_cond})
        prepared = self.condition_provider.prepare([attrs])
        cond_t = self.condition_provider(prepared)
        cond_sum = self.merged_model.fuser.get_sum(cond_t)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return cond_sum, latency_ms

    def _make_null_condition_tensors(self):
        import torch
        from moshi.conditioners import TensorCondition, ConditionAttributes
        null_emb = torch.zeros(1, 1, MERT_DIM, device=self.device, dtype=torch.bfloat16)
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        tensor_cond = TensorCondition(tensor=null_emb, mask=mask)
        attrs = ConditionAttributes(text={}, tensor={"mert": tensor_cond})
        prepared = self.condition_provider.prepare([attrs])
        return self.condition_provider(prepared)

    @modal.method()
    def test1_greedy(self, audio_bytes: bytes) -> dict:
        return self._run_session_core(audio_bytes, temp=0.0, temp_text=0.0,
                                      kv_reset=False, warm_start=False)

    @modal.method()
    def test2_warm_start(self, audio_bytes: bytes) -> dict:
        return self._run_session_core(audio_bytes, temp=0.8, temp_text=0.7,
                                      kv_reset=False, warm_start=True)

    @modal.method()
    def test3_kv_reset(self, audio_bytes: bytes) -> dict:
        return self._run_session_core(audio_bytes, temp=0.8, temp_text=0.7,
                                      kv_reset=True, warm_start=False)

    @modal.method()
    def test4_prefix_inject(self, audio_bytes: bytes) -> dict:
        return self._run_session_core(
            audio_bytes, temp=0.8, temp_text=0.7,
            kv_reset=False, warm_start=False, prefix_inject=True
        )

    @modal.method()
    def test5_extended_prefix(self, audio_bytes: bytes) -> dict:
        return self._run_session_core(
            audio_bytes, temp=0.8, temp_text=0.7,
            kv_reset=False, warm_start=False,
            prefix_inject=False, extended_prefix=True
        )

    @modal.method()
    def run_all_tests(self, audio_bytes: bytes) -> dict:
        results = {}
        results["test1_greedy"] = self._run_session_core(
            audio_bytes, temp=0.0, temp_text=0.0, kv_reset=False, warm_start=False)
        results["test2_warm_start"] = self._run_session_core(
            audio_bytes, temp=0.8, temp_text=0.7, kv_reset=False, warm_start=True)
        results["test3_kv_reset"] = self._run_session_core(
            audio_bytes, temp=0.8, temp_text=0.7, kv_reset=True, warm_start=False)
        return results


    def make_logit_bias_hook(self, bias_state: LogitBiasState):
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
    
    def _run_session_core(
            self,  # Modal class instance (has all model attrs)
            audio_bytes: bytes,
            retrieval_index: "RetrievalIndex",
            temp: float = 0.8,
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
        print(f"Audio: {len(audio) / MOSHI_SR:.1f}s | temp={temp} | mode=retrieval_bias")
    
        # ── Bias state + hook ────────────────────────────────────────────────────
        bias_state = LogitBiasState()
        logit_hook = self.make_logit_bias_hook(bias_state)
    
        # ── Session state ────────────────────────────────────────────────────────
        logs, text_buffer, phrase_tokens = [], [], {}
        current_phrase = -1
        phrase_idx = 0
    
        acc_buffer, acc_silent, acc_active, acc_start_t = [], 0, False, 0.0
        silence_frames = int(SILENCE_SECS * MOSHI_SR / FRAME)
        NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    
        def get_rms(f):
            return float(np.sqrt(np.mean(f ** 2)))
    
        def is_trumpet(fvs):
            return any(fv.f0_hz > 0 and 1400.0 <= fv.spectral_centroid <= 5500.0
                       for fv in fvs)
    
        # ── LMGen with null init and hook ────────────────────────────────────────
        init_ct = self._make_null_condition_tensors()
        lm_gen = LMGen(
            lm_model=self.merged_model,
            condition_tensors=init_ct,
            temp=temp,
            temp_text=temp_text,
            on_text_logits_hook=logit_hook,  # ← Option 2a injection point
        )
    
        with torch.no_grad(), lm_gen.streaming(1), self.mimi.streaming(1):
    
            def process_frame(frame):
                nonlocal acc_buffer, acc_silent, acc_active, acc_start_t
                nonlocal phrase_idx, current_phrase
    
                rms = get_rms(frame)
                self.fe.reset()
                fvs, local_fm = [], self.FrameManager()
                for i in range(0, len(frame) - 512, 512):
                    for f in local_fm.push(frame[i:i + 512]):
                        fvs.append(self.fe.extract(f))
                is_trp = is_trumpet(fvs) if fvs else False
    
                phrase_fired = None
                if not acc_active:
                    if rms > RMS_THRESH and is_trp:
                        acc_active = True
                        acc_silent = 0
                        acc_start_t = time.perf_counter()
                        acc_buffer = [frame.copy()]
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
                    print(f"\n[Phrase {phrase_idx + 1}] {phrase_duration:.2f}s")
    
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
                        print(f"    NN{rank + 1}: sim={sim:.3f}  {label.get('tone_quality', '?')} "
                              f"{label.get('note', '?')} {label.get('register', '?')}")
                    print(f"  Top-10 bias: {retrieval['top10_tokens'][:5]}")  # first 5 for readability
    
                    # ── Update bias state → hook will apply on next step() calls ─
                    retrieval_log = {
                        "phrase_idx": phrase_idx,
                        "mean_similarity": round(mean_sim, 4),
                        "similarities": [round(s, 4) for s in retrieval["similarities"]],
                        "neighbor_indices": retrieval["neighbor_indices"],
                        "neighbor_labels": retrieval["neighbor_labels"],
                        "top10_bias_tokens": retrieval["top10_tokens"],
                        "reliability": reliability,
                    }
                    bias_state.update_for_phrase(retrieval["bias_vector"], retrieval_log)
    
                    # ── Pitch / note analysis (same as baseline) ─────────────────
                    fvs_p, local_fm2 = [], self.FrameManager()
                    self.fe.reset()
                    for i in range(0, len(phrase_audio) - 512, 512):
                        for f in local_fm2.push(phrase_audio[i:i + 512]):
                            fvs_p.append(self.fe.extract(f))
                    pitched = [fv for fv in fvs_p
                               if fv.f0_hz > 0 and fv.pitch_salience >= 0.35]
                    if pitched:
                        midis = [int(round(12 * np.log2(fv.f0_hz / 440) + 69))
                                 for fv in pitched]
                        median_midi = int(np.median(midis))
                        top_note = f"{NOTE_NAMES[median_midi % 12]}{median_midi // 12 - 1}"
                        register = ("low" if median_midi < 52 else
                                    "upper" if median_midi > 72 else "middle")
                    else:
                        top_note, register = None, "unknown"
    
                    total_latency = (time.perf_counter() - boundary_time) * 1000.0
    
                    logs.append({
                        "phrase_idx": phrase_idx,
                        "mert_latency_ms": round(mert_latency, 1),
                        "total_latency_ms": round(total_latency, 1),
                        "phrase_duration_s": round(phrase_duration, 2),
                        "top_note": top_note,
                        "register": register,
                        "retrieval_sim": round(mean_sim, 4),
                        "retrieval_reliability": reliability,
                    })
                    print(f"  MERT: {mert_latency:.0f}ms | Total: {total_latency:.0f}ms")
                    print(f"  Note: {top_note} / {register}")
                    current_phrase = phrase_idx
                    phrase_idx += 1
    
                # ── MIMI encode + LMGen step ─────────────────────────────────────
                chunk = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
                codes = self.mimi.encode(chunk.unsqueeze(0))
                if codes.shape[-1] > 0:
                    for t in range(codes.shape[-1]):
                        result = lm_gen.step(codes[:, :, t:t + 1])
                        # Hook fires inside step() — bias applied before sampling
                        if result is not None:
                            tok_id = int(result[0, 0].item())
                            if tok_id > 3:
                                piece = self.text_tokenizer.id_to_piece(tok_id)
                                text_buffer.append(piece)
                                bucket = current_phrase if current_phrase >= 0 else "pre"
                                phrase_tokens.setdefault(bucket, []).append(piece)
    
            for start in range(0, len(audio) - FRAME, FRAME):
                process_frame(audio[start: start + FRAME])
    
        return {
            "logs": logs,
            "text_tokens": text_buffer,
            "phrase_tokens": {str(k): v for k, v in phrase_tokens.items()},
            "retrieval_logs": bias_state.phrase_retrieval_logs,
        }


    @modal.method()
    def run_file_session(self, audio_bytes: bytes, max_phrases: int = 20) -> dict:
        import torch, torchaudio, time, io, dataclasses
        from moshi.models import LMGen

        # Decode audio bytes
        buf = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buf)
        if sr != MOSHI_SR:
            waveform = torchaudio.functional.resample(waveform, sr, MOSHI_SR)
        audio = waveform.mean(0).numpy().astype(np.float32)
        print(f"Audio: {len(audio) / MOSHI_SR:.1f}s")

        # Session state
        logs = []
        text_buffer = []
        phrase_tokens = {}
        current_phrase = -1
        phrase_idx = 0

        # Phrase accumulator state
        acc_buffer = []
        acc_silent = 0
        acc_active = False
        acc_start_t = 0.0
        silence_frames = int(SILENCE_SECS * MOSHI_SR / FRAME)

        def get_rms(frame):
            return float(np.sqrt(np.mean(frame ** 2)))

        def is_trumpet(fvs):
            return any(
                fv.f0_hz > 0 and 1400.0 <= fv.spectral_centroid <= 5500.0
                for fv in fvs
            )

        NOTE_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

        # Build LMGen with null conditioning
        null_ct = self._make_null_condition_tensors()
        lm_gen = LMGen(
            lm_model=self.merged_model,
            condition_tensors=null_ct,
            temp=0.8,
            temp_text=0.7,
        )

        with torch.no_grad(), lm_gen.streaming(1), self.mimi.streaming(1):

            for start in range(0, len(audio) - FRAME, FRAME):
                frame = audio[start: start + FRAME]
                rms = get_rms(frame)

                # Feature extraction
                self.fe.reset()
                fvs = []
                local_fm = self.FrameManager()
                for i in range(0, len(frame) - 512, 512):
                    for f in local_fm.push(frame[i:i + 512]):
                        fvs.append(self.fe.extract(f))
                is_trp = is_trumpet(fvs) if fvs else False

                # Phrase accumulator
                phrase_fired = None
                if not acc_active:
                    if rms > RMS_THRESH and is_trp:
                        acc_active = True
                        acc_silent = 0
                        acc_start_t = time.perf_counter()
                        acc_buffer = [frame.copy()]
                else:
                    acc_buffer.append(frame.copy())
                    if rms < RMS_THRESH:
                        acc_silent += 1
                        if acc_silent >= silence_frames:
                            phrase_audio = np.concatenate(acc_buffer)
                            phrase_duration = time.perf_counter() - acc_start_t
                            phrase_fired = (phrase_audio, phrase_duration)
                            acc_buffer, acc_silent, acc_active = [], 0, False
                    else:
                        acc_silent = 0

                if phrase_fired is not None and phrase_idx < max_phrases:
                    phrase_audio, phrase_duration = phrase_fired
                    boundary_time = time.perf_counter()
                    print(f"\n[Phrase {phrase_idx + 1}] {phrase_duration:.2f}s")

                    new_cond_sum, mert_latency = self._update_condition_sum(phrase_audio)
                    lm_gen._streaming_state.condition_sum = new_cond_sum
                    current_phrase = phrase_idx
                    total_latency = (time.perf_counter() - boundary_time) * 1000.0

                    # Note/register
                    fvs_p = []
                    local_fm2 = self.FrameManager()
                    self.fe.reset()
                    for i in range(0, len(phrase_audio) - 512, 512):
                        for f in local_fm2.push(phrase_audio[i:i + 512]):
                            fvs_p.append(self.fe.extract(f))
                    pitched = [fv for fv in fvs_p
                               if fv.f0_hz > 0 and fv.pitch_salience >= 0.35]
                    if pitched:
                        midis = [int(round(12 * np.log2(fv.f0_hz / 440) + 69))
                                 for fv in pitched]
                        median_midi = int(np.median(midis))
                        top_note = f"{NOTE_NAMES[median_midi % 12]}{median_midi // 12 - 1}"
                        register = ("low" if median_midi < 52 else
                                    "upper" if median_midi > 72 else "middle")
                    else:
                        top_note, register = None, "unknown"

                    logs.append({
                        "phrase_idx": phrase_idx,
                        "mert_latency_ms": round(mert_latency, 1),
                        "total_latency_ms": round(total_latency, 1),
                        "phrase_duration_s": round(phrase_duration, 2),
                        "top_note": top_note,
                        "register": register,
                    })
                    print(f"  MERT:  {mert_latency:.0f} ms")
                    print(f"  Total: {total_latency:.0f} ms")
                    print(f"  Note:  {top_note} / {register}")
                    phrase_idx += 1

                # LMGen step
                chunk = torch.from_numpy(frame).float().unsqueeze(0).to(self.device)
                codes = self.mimi.encode(chunk.unsqueeze(0))
                if codes.shape[-1] > 0:
                    for t in range(codes.shape[-1]):
                        result = lm_gen.step(codes[:, :, t:t + 1])
                        if result is not None:
                            tok_id = int(result[0, 0].item())
                            if tok_id > 3:
                                piece = self.text_tokenizer.id_to_piece(tok_id)
                                text_buffer.append(piece)
                                # Also bucket into current phrase window
                                bucket = current_phrase if current_phrase >= 0 else "pre"
                                phrase_tokens.setdefault(bucket, []).append(piece)

        return {"logs": logs, "text_tokens": text_buffer,
                "phrase_tokens": {str(k): v for k, v in phrase_tokens.items()}}



@app.local_entrypoint()
def main():
    import json
    TEST_FILE = "AuSep_2_tpt_43_Chorale.wav"
    with open(TEST_FILE, "rb") as f:
        audio_bytes = f.read()

    session = TrubAISession()
    result  = session.test5_extended_prefix.remote(audio_bytes)

    trumpet_vocab = {"air", "tone", "center", "breath", "aperture",
                     "support", "slot", "corners", "embouchure", "column",
                     "lacks", "carries", "cracked", "attack", "sits"}

    print("\n══ test5_extended_prefix ══")
    for k, v in result.get("phrase_tokens", {}).items():
        label = "pre-update (null)" if k == "pre" else f"post phrase {int(k)+1}"
        print(f"  [{label}]: {' '.join(v[:35])}")

    hits = [w for w in trumpet_vocab
            if w in " ".join(result["text_tokens"]).lower()]
    print(f"\nTrumpet vocab hits: {hits if hits else 'none'}")

    with open("report_test5.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to report_test5.json")

# @app.local_entrypoint()
# def main():
#     import json
#     TEST_FILE = "AuSep_2_tpt_43_Chorale.wav"
#     with open(TEST_FILE, "rb") as f:
#         audio_bytes = f.read()
# 
#     session = TrubAISession()
#     results = session.run_all_tests.remote(audio_bytes)
# 
#     trumpet_vocab = {"air", "tone", "center", "breath", "aperture",
#                      "support", "slot", "corners", "embouchure", "column"}
# 
#     for test_name, result in results.items():
#         print(f"\n{'═' * 50}")
#         print(test_name)
#         print(f"{'═' * 50}")
#         print("Tokens by phrase window:")
#         for k, v in result.get("phrase_tokens", {}).items():
#             label = "pre-update (null)" if k == "pre" else f"post phrase {int(k) + 1}"
#             print(f"  [{label}]: {' '.join(v[:30])}")
#         hits = [w for w in trumpet_vocab
#                 if w in " ".join(result["text_tokens"]).lower()]
#         print(f"Trumpet vocab hits: {hits if hits else 'none'}")
# 
#     results["test4_prefix"] = session.test4_prefix_inject.remote(audio_bytes)
# 
#     # Print it the same way
#     print("\n══ test4_prefix_inject ══")
#     for k, v in results["test4_prefix"].get("phrase_tokens", {}).items():
#         label = "pre-update (null)" if k == "pre" else f"post phrase {int(k) + 1}"
#         print(f"  [{label}]: {' '.join(v[:30])}")
#     hits = [w for w in trumpet_vocab
#             if w in " ".join(results["test4_prefix"]["text_tokens"]).lower()]
#     print(f"Trumpet vocab hits: {hits if hits else 'none'}")
# 
#     with open("all_tests_report.json", "w") as f:
#         json.dump(results, f, indent=2)
#     print("\nSaved to all_tests_report.json")

# @app.local_entrypoint()
# def main():
#     import json
# 
#     TEST_FILE = "AuSep_2_tpt_43_Chorale.wav"   # ← local path
# 
#     with open(TEST_FILE, "rb") as f:
#         audio_bytes = f.read()
# 
#     session  = TrubAISession()
#     result   = session.run_file_session.remote(audio_bytes, max_phrases=20)
#     logs     = result["logs"]
#     tokens   = result["text_tokens"]
# 
#     print(f"\n{'═'*50}")
#     print("SESSION REPORT")
#     print(f"{'═'*50}")
#     print(f"Phrases detected: {len(logs)}")
# 
#     if logs:
#         print("\nTokens by phrase window:")
#         for k, v in result.get("phrase_tokens", {}).items():
#             label = "pre-update (null)" if k == "pre" else f"post phrase {int(k)+1}"
#             print(f"  [{label}]: {' '.join(v[:30])}")
#             
#         mert_lat  = [l["mert_latency_ms"]  for l in logs]
#         total_lat = [l["total_latency_ms"] for l in logs]
#         registers = [l["register"]         for l in logs]
# 
#         print(f"\nLatency:")
#         print(f"  MERT  mean:    {np.mean(mert_lat):.0f} ms")
#         print(f"  MERT  min/max: {np.min(mert_lat):.0f} / {np.max(mert_lat):.0f} ms")
#         print(f"  Total mean:    {np.mean(total_lat):.0f} ms")
#         print(f"  Total min/max: {np.min(total_lat):.0f} / {np.max(total_lat):.0f} ms")
# 
#         print(f"\nPer-phrase:")
#         print(f"  {'#':<4} {'note':<6} {'register':<8} {'dur':>6}  {'MERT':>6}  {'total':>7}")
#         print(f"  {'─'*46}")
#         for l in logs:
#             print(f"  {l['phrase_idx']+1:<4} {str(l['top_note']):<6} "
#                   f"{l['register']:<8} {l['phrase_duration_s']:>5.2f}s "
#                   f"{l['mert_latency_ms']:>6.0f}  {l['total_latency_ms']:>6.0f}")
# 
#         reg_counts = {r: registers.count(r) for r in set(registers)}
#         print(f"\nRegisters: {reg_counts}")
# 
#     print(f"\nInner monologue (sample):")
#     print(" ".join(tokens[:80]) if tokens else "  (none)")
#     print(f"\n{'═'*50}")
# 
#     with open("session_report.json", "w") as f:
#         json.dump(result, f, indent=2)
#     print("Full report saved to session_report.json")






