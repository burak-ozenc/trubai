# TRUB.AI

A real-time AI trumpet teacher. Listens to you play, understands what is happening musically, and responds in a natural conversational voice — like a real teacher would.

The key design decision: a real teacher responds to what they *hear*, not just what you *say*. The entire system is built around audio, not transcription. Trumpet audio is encoded directly into the conversation engine — pitch accuracy, tone quality, and playing errors are understood acoustically, not inferred from speech.

---

## Architecture

```
[Full-duplex audio stream]
      │
      ├── trublib (TAD)
      │     Detects phrase boundaries: Silent → Onset → Active → Trailing
      │     Routes audio to MERT only during Active state
      │
      ├── MERT-v1-95M
      │     Encodes trumpet audio into a 768-dim embedding
      │     Runs once per phrase at the Active→Trailing boundary (~28ms)
      │
      ├── TensorConditioner  [Modal Volume: /checkpoints/retrain_v13/best/tensor_conditioner.pt]
      │     Translates MERT embedding into Moshi's conditioning space
      │     Injects via condition_sum — acoustic domain only
      │
      ├── Moshi  [kyutai/moshika-pytorch-bf16]
      │     Full-duplex conversational core
      │     Inner monologue text channel receives all language-space steering
      │
      ├── PhraseConditioner  [conditioner/phrase_conditioner.py]
      │     Measures pitch accuracy + tone quality at phrase boundary
      │     Assigns LOW/MED/HIGH bucket pair (9 possible states)
      │     Forces opening sentence fragment into inner monologue
      │
      ├── RAG Layer  [RAG_PASSAGES dict, inline in conditioner/phrase_conditioner.py]
      │     9 pre-authored pedagogical passages, one per bucket pair
      │     Forced as 12 tokens after the phrase opener
      │     Completes the sentence scaffold before LoRA takes over
      │
      ├── LogitBiasVector  [LogitBiasState, inline in streaming/trubai_streaming_v14.py;
      │                     checkpoint: Modal Volume /checkpoints/logit_bias/bias_v3.pt]
      │     Static bias applied to inner monologue logits
      │     Active for 24 steps post-phrase-boundary
      │     Suppresses LaTeX/academic register, attracts trumpet vocabulary
      │
      └── LoRA  [Modal Volume: /checkpoints/lora_v4/best]
            Trumpet pedagogical vocabulary layer on top of base Moshi
            Trained on 4,815 labeled (audio, inner_monologue) pairs
            Generates continuation after the RAG scaffold drains
```

**The inner monologue sequence per phrase:**
```
[F] ▁Your ▁tone ▁center ▁is    ← PhraseConditioner forced prefix
[B] ▁spreading                  ← bridge token
[R] ▁slightly ▁at ▁the ▁phrase  ← RAG passage (12 forced tokens)
[D] ▁end ...                    ← LoRA free generation, logit-biased (12 steps)
    ...                         ← unsteered free generation
```

---

## Project Structure

```
trubai-synthetic-data/
├── conditioner/
│   ├── phrase_conditioner.py         # PhraseConditioner + inline RAG_PASSAGES dict
│   ├── trubai_retrain_conditioner.py # TensorConditioner training
│   └── trubai_retrain_conditioner_v3/v4.py
│
├── training/
│   ├── trubai_train_v5.py ... v13.py # LoRA/conditioner training runs
│   ├── trubai_lora_v3.py / v4.py     # LoRA training (v4 current)
│   └── trubai_lora_retrain_v2.py
│
├── preprocessing/
│   ├── track_a.py                    # Breathiness synthesis pipeline
│   ├── track_b.py                    # Pitch variant augmentation pipeline
│   ├── pair_formation.py             # Dataset pair formation from raw audio
│   ├── lora_v4_split.py              # Stratified train/eval split
│   └── upload_lora_v4_data.py        # Pushes training data to Modal volume
│
├── streaming/
│   ├── trubai_streaming_v14.py       # Current production streaming app (Modal)
│   ├── trubai_streaming_v14_diagnostic.py  # Full token-tag logging version
│   ├── trubai_session.py, trubai_modal_option2a.py, trubai_diagnostic_no_lora.py
│   └── trubai_streaming_v2.py ... v13.py    # Prior versions, kept for reference
│
├── calibration/
│   ├── trubai_calibrate_alpha.py ... v3.py  # Logit bias calibration
│   ├── trubai_derive_logit_bias.py, trubai_norm_check.py
│   └── trubai_outlier_diagnostic.py, trubai_inspect_leakers.py, trubai_verify_rag_tokens.py
│
├── reports/                          # Test reports, logs, commercial-safe vocabulary list
│
├── augmented/                        # Small wav set tracked in git
├── data/                             # Synthesized audio + json pair logs (wavs gitignored)
│
├── checkpoints/, epoch_044/, new_embeddings/, embeddings.zip, bias_v3.pt
│                                      # Model artifacts — gitignored, mirrored to HF
│
├── push_to_hf.py                     # Pushes model artifacts to HuggingFace
├── SUMMARY.md, Technical_Log.md      # Plain-language overview + full build history
├── .gitignore
└── README.md
```

---

## Model Weights

Model weights are not stored in this git repository — they live in two places:

**Production runtime** reads them from a Modal Volume (`trubai-checkpoints`), mounted
at `/checkpoints` inside the streaming app:

```
/checkpoints/retrain_v13/best/tensor_conditioner.pt
/checkpoints/lora_v4/best/
/checkpoints/logit_bias/bias_v3.pt
```

**A snapshot is also mirrored to HuggingFace** (private repo) for sharing/backup —
local folder names differ slightly from the Modal volume (e.g. `epoch_044/` instead
of `retrain_v13/`) since it's a point-in-time copy, not the live volume:

| Component | HuggingFace |
|---|---|
| TensorConditioner | [huggingface.co/burak-ozenc/trubai/epoch_044/tensor_conditioner.pt](https://huggingface.co/burak-ozenc/trubai/blob/main/epoch_044/tensor_conditioner.pt) |
| LoRA | [huggingface.co/burak-ozenc/trubai/epoch_044/lora_adapter](https://huggingface.co/burak-ozenc/trubai/tree/main/epoch_044/lora_adapter) |
| LogitBiasVector | [huggingface.co/burak-ozenc/trubai/logit_bias/bias_v3.pt](https://huggingface.co/burak-ozenc/trubai/blob/main/logit_bias/bias_v3.pt) |

---

## Setup

**Requirements:**
- Python 3.10
- WSL2 (Windows) or Linux
- Modal.com account (compute — Modal H100 for MERT and training)
- HuggingFace account (model weight access)

**Install dependencies:**

```bash
pip install modal torch torchaudio sentencepiece praat-parselmouth \
            soundfile scipy numpy librosa pyrubberband
```

**System dependency (pitch shifting):**

```bash
sudo apt install rubberband-cli
```

**Modal setup:**

```bash
pip install modal
modal token new
```

**Environment variables — create a `.env` file (never commit this):**

```
HF_TOKEN=your_huggingface_token
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

**For local development** (not required for running the app — that reads from the
Modal Volume directly), you can pull the HuggingFace snapshot listed above with
`huggingface-cli download burak-ozenc/trubai --local-dir ./hf_snapshot`.

---

## Running

These are Modal apps (`@app.local_entrypoint()`), so they're invoked with the
Modal CLI, not plain `python`:

**Main streaming script:**

```bash
modal run streaming/trubai_streaming_v14.py
```

**With full token diagnostic output:**

```bash
modal run streaming/trubai_streaming_v14_diagnostic.py
```

The diagnostic script logs every inner monologue token with its tag — `[F]` forced prefix, `[B]` bridge token, `[R]` RAG passage, `[D]` logit-biased free generation, and untagged post-window free generation.

---

## Key Concepts

**Why not STT → LLM → TTS?**
Transcription throws away all musical information. A teacher who hears "C" cannot tell you whether the note was flat, breathy, or cracked. The pitch and tone information lives in the audio, not the words. The whole architecture is designed to preserve and respond to that acoustic information.

**trublib / TAD (Trumpet Activity Detector)**
A 4-state machine (Silent → Onset → Active → Trailing) that detects phrase boundaries in real time. MERT only runs when a phrase is complete — at the Active→Trailing transition. Running MERT per frame would be both wasteful and acoustically wrong (a phrase has a shape that a single frame does not capture).

**condition_sum vs. text channel**
Two separate injection points in Moshi. `condition_sum` is the acoustic domain — MERT embeddings go here, nothing else. The text channel (inner monologue) is the language domain — everything that steers what Moshi says goes here. These two channels do not mix.

**The 12-token RAG scaffold**
The LoRA was trained on 4,815 pairs but the inner monologue still produces fragments rather than complete sentences in free generation. The RAG layer forces the first 12 tokens of each phrase response as a complete observational sentence opening. The LoRA generates continuation within the logit bias window after the scaffold drains.

---

## Dataset

The LoRA training dataset contains 4,815 labeled pairs across five categories:

| Label | Count | Source |
|---|---|---|
| slightly_breathy | 1,581 | Track A — white noise synthesis on clean recordings |
| breathy | 1,581 | Track A — white noise synthesis on clean recordings |
| flat | 814 | Track B — pitch shifting (−45c, −60c) on clean recordings |
| sharp | 814 | Track B — pitch shifting (+45c, +60c) on clean recordings |
| cracked | 25 | Track C — manually recorded crack takes |

Each pair is `(audio_file, inner_monologue_sentence)`. The inner monologue is always in observational/diagnostic register — what the playing indicates, not what the student should do.

---

## Current Status

| Component | Status |
|---|---|
| trublib (TAD) | ✅ Complete, published on PyPI |
| TensorConditioner (retrain_v13) | ✅ Production |
| LoRA v4 | 🔄 Training in progress |
| LogitBiasVector (bias_v3) | ✅ Production |
| PhraseConditioner | ✅ Production |
| RAG layer (Option A, 9 passages) | ✅ Production |
| Spoken output alignment (DPO-LN) | ⏳ Phase 6/7 — after LoRA v4 |
| Session state layer | ⏳ Phase 6/7 — after alignment |

---

## Moshi Fork

The project uses a modified fork of the Kyutai Moshi model:

```
github.com/burak-ozenc/moshi
```

Changes from upstream:
- `forced_text_token` parameter added to `_step()` and `step()` — forced token injection
- `text_logit_modifier` callable parameter added — external logit modification hook
- `text_steering_delta` field added to `_LMGenState` — Phase 5b stub, always None in current version

All changes operate after `graphed_main()` returns and do not cross the CUDA graph boundary.

---

## References

- [Moshi paper — Kyutai](https://arxiv.org/abs/2410.00037)
- [MERT — Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-95M)
- [trublib on PyPI](https://pypi.org/project/trublib)
- [Kyutai alignment paper — arXiv:2506.21463](https://arxiv.org/abs/2506.21463)
- [Arban Complete Method — IMSLP](https://imslp.org)