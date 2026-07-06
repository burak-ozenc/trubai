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
      ├── TensorConditioner  [ConditionerV12, retrain_v13/epoch_200.pt]
      │     Translates MERT embedding into Moshi's conditioning space
      │     Injects via condition_sum — acoustic domain only
      │
      ├── Moshi  [kyutai/moshika-pytorch-bf16]
      │     Full-duplex conversational core
      │     Inner monologue text channel receives all language-space steering
      │
      ├── PhraseConditioner  [src/phrase_conditioner.py]
      │     Measures pitch accuracy + tone quality at phrase boundary
      │     Assigns LOW/MED/HIGH bucket pair (9 possible states)
      │     Forces opening sentence fragment into inner monologue
      │
      ├── RAG Layer  [src/rag_passages.py]
      │     9 pre-authored pedagogical passages, one per bucket pair
      │     Forced as 12 tokens after the phrase opener
      │     Completes the sentence scaffold before LoRA takes over
      │
      ├── LogitBiasVector  [checkpoints/logit_bias/bias_v3.pt]
      │     Static bias applied to inner monologue logits
      │     Active for 24 steps post-phrase-boundary
      │     Suppresses LaTeX/academic register, attracts trumpet vocabulary
      │
      └── LoRA  [checkpoints/lora_v4/best]
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
trubai/
├── src/
│   ├── phrase_conditioner.py   # Phrase analysis, forced token injection
│   ├── rag_passages.py         # 9 pedagogical passages, one per bucket pair
│   ├── logit_bias.py           # LogitBiasVector — static logit modifier
│   └── session.py              # Session utilities
│
├── training/
│   ├── retrain_conditioner.py  # TensorConditioner training (ConditionerV12)
│   ├── lora_v4.py              # Current LoRA training script
│   └── pair_formation.py       # Dataset pair formation from raw audio
│
├── preprocessing/
│   ├── track_a.py              # Breathiness synthesis pipeline
│   ├── track_b.py              # Pitch variant augmentation pipeline
│   └── lora_v4_split.py        # Stratified train/eval split
│
├── diagnostics/
│   ├── streaming_v14.py        # Main streaming script — run this
│   ├── streaming_v14_diagnostic.py  # Diagnostic version with full token logging
│   └── verify_rag_tokens.py    # Tokenizer verification for RAG passages
│
├── data/
│   ├── failure_tokens.json     # 44 suppressed tokens, 767 occurrences
│   └── pitch_pairs_log.json    # Track B source file register classification
│
├── archive/                    # Superseded scripts — kept for reference
│   └── ...
│
├── .gitignore
└── README.md
```

---

## Model Weights

Model weights are not stored in this repository. They are hosted on HuggingFace:

| Component | File | HuggingFace |
|---|---|---|
| TensorConditioner | `retrain_v13/epoch_200.pt` | [huggingface.co/burak-ozenc/trubai/epoch_044/tensor_conditioner.pt](https://huggingface.co/burak-ozenc/trubai/blob/main/epoch_044/tensor_conditioner.pt) |
| LoRA | `lora_v4/best` | [huggingface.co/burak-ozenc/trubai/epoch_044/lora_adapter](https://huggingface.co/burak-ozenc/trubai/tree/main/epoch_044/lora_adapter) |
| LogitBiasVector | `logit_bias/bias_v3.pt` | [huggingface.co/burak-ozenc/trubai/logit_bias/bias_v3.pt](https://huggingface.co/burak-ozenc/trubai/blob/main/logit_bias/bias_v3.pt) |

Download and place under `checkpoints/` before running:

```
checkpoints/
├── retrain_v13/
│   └── epoch_200.pt
├── lora_v4/
│   └── best/
└── logit_bias/
    └── bias_v3.pt
```

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

**Download model weights from HuggingFace:**

```bash
# Place downloaded weights under checkpoints/ as shown above
```

---

## Running

**Main streaming script:**

```bash
python diagnostics/streaming_v14.py
```

**With full token diagnostic output:**

```bash
python diagnostics/streaming_v14_diagnostic.py
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

- [Moshi paper — Kyutai](https://kyutai.org)
- [MERT — Music Understanding Model](https://huggingface.co/m-a-p/MERT-v1-95M)
- [trublib on PyPI](https://pypi.org/project/trublib)
- [Kyutai alignment paper — arXiv:2506.21463](https://arxiv.org/abs/2506.21463)
- [Arban Complete Method — IMSLP](https://imslp.org)