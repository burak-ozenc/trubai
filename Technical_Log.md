# TRUB.AI — Technical Log
## Section 1: Project Overview & Design Philosophy

---

### What the project is

TRUB.AI is a real-time AI trumpet teacher. It listens to a student play, understands
what is happening musically at the acoustic level, and responds in a natural
conversational voice — the way a real teacher would.

The target interaction is not a rigid exercise system. It is a conversation. The
teacher hears a phrase, understands what was good and what was wrong, and responds
immediately. The student plays again. The teacher listens again. This loop is the
core of a real lesson, and it is what the system is designed to support.

---

### The founding decision: abandon STT → LLM → TTS

The first architectural question was the most important one, and it was answered
early and has not been revisited.

The obvious pipeline for a voice AI in 2024 is:

```
Student speaks or plays → STT (transcription) → LLM (text response) → TTS (voice)
```

This pipeline is wrong for a music teacher. The reason is simple: transcription
destroys musical information.

When a student plays a C and the transcription says "C", all of the following
information is gone:

- Was the note flat or sharp, and by how much?
- Was the tone clean or breathy?
- Did the note crack at the attack?
- Did the pitch drift across the phrase?
- Was the air support consistent through the note?

A text transcript of "C" tells a teacher nothing. A real teacher responds to what
they hear, not to a label. A system that transcribes first cannot be a music teacher
— it is a quiz machine at best.

The decision: the system must be audio-native. Musical information must be preserved
from the student's audio all the way to the response generation. Transcription is
never performed on the musical content.

---

### What audio-native means in practice

Audio-native means two things structurally:

**First:** The student's trumpet audio is encoded directly into a mathematical
representation that captures musical content — pitch accuracy, tone quality,
breathiness, crack events. This encoding is done by MERT, a music understanding
model. The encoding is injected directly into the conversation engine, not converted
to text first.

**Second:** The conversation engine must be full-duplex — capable of listening and
speaking simultaneously, without a push-to-talk model. A real teacher does not wait
for the student to finish speaking before they begin processing. They are listening
continuously. Moshi, the conversational AI at the center of the system, provides
this full-duplex capability.

---

### Why Moshi specifically

Moshi (kyutai/moshika-pytorch-bf16, MIT license) was selected over other
conversational models for three reasons:

**Full-duplex by architecture.** Most voice AI models are half-duplex — they
transcribe input, generate a response, and play it back. Moshi processes audio
input and generates audio output simultaneously, in real time, with a theoretical
latency of 160ms. This is below typical human conversational latency (200–250ms).
For a music teacher, timing matters. A response that arrives after the student has
moved on to the next phrase is useless.

**Inner monologue text channel.** Moshi has a dual-channel architecture: an inner
monologue text channel (what the model "thinks") and an audio output channel (what
the model "says"). These are coupled but distinct. The inner monologue channel is
the injection point for all of our acoustic conditioning and pedagogical steering —
we can write context into what the model thinks, and that influences what it says.
This channel is the architectural foundation of everything built in Phases 5 and
beyond.

**Open weights, MIT license.** The system needed to be modifiable at the model
level — not just prompted. We fork Moshi and add forced token injection and logit
modification hooks directly to the generation loop. A closed model makes this
impossible.

---

### The agent structure

The project was developed under a structured multi-agent workflow:

- **Muse (Scholar 1):** Architecture leadership. Argues approaches, produces
  specifications, holds final decision authority on all technical questions. Does
  not implement.

- **Spira (Scholar 2):** Research specialist. Brings ranked options with failure
  modes before any specification is written. Reports to Muse. Does not coordinate
  with Faber directly.

- **Faber (Builder):** Implementation only. Works from Muse-approved
  specifications. Does not make architectural decisions. Reports results back to
  Muse for assessment before proceeding.

- **Lituus (Builder 2):** A second builder for smaller, self-contained tasks —
  preprocessing pipelines, diagnostics, extraction jobs — that do not require the
  full project context that Faber holds.


Process rules that are maintained throughout:
- Spira never delivers specs directly to Faber. Muse reviews and approves first.
- Faber never makes architectural decisions. Flags ambiguity to Muse.
- Faber does not proceed past any gate without Muse confirmation.
- When Spira brings options, Muse argues them and asks for failure modes.
- When Faber reports results, Muse asks what the result means architecturally
  before giving the next task.

---

### Architecture principle: two injection domains, never mixed

The most important structural rule in the entire system:

```
condition_sum  →  acoustic domain only
                  MERT embeddings go here
                  Nothing else touches this

text channel   →  language domain only
                  All logit bias, forced tokens, RAG passages,
                  session context, and any future steering goes here
```

These two channels are kept strictly separate throughout the entire development
history. Violating this separation would couple the acoustic conditioning to the
language steering in ways that are difficult to debug and impossible to attribute
when something goes wrong.

---

### What the system can do at the point this log was written

The inner monologue side of the system is fully operational:

- Detects phrase boundaries in real time via trublib
- Encodes trumpet audio into a MERT embedding at phrase completion
- Injects the embedding into Moshi's conditioning space
- Analyzes pitch accuracy and tone quality into a 9-state bucket pair
- Forces a grammatically complete observational sentence into the inner monologue
  (PhraseConditioner prefix + bridge token + RAG scaffold)
- Generates continuation with trumpet vocabulary bias active for 24 steps
- Suppresses LaTeX, academic register, and attractor words throughout

The spoken output side — what Moshi actually says out loud — is not yet aligned.
The base model generates incoherent cycling audio when driven by pedagogical inner
monologue content. Aligning the spoken output channel is Phase 6/7 work, after
LoRA v4 training completes.

The interactive lesson capability — session memory, goal tracking, curriculum
progression — is architecturally supported but not yet built. It depends on the
spoken output alignment work being complete first.

---


## Section 2: Infrastructure & trublib

---

### Compute Infrastructure

The project runs on Modal.com, a serverless GPU platform. The primary compute
resource is an H100 GPU, used for:

- MERT inference (phrase embedding at phrase boundaries)
- TensorConditioner training runs
- LoRA training runs
- Streaming inference loop

**Modal volumes (persistent storage across runs):**

```
trubai-checkpoints   — model checkpoints (.pt files)
trubai-hf-cache      — HuggingFace model cache (MERT, Moshi base weights)
trubai-audio-cache   — audio file cache for training runs
```

Modal was chosen over alternatives (Kaggle, local GPU) for two reasons: the H100
keeps MERT loaded persistently between calls (no load/unload cost per phrase), and
the volume system provides persistent checkpoint storage without managing cloud
storage separately.

**MERT latency benchmark (established in Phase 4):**
- Range: 15–70ms
- Mean: 28ms
- Status: not a bottleneck. This question was answered once and has not been
  reopened.

The 28ms MERT latency figure became the benchmark reference for all subsequent
latency discussions — including the RAG retrieval latency constraint in Phase 6.

---

### trublib — Trumpet Activity Detector

trublib is the audio routing layer. It listens to the raw audio stream and decides,
frame by frame, what state the audio is in. Every other component depends on it
knowing when a phrase starts and ends.

**Published on PyPI as `trublib`.**

---

### The 4-State Machine

```
Silent → Onset → Active → Trailing → Silent
```

**Silent:** No significant audio. Nothing is routed to MERT or Moshi.

**Onset:** Audio has crossed the detection threshold but has not yet been confirmed
as trumpet playing. A short confirmation window prevents noise spikes from
triggering the pipeline.

**Active:** Confirmed trumpet playing. Audio is routed to MERT. The ring buffer
accumulates audio frames for embedding at phrase end.

**Trailing:** The student has stopped playing but the phrase is not yet closed.
A hold-off period prevents premature phrase closure on brief pauses within a phrase.
When the first Trailing frame fires, MERT runs on the accumulated buffer, the
condition_sum is updated, and the buffer is cleared.

**Key design property — asymmetric transitions:**
- Hard to enter Active: requires N confirmed onset frames (prevents noise triggers)
- Gradual to exit Active: hold-off before Trailing (prevents phrase fragmentation
  on brief pauses)

This asymmetry is intentional. A false positive (treating noise as a phrase) is
more disruptive than a false negative (missing a very short phrase). The teacher
should not respond to background noise.

---

### Retroactive Ring Buffer

When the system is in Onset state, audio is already being recorded into a ring
buffer — even though Active has not yet been confirmed. When Active is confirmed,
the buffer is flushed from the beginning of the onset, not from the confirmation
point.

Without this, the first N frames of every phrase (the onset confirmation window)
would be silently discarded. Those frames contain the attack — the most
diagnostically important part of the note, where cracks and pitch instability
first appear.

---

### Configuration Parameters

```python
SILENCE_GATE_SECS = 1.5    # default — hold-off before Trailing closes
                            # use 0.6 for diagnostic runs (faster phrase cycling)
```

**STFT parameters (fixed — do not change without re-validating downstream):**
```python
n_fft       = 1024
hop_length  = 256
sample_rate = 24000         # Hz
center      = False         # causal STFT — no future frames required
```

The `center=False` setting is critical for real-time operation. A centered STFT
requires future audio frames to compute the current frame's transform. With
`center=False`, each frame is computed from past audio only — fully causal.

---

### The `is_trumpet` Flag

trublib returns an `is_trumpet` flag per chunk alongside the state. This flag
is used downstream to gate MERT — audio chunks where `is_trumpet=False` are
not accumulated in the ring buffer even during Active state.

This handles the case where background speech or non-trumpet audio bleeds into
the audio stream during a session. A student talking between phrases does not
corrupt the MERT buffer.

---

### What trublib does not do

trublib does not analyze pitch, tone, or any acoustic features beyond the
activity detection needed for state transitions. It is a gatekeeper, not an
analyzer. All acoustic analysis happens downstream in MERT.

---

### Audio Source Catalogue

The dataset used for LoRA training was assembled from the following sources.
License status matters for any future public release.

| Group | Source | Files | License | Partition |
|---|---|---|---|---|
| 1 | Personal recordings (Burak) | 1,810 | Original — fully unrestricted | commercial_safe |
| 2 | Medley-solos-DB / MTG good-sounds | 1,616 | CC BY 4.0 | commercial_safe |
| 3 | VSCO 2 CE | 119 | CC0 | commercial_safe |
| 4 | tinySOL | 161 | CC BY 4.0 | commercial_safe |
| 5 | Philharmonia Orchestra | 187 | ok-for-commercial-use | commercial_safe |
| 6 | Freesound / emirdemirel | 198 | verify-per-file | verify_pending |
| 7–11 | Various Freesound | 302 | verify-per-file | verify_pending |
| 12 | IRMAS | 376 | CC BY-NC-SA 3.0 | nc_flagged |

**Important notes:**

Groups 1–5 are clean professional recordings with no breathiness, cracks, or
technique variation. They are TensorConditioner diversity drivers — used to train
the conditioner to distinguish trumpet-from-not-trumpet across a wide acoustic
range. They are not LoRA training sources.

Groups 6–11 (~265 files, verify_pending) need per-file license verification before
any public release. This has not been done. These files are not currently in the
LoRA training pipeline.

Group 12 (IRMAS) is non-commercial flagged. Not used in training.

The LoRA training dataset was built entirely from Group 1 (personal recordings,
fully unrestricted) via the Track A/B/C preprocessing pipeline described in
Section 8.

---

### Moshi Fork

The project requires modifications to the Moshi model internals that cannot be
achieved through prompting or external hooks. A fork of the Kyutai Moshi
repository is maintained at:

```
github.com/burak-ozenc/moshi
```

The fork diverges from upstream in `moshi/moshi/models/lm.py` only. Changes:

**`_LMGenState` — new field:**
```python
text_steering_delta: torch.Tensor | None = None
# Phase 5b stub — always None in current production version
# Reserved for future text-channel delta injection
```

**`_step()` — forced token injection:**
```python
forced_text_token: torch.Tensor | None = None,
# After text_token = text_token[:, 0, 0]:
if forced_text_token is not None:
    text_token = forced_text_token
```

**`_step()` and `step()` — logit modifier hook:**
```python
text_logit_modifier: callable | None = None
# Applied after text_logits computed, before sampling
# Used by LogitBiasVector
```

**Critical implementation note:** Both modifications operate after
`graphed_main()` returns. They do not cross the CUDA graph boundary. This is
non-negotiable — modifying the CUDA graph itself would require recompilation and
would break the streaming loop.

---

### Moshi Token Architecture

Understanding Moshi's token structure is essential for understanding why every
intervention in the system goes through the text channel.

Moshi produces tokens at each timestep across multiple codebooks:

```
1  text codebook        — inner monologue / semantic channel
8  Moshi audio codebooks  — Moshi's spoken output
8  user audio codebooks   — student's audio input
                           ───
17 codebooks total
```

The text codebook (codebook 0) is the only channel we control. The 8 Moshi audio
codebooks produce the actual sound of speech — rhythm, prosody, voice quality.
These run on base Moshi behavior. We have no direct handle on them.

All conditioning, forced tokens, logit bias, and RAG passages operate exclusively
on the text codebook. The audio codebooks follow from the base model's learned
relationship between semantic content and acoustic output. Getting the audio
codebooks to produce teacher-appropriate speech requires alignment work on the
spoken output channel — this is Phase 6/7, not yet done.

---

## Section 3: TensorConditioner — Retraining History (v1–v13)

---

### What the TensorConditioner does

The TensorConditioner sits between MERT and Moshi. MERT produces a 768-dimensional
embedding that represents the acoustic content of a trumpet phrase. Moshi expects
conditioning in a specific format via its `condition_sum` field. The
TensorConditioner is a learned projection that translates between these two spaces.

The goal: after training, similar-sounding trumpet phrases should produce similar
condition_sum vectors, and acoustically distinct phrases (clean vs. breathy,
in-tune vs. flat) should produce distinct vectors. Moshi's inner monologue
generation is then influenced by which region of the conditioning space the current
phrase maps to.

---

### The Core Problem

The initial training dataset contained 22 pairs of trumpet recordings — clean
recordings paired with variants (flat, sharp, slightly breathy, breathy, cracked).

After training the first conditioner on these 22 pairs, a diagnostic revealed the
fundamental problem:

**All embeddings were clustered at 0.96–0.97 mutual cosine similarity.**

The conditioner had learned a single thing: "this is trumpet audio." It could not
distinguish breathy from cracked, flat from sharp, or any pair from any other pair.
It had become a binary trumpet/not-trumpet gate, not an acoustic conditioner.

The cause: 22 pairs of clean professional recordings, all played on the same
instrument by the same player in the same room, produce MERT embeddings that are
geometrically very close to each other. The conditioner collapsed to their centroid
because that centroid minimized loss.

The fix required two things simultaneously:
1. A diverse training set — recordings from many sources, instruments, and
   conditions — to prevent centroid collapse
2. A contrastive loss term — explicitly penalizing similarity between embeddings
   that should be distinct, and rewarding similarity between embeddings that should
   be similar

This required a complete retraining pipeline, not a fine-tune of the original.

---

### Vocabulary: W_positive and W_negative

The conditioner is trained to steer Moshi's inner monologue toward trumpet
pedagogical vocabulary (W_positive) and away from attractor words (W_negative).

**W_positive (17 tokens — final verified set):**
```python
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
    13369,  # ▁spreading
    19664,  # ▁hollow
]
# embouchure excluded — tokenizes to 5 fragments, no clean injection target
# breathy deduped into ▁breath — tokenizer limitation, documented
```

**W_negative (19 tokens — final verified set):**

Original 12 attractor tokens discovered in early streaming runs:
```
proud, brittle, ville, nik, tower, Az, nationality,
Tournament, disagreement, major (id=916), cherish, mascot
```

Amendment A1 — 7 space-token-region additions (discovered in retrain_v3/v4):
```python
▁     id=260   # bare space token — distributional centroid, confirmed collapse target
▁the  id=262
▁of   id=264
▁and  id=267
▁to   id=269
▁in   id=271
▁a    id=272
```

**Tokenizer note (critical):** `sp.encode(' ' + word)` produces id=260 as a
leading artifact. To verify the bare space token directly:
`sp.piece_to_id('▁')` returns 260. Do not use sp.encode for this check.

---

### Hard Rules (non-negotiable — established through failures)

These rules were established through painful experience and must not be revisited:

1. **gap alone is not a success criterion.** pos_sim MUST be positive at the best
   checkpoint. If pos_sim is negative, the checkpoint is unusable regardless of gap.

2. **All reverts target k=17 universally.** k=17 is the only empirically confirmed
   stable recovery baseline. A sub-phase level is not a valid revert target unless
   it has demonstrated stability in a prior run.

3. **The gap metric from retrain_v2 (0.383) is not a valid criterion for
   diverse-trained conditioners.** Streaming verification is the gate.

4. **output_scale must be register_buffer, not nn.Parameter.** No loss term
   can or should move it.

5. **Anchor prefix fix required in all future runs:**
   `anchor_name = name[3:] if name.startswith('tc.') else name`

---

### Training Metrics Glossary

- **pos_sim:** Cosine similarity between projected trumpet embeddings and the
  W_positive token region. Must be positive — if negative, the conditioner is
  steering away from trumpet vocabulary.
- **neg_sim:** Cosine similarity between projected embeddings and W_negative.
  Should be negative (repulsion).
- **gap:** pos_sim − neg_sim. Measures separation between target and attractor
  regions.
- **proj_norm:** L2 norm of the projected embedding. Should stay near 36.5
  (the output_scale value). Norm collapse (→0) is a catastrophic failure.
- **anc_dist:** Distance from anchor checkpoint. Measures how far the conditioner
  has drifted from its initialization.

---

### retrain_v2 — Documented Baseline

The first successful run. 22-pair dataset, Phase 5 training.

```
Best checkpoint: epoch 104
pos_sim:   +0.036
neg_sim:   −0.347
gap:        0.383
proj_norm: 37.82
```

Streaming verification: PASS. Trumpet vocabulary dominant, zero attractor words.

This run established the gap=0.383 figure that was incorrectly used as a criterion
in later diverse-trained runs. It is not a valid target for diverse datasets —
documented as such after retrain_v13.

Status: Superseded. Documented as baseline reference only.

---

### retrain_v3 and retrain_v4 — FAIL: Space Token Collapse

First attempt to add diversity to the training set.

**Failure mode:** Space token (id=260, `▁`) not in W_negative. The bare space
token is the distributional centroid of the tokenizer — the point equidistant from
all vocabulary tokens in embedding space. When the contrastive loss pushes
embeddings away from W_negative, and W_negative does not include the centroid, the
projection collapses toward the centroid.

**Result:** pos_sim insufficient (+0.0057, +0.0034). The conditioner projected
everything toward `▁` and the 6 most common function words. Streaming output:
`▁ ▁ ▁ ▁ ▁` — bare space cycling.

**Fix applied:** Space token and 6 common function words added to W_negative as
Amendment A1. This fix carries forward to all subsequent runs.

---

### retrain_v5 — FAIL: Phase 3 Magnitude Collapse

Added curriculum: start with anchor-only training, gradually introduce diverse
pairs.

**Failure mode:** lambda_anchor=0.01 was insufficient against 3,203-pair gradient
pressure in Phase 3. The diverse set (thousands of recordings) produces gradients
that overwhelm a weak anchor term. proj_norm collapsed in Phase 3 from ~36 to near
zero.

**What was learned:** The anchor loss coefficient must be tuned relative to the
number of diverse pairs, not set as a fixed small value.

---

### retrain_v6 — FAIL: No Phase 2 Collapse Detection

**Failure mode:** Phase 2 had no collapse detection. By the time Phase 3 began,
the conditioner had already collapsed silently in Phase 2 (proj_norm=22.77,
anc_dist=8.47). Phase 3 inherited a collapsed state and could not recover.
Triple-fire stop at epoch 142.

**What was learned:** Collapse detection must be active in every phase, not just
Phase 3. A collapsed state inherited from an earlier phase cannot be recovered by
later training.

---

### retrain_v7 — FAIL: Single-Epoch Crash Identified

Added Phase 2 bridge trigger and per-epoch logging.

**Failure mode:** Collapse at k=2 minimum diverse pressure. Per-epoch logging
revealed something new: crashes were not accumulated drift — they were
single-epoch catastrophic events. One epoch, the norm was 36. Next epoch, it was 0.
This meant the crash mechanism was not gradual; it was a single gradient step that
moved the projection to zero.

`random.seed(42)` added at this point — crashes became deterministic and
reproducible, which was necessary for diagnosis.

**What was learned:** The crash is a single-step event caused by a single outlier
in the diverse set whose gradient, when applied, catastrophically moves the
projection. The fix cannot be in the loss function (which sees the result, not the
step) — it must be in the gradient computation itself.

---

### retrain_v8 — FAIL: Gradient Clipping Too Late

Added `clip_grad_norm_(max_norm=1.0)`.

**Failure mode:** The clipping was placed at the epoch level — after all 522
records had accumulated gradients via backward passes. By the time clip fired,
the outlier's catastrophic gradient was already encoded in the accumulated gradient
sum. Clipping the sum after accumulation does not undo a catastrophic individual
contribution.

Crash identical to v7 at epoch 131.

**What was learned:** Gradient clipping must be per-step, not per-epoch. The
accumulated gradient is not the same as the per-step gradient from the outlier.

---

### retrain_v9 — FAIL: Per-Record Loop, Direction Problem

Restructured to per-record mini-batch loop: zero_grad → forward → backward →
clip → step for each individual record. Added shuffle of anchor + diverse records
each epoch.

**Failure mode:** 500 sequential gradient steps in the same direction still dragged
proj_norm from 36 to 13 — even with per-step clipping. The diverse set as a
distribution (not as individual outliers) pulls the projection toward a low-norm
region. Per-step clipping bounds magnitude of each step, but it cannot change the
direction. 500 small steps in the wrong direction still arrive at the wrong place.

**What was learned:** The crash mechanism is direction-magnitude coupling. The
diverse set's gradient direction is pulling toward low norm, and neither accumulation
nor clipping addresses this. The fix must decouple direction from magnitude entirely.

---

### retrain_v10 — FAIL: Staged Ramp Is The Wrong Lever

Intra-Phase-3a staged ramp: k=40→70→110→170→250 across 3-epoch stages.

**Failure mode:** Stability boundary confirmed empirically between k=17 (always
recovers after bridge revert) and k=40 (single-epoch crashes occur). The ramp
entered above the stability boundary at its first step (k=40). Single-epoch crashes
occurred even at k=40. Ramp rate is not the problem.

**What was learned:** The stability boundary is at k=17. The ramp is the wrong
instrument — it changes how quickly you enter the unstable region, not whether you
are in it.

---

### retrain_v11 — FAIL: L_guard Cannot See The Crash Coming

Added L_guard loss term: `lambda_guard * clamp(35.0 - proj_norm, min=0)²`

This term activates when proj_norm drops below 35.0, penalizing the loss to
push norm back up.

**Failure mode:** L_guard evaluates at the forward-pass norm. At the crash step,
proj_norm ≈ 36 — the norm looks healthy going into the step. L_guard = 0 because
36 > 35. The crash happens in that step, moving norm from 36 to 0. L_guard never
fired.

The problem: a loss term cannot prevent a crash it evaluates before the crash
occurs. By the time L_guard would have a nonzero value, the crash has already
happened.

**What was learned:** No loss-based approach can prevent a single-step crash where
the norm appears healthy at the start of the step. The fix must be structural —
it must make the crash geometrically impossible, not just penalized.

---

### ConditionerV12 — The Structural Fix

**The insight:** Direction and magnitude are coupled in a standard linear projection.
A single gradient step can move both simultaneously. The fix is to decouple them
entirely — normalize the projection to the unit sphere, then scale by a fixed
constant. The gradient can only affect direction. Magnitude becomes a constant.

```python
# TensorConditioner forward():
direction = F.normalize(proj.squeeze(1), dim=-1)   # always unit sphere
# output_scale is register_buffer initialized at 36.5 — NOT nn.Parameter
x = direction * self.output_scale                   # ~36.5 magnitude always

# Streaming injection (CRITICAL — must use scaled vector):
direction, output_scale = conditioner(mert_embedding)
condition_sum_value = direction * output_scale.detach()  # ~36.5 magnitude
lm_gen._streaming_state.condition_sum = condition_sum_value
# DO NOT inject bare direction (unit-norm) — 37× signal reduction
```

With this structure, proj_norm cannot collapse. It is a constant, not a learned
value. The loss can only move the direction — and moving to zero norm is
geometrically impossible when the output is always normalized then scaled.

**output_scale must be `register_buffer`, not `nn.Parameter`.** If it is a
Parameter, the optimizer will attempt to update it. No loss term should move it.
A buffer is saved and restored with the model state but receives no gradients.

---

### retrain_v12 — PASS (structural fix confirmed, two bugs found)

First run with ConditionerV12 architecture.

```
Zero norm crashes: 0 (across 70 epochs including full Phase 3d at k=3181)
output_scale: frozen at 36.50 throughout — correct
pos_sim: grew from 0.029 → 0.246
gap: 0.221
```

Structural fix confirmed working. Zero crashes at any k value.

**Bug 1 found:** output_scale was accidentally declared as `nn.Parameter`, not
`register_buffer`. No gradient flowed to it (because no loss term targeted it),
so output_scale happened to stay frozen — but it was architecturally wrong and
fragile. Fixed in v13.

**Bug 2 found:** Anchor prefix mismatch. State dict keys were stored as
`tc.learnt_padding` but the anchor comparison used `learnt_padding`. L_anchor
contributed zero throughout the entire v12 run — the anchor constraint was
silently inactive. This explains why pos_sim grew to 0.246 unconstrained: the
contrastive loss had free rein with no anchor term opposing it. gap=0.221 is
below target, explained by inactive anchor.

Both bugs fixed in v13.

---

### retrain_v13 — PASS ✅ PRODUCTION CONDITIONER

Both v12 bugs fixed: output_scale as `register_buffer`, anchor prefix fix applied.

**Loss function:**
```
L_total = L_ce + 0.3 × L_cont + L_anc
```
Three terms. No L_norm, no L_guard (both were failed approaches — not needed with
the structural fix).

**Phase 3 ramp:**
```
3a: k=500
3b: k=1200
3c: k=2000
3d: k=3181
```

**BRIDGE_REVERT:** k=17 universally. Single-condition trigger: pos_sim < 0.03
for 2 consecutive epochs.

**Training result:**
```
Epochs completed: 200
Zero norm crashes: 0

pos_sim:   0.029 → 0.246
neg_sim:   +0.026 (positive throughout — see note below)
anc_dist:  0.82  (< 7.5 target ✓)
gap:       0.185 (gap criterion retired for diverse-trained conditioners)
```

**Streaming verification: PASS.** Zero attractor words. Trumpet vocabulary present.
No space cycling.

**Checkpoint:** `checkpoints/retrain_v13/epoch_200.pt`
**Instantiation:** `ConditionerV12` (not raw `TensorConditioner`)

---

### Why neg_sim stayed positive in v13

This was initially concerning — neg_sim positive means the projection has some
cosine similarity with the W_negative region, which seems wrong. The explanation:

At k=500–3181 diverse pairs (Phase 3), the diverse set dominates the gradient
direction. Diverse recordings represent generic musical content — not trumpet
pedagogy, not attractor vocabulary. They push the projection toward a "generic
music" direction that is equidistant from both W_positive and W_negative. This
means both pos_sim and neg_sim trend positive (equidistant from both means
positive similarity with both).

The anchor term (λ=0.05) cannot overpower 500–3181 diverse gradients in per-record
stepping. At k=17 (bridge revert), the anchor fraction is 56% — pos_sim
immediately returns to 0.35+.

The architecture is correct. The diverse set geometry is the constraint. The
gap criterion (which assumed neg_sim would be strongly negative) is not valid for
diverse-trained conditioners. Streaming verification is the only valid gate.

---

### Critical Usage Notes (must not be forgotten)

**Instantiation:** Always use `ConditionerV12`, never raw `TensorConditioner`.
The production checkpoint was trained with the V12 architecture (normalized
direction + fixed output_scale). Loading it into the wrong class produces silent
incorrect behavior.

**Injection site:**
```python
# CORRECT:
condition_sum_value = direction * output_scale.detach()   # ~36.5 magnitude

# WRONG — 37× signal reduction:
condition_sum_value = direction   # unit norm, not scaled
```

Injecting bare direction (unit-norm vector) reduces the conditioning signal by 37×
relative to what the conditioner was trained to produce. The model will not crash —
it will simply be very weakly conditioned. This bug cost real time in v12.

**TensorConditioner is not exported from `moshi.conditioners.__init__`.**
Import from: `moshi.conditioners.tensors`

---

## Section 4: Phase 4 — LMGen Streaming Loop

---

### What Phase 4 established

Phase 4 was the first end-to-end verification that MERT conditioning actually
influences Moshi's inner monologue generation. Everything before this point was
component-level work. Phase 4 connected the components into a live streaming loop
and measured whether the conditioning signal was real.

The key result: **KL = 11.47** between conditioned and unconditioned inner
monologue distributions. This is a strong signal — not a trace. Conditioning is
real and measurable.

---

### The Streaming Loop Structure

The streaming loop is the runtime core of the system. At each timestep it:

1. Reads an audio chunk from the input stream
2. Passes it through trublib to get the current TAD state
3. If state is Active: accumulates audio into the MERT ring buffer
4. If state transitions to Trailing: runs MERT on the buffer, updates condition_sum
5. At each step regardless of state: calls `lm_gen.step()` with the current
   conditioning and any forced tokens
6. Reads the generated text token and audio tokens from the step result
7. Routes audio tokens to the output stream (Moshi's speech)

The loop runs continuously. It does not pause to wait for a phrase — it is
processing every frame in real time. The phrase boundary events (MERT run,
condition_sum update, PhraseConditioner prime) are interrupts within the loop,
not stops.

---

### Connecting MERT to Moshi: condition_sum

Moshi's LMGen streaming state has a field called `condition_sum`. This is the
acoustic conditioning vector — the value that influences what Moshi generates.

The injection:

```python
# At ACTIVE→TRAILING transition:
mert_embedding = mert_model(audio_buffer)           # [1, 1, 768], float32
direction, output_scale = conditioner(mert_embedding)
condition_sum_value = direction * output_scale.detach()  # ~36.5 magnitude

lm_gen._streaming_state.condition_sum = condition_sum_value
```

This is a direct mutation of the streaming state between `lm_gen.step()` calls.
Moshi's architecture accepts this — the condition_sum is read at the start of each
step, so updating it between steps takes effect immediately on the next step.

**condition_sum is never zeroed between phrases.** The last known embedding stays
live. This means if the student stops playing entirely, the conditioning remains
from the last phrase — the teacher's response is still informed by the most recent
playing. This is the correct behavior for a teacher listening to a student.

---

### ConditionFuser — A Subtle Requirement

Moshi's ConditionFuser requires both `"sum"` and `"cross"` keys to be present,
even when cross-attention is not used.

```python
# Wrong — KeyError at runtime:
fuser_input = {"sum": condition_sum_value}

# Correct:
fuser_input = {
    "sum": condition_sum_value,
    "cross": torch.zeros(...)   # empty cross-attention, must still be present
}
```

Base Moshi has no cross-attention heads. Only condition_sum is used. But the
ConditionFuser validates the presence of both keys before dispatching. This is an
implementation detail that is not documented upstream — it was discovered by
KeyError during Phase 4 integration.

---

### MERT Embedding Requirements

MERT produces embeddings in its native dtype. Moshi expects float32 in the
conditioning space.

```python
mert_embedding = mert_embedding.float()   # explicit cast — do not skip
```

Passing bfloat16 or float16 embeddings into the conditioner produces silent
numerical errors. The conditioner does not validate dtype. The cast must be
explicit.

**Mean pooling:** MERT produces a sequence of embeddings across time frames.
The embedding injected into the conditioner is the mean across the time dimension:

```python
mert_embedding = mert_output.mean(dim=1, keepdim=True)   # [1, 1, 768]
```

This reduces the full phrase representation to a single vector before projection.
The temporal structure within the phrase is collapsed — the conditioner sees the
average acoustic character of the phrase, not its moment-by-moment evolution.
This is a deliberate design choice: the teacher's response is to the phrase as a
whole, not to individual frames within it.

---

### num_codebooks

```python
moshi_model.num_codebooks = 17
# 1 text + 8 Moshi audio + 8 user audio
```

This value matters for correctly indexing the step output. The text token is at
codebook index 0. The Moshi audio tokens are at indices 1–8. The user audio tokens
are at indices 9–16.

When reading the generated text token:

```python
text_token = result[0, 0, 0]   # batch=0, codebook=0, step=0
```

Getting this indexing wrong produces silent garbage — the loop reads an audio
token as if it were a text token, and the inner monologue stream contains
meaningless values.

---

### KL Verification

The KL divergence test compared inner monologue token distributions under two
conditions:

- **Conditioned:** condition_sum set from a real MERT trumpet embedding
- **Unconditioned:** condition_sum set to zeros

**Result: KL = 11.47**

For reference: KL near zero would mean conditioning has no effect. KL = 11.47
means the conditioned distribution is strongly different from the unconditioned
distribution. The conditioning signal is doing real work.

This result closed Phase 4. The connection between trumpet audio and inner
monologue generation was verified as real. All subsequent work builds on this
verified foundation.

---

### What Phase 4 does not guarantee

KL = 11.47 confirms that conditioning changes the distribution. It does not
confirm that conditioning changes it in the right direction — toward trumpet
pedagogy vocabulary and away from attractor words. That question is answered
by streaming verification (which checks actual token output) and by LoRA training
(which shapes what direction the conditioning pushes toward).

Phase 4 answered: "is conditioning real?"
Phase 5 answers: "is conditioning pointing the right way?"

---

### The Streaming Script Versioning

The streaming script has been revised through multiple versions as components
were added:

```
trubai_streaming_v2.py    — early diagnostic, base Moshi only
trubai_streaming_v3.py    — MERT conditioning added
trubai_streaming_v4.py    — TensorConditioner integrated
trubai_streaming_v13.py   — v13 conditioner integrated
trubai_streaming_v14.py   — current production script
                            LogitBiasVector + PhraseConditioner + RAG active
trubai_streaming_v14_diagnostic.py — full token tag logging version
```

All versions are kept in place in `streaming/` rather than archived separately.
The current production script is `streaming/trubai_streaming_v14.py`.

**One constant that must carry forward to any future streaming script:**

```python
BIAS_WINDOW_STEPS = 24   # named constant — not a magic number
```

This defines the logit bias active window. The original diagnostic scripts used
32, which was a measurement error — it included 8 post-window unbiased steps
in the bias verification window, producing false failures. The correct value
is 24. Any new streaming script must use `BIAS_WINDOW_STEPS`, not a hardcoded
integer.

---

## Section 5: Phase 5a — PhraseConditioner & Forced Token Injection

---

### The problem Phase 5a solves

After Phase 4, conditioning was verified as real. But the inner monologue
generation still started from scratch at every phrase — Moshi had no structured
starting point for its response. The base model would begin generating from whatever
token was most probable given the conditioning state, which produced inconsistent
and often incoherent openings.

A real teacher does not start a response from a random word. They start from an
observation about what they just heard. Phase 5a forces that starting point by
injecting a structured opening directly into the inner monologue text channel.

---

### How forced token injection works

The Moshi fork adds a `forced_text_token` parameter to `_step()` and `step()`:

```python
# In moshi/moshi/models/lm.py — _step():
forced_text_token: torch.Tensor | None = None,

# After the model generates its text token prediction:
text_token = text_token[:, 0, 0]
if forced_text_token is not None:
    text_token = forced_text_token   # override the model's prediction
```

When a forced token is provided, the model's own prediction is discarded and
replaced with the forced value. The forced token is then fed back into the model
as context for the next step — so the model's future predictions are conditioned
on the forced tokens having appeared, even though the model did not generate them.

**CUDA graph boundary:** The injection happens after `graphed_main()` returns.
The CUDA graph (which compiles the core transformer computation for speed) is not
modified. This is non-negotiable — modifying inside the graph requires
recompilation and breaks the streaming loop.

---

### PhraseConditioner

PhraseConditioner is the module that decides what to force. It lives in
`conditioner/phrase_conditioner.py`.

At each phrase boundary (ACTIVE→TRAILING transition), it receives a `PhraseFeatures`
object containing two measurements from the phrase just completed:

```python
class PhraseFeatures(NamedTuple):
    pitch_accuracy: float   # normalized autocorrelation peak [0, 1]
    tone_quality: float     # derived from HNR (harmonics-to-noise ratio)
```

**Tone quality derivation:**
```python
tone_quality = clamp((hnr_db - 8.0) / 6.0, 0.0, 1.0)
```

This maps HNR in dB to a [0,1] range. HNR below 8dB → tone_quality=0 (poor).
HNR above 14dB → tone_quality=1 (clean). The 8–14dB range is the transition zone
between acceptable and poor tone.

**Note:** `pitch_accuracy = pitch_salience` in the implementation. Pitch salience
is the normalized autocorrelation peak — how strongly periodic the signal is at
the fundamental frequency. High salience = stable pitch. Low salience = unstable
or drifting pitch.

---

### Bucket Assignment

Each feature is mapped to a three-level bucket:

```python
LOW  < 0.40
MED    0.40 – 0.75
HIGH ≥ 0.75
```

The two buckets form a pair: `(pitch_bucket, tone_bucket)`. Nine possible states:

```
(LOW,  LOW)   (LOW,  MED)   (LOW,  HIGH)
(MED,  LOW)   (MED,  MED)   (MED,  HIGH)
(HIGH, LOW)   (HIGH, MED)   (HIGH, HIGH)
```

---

### The Conditioning Table

Nine entries, one per bucket pair. Each entry is a short sentence fragment that
opens an observational description of what the bucket pair indicates:

```python
CONDITIONING_TABLE = {
    ("LOW",  "LOW"):  "Your air and tone need",
    ("LOW",  "MED"):  "The air column wants",
    ("LOW",  "HIGH"): "Your air support needs",
    ("MED",  "LOW"):  "The tone center wants",
    ("MED",  "MED"):  "Notice how the air",
    ("MED",  "HIGH"): "The tone is nearly",
    ("HIGH", "LOW"):  "Good pitch, the tone",
    ("HIGH", "MED"):  "Your tone center is",
    ("HIGH", "HIGH"): "The column and tone",
}
```

Register requirements for all entries:
- Observational, not prescriptive ("Your tone center is" not "Fix your tone")
- Diagnostic framing — describes what is happening, not what to do
- No encouragement in the inner monologue (encouragement belongs in Moshi's
  spoken output, not in what it thinks)

**MAX_TOKENS = 8.** The conditioning table text is tokenized via Moshi's
SentencePiece tokenizer and capped at 8 tokens. Longer fragments are truncated.

---

### The Three Queues

PhraseConditioner manages three sequential queues that drain in order at each
phrase boundary. Each queue produces one type of forced token:

```
Queue 1: _queue         — primary forced prefix (conditioning table text)
Queue 2: _bridge_queue  — bridge token (one token, closes the prefix clause)
Queue 3: _rag_queue     — RAG passage (up to 12 tokens, completes the sentence)
```

`next_token()` drains Queue 1 first, then Queue 2, then Queue 3. Once all three
queues are empty, `next_token()` returns None and free generation resumes.

The diagnostic tags:
```
[F] — token from _queue (forced prefix)
[B] — token from _bridge_queue (bridge token)
[R] — token from _rag_queue (RAG passage)
[D] — free generation, logit bias active
    — free generation, bias window expired (untagged)
```

---

### `is_active` property

```python
@property
def is_active(self) -> bool:
    return bool(self._queue) or bool(self._bridge_queue) or bool(self._rag_queue)
```

The streaming loop uses `is_active` to determine when to activate the logit bias
window. The [D]-window does not begin until `is_active` returns False — meaning
all three queues have fully drained. This sequencing is critical: the bias must
not activate while forced tokens are still being injected, or the bias window
steps would be consumed during forced injection.

---

### `prime()` — called at every phrase boundary

```python
def prime(self, features: PhraseFeatures) -> None:
    # Compute bucket pair
    pitch_bucket = _bucket(features.pitch_accuracy)
    tone_bucket  = _bucket(features.tone_quality)
    bucket_pair  = (pitch_bucket, tone_bucket)

    # Queue 1: conditioning table text
    text = CONDITIONING_TABLE[bucket_pair]
    ids  = self._sp.encode(text)[:self.MAX_TOKENS]
    self._queue.clear()
    self._queue.extend(ids)

    # Queue 2: bridge token
    self._bridge_queue.clear()
    bridge_id = BRIDGE_TOKEN_IDS.get(bucket_pair)
    if bridge_id is not None:
        self._bridge_queue.append(bridge_id)

    # Queue 3: RAG passage
    self._rag_queue.clear()
    rag_text = RAG_PASSAGES.get(bucket_pair)
    if rag_text is not None:
        rag_ids = self._sp.encode(rag_text)[:self.RAG_MAX_TOKENS]
        self._rag_queue.extend(rag_ids)
```

**Cross-phrase reset — mandatory, do not remove:**

```python
if tad_result.phrase_boundary:
    bias_steps_remaining = 0    # cancel any active bias window from prior phrase
    phrase_conditioner.prime(phrase_features)   # reprime all three queues
```

If a new phrase boundary fires while the previous phrase's queues are still
draining, `prime()` clears and repopulates all queues from scratch. The prior
phrase's forced sequence is cancelled. This prevents contamination between phrases.

---

### The Display Artifact

The diagnostic output shows a one-step lag between what is forced and what is
displayed. This is not a bug — it is a consequence of how the streaming loop
reads the step result.

At each step, `next_token()` returns the token to force as *input* to
`lm_gen.step()`. The step result `result[0, 0, 0]` contains the model's
*prediction* — which is the model's response to the previous step's context.
The model predicts what comes after what was just forced, not what was just forced.

Mapping for a sample phrase (HIGH, MED):
```
next_token() forces → model displays
  260  (▁)          → ▁spreading   [R]   (bridge context → model predicts continuation)
  2440 (▁slightly)  → ▁            [R]
  302  (▁at)        → ▁slightly    [R]
  262  (▁the)       → ▁at          [R]
  5461 (▁phrase)    → ▁the         [R]
  606  (▁end)       → ▁phrase      [R]   ← 6th and final RAG token forced here
  None (free)       → ▁end         [D]   ← model predicts given forced context
```

The display is one step behind the forced sequence. All 6 RAG tokens were forced
even though ▁end appears with a [D] tag. This is the confirmed display artifact —
it appears consistently across all forced token types ([F], [B], [R]) and is
not a drain error.

---

### Phase 5a Streaming Verification

Verified before Phase 5b work began. Criteria:

```
§10-1  LaTeX tokens in [D]-window:  0        ✓
§10-2  Trumpet tokens in [D]-window: ≥ 3     ✓ (6 tokens)
§10-4  Forced prefix intact:         ✓
§10-5  Audio continuous:             ✓
```

The [D]-window criterion measures exactly `BIAS_WINDOW_STEPS` (24) steps — not
32. The original 32-step criterion was a spec error that included 8 post-window
    unbiased steps as false failures. This was corrected when the error was identified,
    and `BIAS_WINDOW_STEPS = 24` became a named constant in the streaming script.

---

## Section 6: Phase 5b — LogitBiasVector & Bridge Tokens

---

### The problem Phase 5b solves

After Phase 5a, the inner monologue had a structured forced opening. But the free
generation that followed the forced prefix was still unreliable. Three distinct
failure modes appeared across different stages of development:

**Fragment cycling (Phase 5b-C era):**
```
▁air ▁air ▁over ▁air ▁air ▁over ▁air ...
```
The model entered a repetitive loop on trumpet vocabulary fragments.

**LaTeX/academic register collapse (v13 streaming era):**
```
Gmina  pgfplots  martingale  HCl  filecontents
```
The model drifted into academic and LaTeX vocabulary completely unrelated to
trumpet pedagogy. This was the dominant failure mode after v13 was integrated.

**▁Mr cycling (post-LoRA v3/epoch-20):**
```
▁Mr ▁Mr ▁Mr ▁Mr ...
```
After LoRA v3 training, ▁Mr became the dominant free-generation token in the
post-window region. Max run length up to 5, high total occurrence count.

Phase 5b addresses these failures with two mechanisms: bridge tokens (5b-C) and
the LogitBiasVector (5b-A).

---

### Phase 5b-C — Bridge Tokens

**What they do:** A bridge token closes the forced prefix clause. It is a single
token injected from Queue 2 (_bridge_queue) immediately after the forced prefix
drains. Its role is to provide a grammatically complete sentence boundary before
the RAG passage begins.

**Bridge token table (one per bucket pair):**

```python
BRIDGE_TOKEN_IDS = {
    ("LOW",  "LOW"):  711,    # support
    ("LOW",  "MED"):  711,    # support
    ("LOW",  "HIGH"): 2484,   # opening
    ("MED",  "LOW"):  711,    # support
    ("MED",  "MED"):  7654,   # feels
    ("MED",  "HIGH"): 15654,  # flowing
    ("HIGH", "LOW"):  1215,   # needs
    ("HIGH", "MED"):  13369,  # spreading
    ("HIGH", "HIGH"): 3610,   # sounds
}
```

**Option C falsified:** The original hypothesis was that closing the forced clause
with a bridge token would change the continuation prior — the model, having seen
a complete clause, would generate the next clause differently. This was tested and
falsified. Closed predicates do not change the continuation prior. Free generation
cycling resumed immediately after the bridge token drained.

The problem was not clause closure — the LoRA had no structural signal for what
comes after any clause boundary. Bridge tokens alone cannot fix an undertrained
prior.

**Bridge tokens are kept in production despite this.** They are not hurting
anything, they provide a grammatically clean join point between the forced prefix
and the RAG passage, and removing them would require re-verifying the full sequence.
They remain active in `trubai_streaming_v14.py`.

---

### Phase 5b-A — LogitBiasVector

**What it does:** A static vector of size [32000] (the full vocabulary size) is
added to the text logits before sampling, during a bounded active window after
each phrase boundary. Positive values attract. Negative values suppress.

```python
class LogitBiasVector:
    def apply(self, text_logits: torch.Tensor) -> torch.Tensor:
        return text_logits + self.alpha * self._bias.to(
            device=text_logits.device,
            dtype=text_logits.dtype
        )
```

The modifier is injected via the `text_logit_modifier` hook added to the Moshi
fork's `_step()`. It fires after logits are computed, before sampling. CUDA graph
safe — operates after `graphed_main()` returns.

**Active window:** `BIAS_WINDOW_STEPS = 24` steps post-phrase-boundary, starting
when all forced queues have drained (is_active = False).

**alpha = 0.5 (committed).** The alpha parameter scales the entire bias vector.
Too high: collateral suppression of legitimate vocabulary. Too low: insufficient
suppression of LaTeX register. 0.5 was established through calibration and is
not adjusted per-run.

---

### Derivation History: Three Rejected Versions

#### First derivation — REJECTED: Global formula, catastrophic function word bias

The initial approach applied a log-ratio formula globally:

```
bias(token) = log(p_ped(token)) - log(p_fail(token))
```

Where p_ped was the probability under the pedagogical distribution and p_fail
under the failure distribution.

**Failure:** The failure sample contained only 44 unique tokens. For all other
tokens in the vocabulary (including common function words like `▁the`, `,`, `.`),
p_fail = 0. The formula produced +16 bias for function words because
log(p_fail) → -∞.

Result: the model collapsed to near-deterministic function word output. Pedagogical
vocabulary was drowned out by artificially boosted grammar tokens.

**Fix:** Split the positive and negative sides. Apply the formula only to tokens
that actually appear in the relevant distribution.

---

#### bias_v1.pt — REJECTED: W_positive attractor basin

Split derivation applied:
- Negative side: formula applied only to the 44 failure tokens
- Positive side: W_positive vocabulary only, capped at +8.0

**Failure:** All 17 W_positive tokens simultaneously at +7–8 created a closed
attractor basin. The logit surface had a sharp peak over W_positive. The LoRA
could not escape it. W_positive cycling replaced LaTeX cycling — the model
looped over trumpet vocabulary tokens instead of LaTeX tokens, but the cycling
problem was unchanged.

**Fix:** Zero the positive side entirely. Suppression-only approach.

---

#### bias_v2.pt — REJECTED: Insufficient suppression at alpha=0.5

Positive side zeroed. Suppression-only at alpha=0.5.

**Calibration result:** 1.7 LaTeX tokens per 32-step window — not zero. Three
specific tokens were leaking through suppression:

1. `▁incarnation` (id=25888): Had a bias value of 0.0 — it was not in the failure
   token set at all, so it received zero suppression. An unimpeded leaker.

2. `▁significantly` (id=1117): In the failure set at −3.39 bias but appeared 128
   times in the failure sample. The suppression was too weak for its frequency.

3. `▁Mr` (id=2048): In the failure set at −0.62 bias. Weak suppression, high
   occurrence, dominant in post-window free generation.

**alpha=1.0 attempted:** Strengthening to alpha=1.0 eliminated LaTeX but caused
collateral suppression — trumpet vocabulary density dropped from 22.1% to 15.6%,
below the established baseline. alpha=1.0 rejected.

**Fix:** Three targeted patches rather than raising alpha globally.

---

### bias_v3.pt — PRODUCTION ✅

Three patches applied to bias_v2:

```python
# Patch 1: ▁incarnation (id=25888) — zero entry, unimpeded leaker
0.0 → −14.0   # strong suppression added

# Patch 2: ▁significantly (id=1117) — strengthened
−3.39 → −7.0  # 128 failure occurrences, needed harder suppression

# Patch 3: ▁Mr (id=2048) — moderate strengthen
−0.62 → −4.0  # preserve subword role, reduce cycling frequency
```

**Calibration result at alpha=0.5:**
```
LaTeX tokens in 24-step window:  0
Trumpet tokens per 32 steps:     7.7
Unique ratio:                    0.646
Coherence assessment:            PEDAGOGICAL
Audio:                           continuous
```

**▁Mr cycling in post-window:** Still present (max_run=5, total=98 in
diagnostic runs). This is post-window free generation — outside the 24-step
bias window, the model's prior takes over. The ▁Mr cycling is a LoRA data
limitation, not a bias failure. It is confined to post-window steps and does
not appear inside the [D]-window.

---

### Failure Token Distribution

The failure token set was assembled from two diagnostic runs:

- v13 streaming diagnostic (dominant failures: ▁Mr id=2048 count=234,
  ▁significantly id=1117 count=128)
- Phase 5b-C diagnostic (LaTeX register: ▁Gmina id=17035, pgfplots id=24388,
  martingale id=27459, etc.)

**Final set:**
```
44 unique tokens
767 total occurrences
```

Stored in `data/failure_tokens.json`. Two distinct registers in one file:
LaTeX/academic vocabulary from the early v13 runs, and the ▁Mr/▁significantly
cluster from the LoRA v3 era.

---

### The Logit Modifier Hook in the Moshi Fork

The `text_logit_modifier` is a callable parameter added to `_step()` and `step()`:

```python
# In moshi/moshi/models/lm.py:
text_logit_modifier: callable | None = None

# Applied after logits computed, before sampling:
if text_logit_modifier is not None:
    text_logits = text_logit_modifier(text_logits)
```

In the streaming loop:

```python
logit_bias_vector = LogitBiasVector(bias_path, alpha=0.5)

# At each step, when bias window is active:
modifier = logit_bias_vector.apply if bias_steps_remaining > 0 else None

result = lm_gen.step(
    ...,
    text_logit_modifier=modifier
)

if bias_steps_remaining > 0:
    bias_steps_remaining -= 1
```

The modifier is passed as None when the bias window has expired. The streaming
loop tracks `bias_steps_remaining` and decrements it each step. When it reaches
zero, the modifier is no longer passed — free generation resumes with no bias.

---

### Combined Stack Streaming Verification

Final verification with all three components active simultaneously:
TensorConditioner (v13) + LoRA (v3/best) + LogitBiasVector (bias_v3) +
PhraseConditioner + RAG passages.

```
LaTeX in 24-step window:     0
Trumpet vocabulary density:  23.96%
Unique ratio:                0.646
Coherence:                   PEDAGOGICAL (Muse assessment)
Audio:                       continuous
```

The combined stack produces trumpet-vocabulary-biased inner monologue fragments
with complete grammatical scaffold sentences for the first 12 forced steps.
Post-window free generation remains incoherent (vocabulary present, sentence
structure absent) — this is the data ceiling for 22 training pairs. The RAG
layer addresses this in Phase 6.

---

### W_positive Token IDs — Final Reference

```python
W_POSITIVE_IDS = [
    1142,   # ▁air
    2368,   # ▁column
    9064,   # ▁tone
    8735,   # ▁breath      (covers breathy — tokenizer limitation)
    16252,  # ▁aperture
    1611,   # ▁center
    6615,   # ▁crack        (root stem)
    3077,   # ▁flat
    6064,   # ▁sharp
    6396,   # ▁pitch
    4107,   # ▁partial
    21938,  # ▁buzz
    1563,   # ▁focus
    24657,  # ▁pinch        (root stem for pinched)
    11984,  # ▁diffuse
    13369,  # ▁spreading
    19664,  # ▁hollow
]
```

Notes:
- `embouchure` tokenizes to 5 fragments — no single token form exists.
  Excluded entirely.
- `breathy` tokenizes as `▁breath` + `y`. Uses ▁breath (id=8735) as proxy.
  Documented limitation.
- These IDs are for Moshi's SentencePiece tokenizer
  (`tokenizer_spm_32k_3.model`, vocab 32000). They are not portable to other
  tokenizers.

---

## Section 7: LoRA Training History (v3 → v4)

---

### What the LoRA does

LoRA (Low-Rank Adaptation) is a technique for fine-tuning a large model by adding
a small number of trainable parameters on top of frozen base weights. Instead of
retraining the entire model, LoRA inserts small low-rank matrices at specific
layers. During training, only these matrices are updated. The base model is
untouched.

For TRUB.AI, the LoRA trains on top of Moshi's base weights. Its job: shift the
inner monologue token distribution toward trumpet pedagogical vocabulary. Without
the LoRA, the base Moshi model produces generic conversational text — or, with
conditioning alone, drifts into academic register. The LoRA is what makes trumpet
vocabulary (▁air, ▁column, ▁tone, ▁center, ▁pitch) the natural output register.

---

### Training Data Format

Each training example is an (audio, inner_monologue) pair:

```python
{
    "file": "c_major_proper__flat60c.wav",
    "label": "flat",
    "cents": -74,
    "inner_monologue": "The pitch is a severe drop below center"
}
```

The inner monologue is always:
- One sentence maximum
- Observational/diagnostic register — describes what the playing indicates
- No encouragement ("Great job!") — encouragement belongs in spoken output
- No prescription ("Support your air") — diagnoses, does not instruct

The model is trained to produce this inner monologue text given the acoustic
conditioning from the corresponding audio file. It learns to associate MERT
embedding regions with specific pedagogical language patterns.

---

### The 22-Pair Original Dataset

The original training set contained 22 labeled pairs across four scale patterns
(C major, G major, Bb major, descending C) with five variants each (in tune,
flat 45c, flat 60c, sharp 45c, sharp 60c) plus two crack pairs.

**Full 22-pair inner monologue dataset:**

```
Pair 01 [c_major_proper, in tune]:
  "The tone feels slightly diffuse with breath around the core"
Pair 02 [c_major_proper_flat60c, -74c]:
  "The pitch is a severe drop below center"
Pair 03 [c_major_proper_flat45c, -59c]:
  "The pitch is a notable drop below center"
Pair 04 [c_major_proper_sharp45c, +31c]:
  "The pitch pushes noticeably above center with slight breathiness"
Pair 05 [c_major_proper_sharp60c, +46c]:
  "The pitch pushes hard above center with slight breathiness"
Pair 06 [g_major_ascending, in tune]:
  "The tone feels centered with a whisper of breath escaping"
Pair 07 [g_major_ascending_flat60c, -67c]:
  "The pitch is a significant sag below center"
Pair 08 [g_major_ascending_flat45c, -52c]:
  "The pitch is a moderate sag below center"
Pair 09 [g_major_ascending_sharp45c, +38c]:
  "The tone feels slightly pressed above the slot center"
Pair 10 [g_major_ascending_sharp60c, +53c]:
  "The tone feels noticeably pressed above the slot center"
Pair 11 [bb_major_ascending, in tune]:
  "The air column feels spread and diffuse before the aperture"
Pair 12 [bb_major_ascending_flat60c, -59c]:
  "The pitch is a full collapse below center"
Pair 13 [bb_major_ascending_flat45c, -44c]:
  "The pitch is a gentle collapse below center"
Pair 14 [bb_major_ascending_sharp45c, +46c]:
  "The pitch pushes above center with breath leaking through"
Pair 15 [bb_major_ascending_sharp60c, +61c]:
  "The pitch pushes hard above center with diffuse air escaping"
Pair 16 [c_major_descending, in tune]:
  "The tone feels open and breathy through the descent"
Pair 17 [c_major_descending_flat60c, -54c]:
  "The pitch collapses below the slot with breath escaping"
Pair 18 [c_major_descending_flat45c, -39c]:
  "The pitch drifts below the slot with an airy quality"
Pair 19 [c_major_descending_sharp45c, +51c]:
  "The sound is a spread push above the slot center"
Pair 20 [c_major_descending_sharp60c, +66c]:
  "The sound is a wide push above the slot center"
Pair 21 [low_cracked_c4_bb3, breathy]:
  "The note breaks apart as the embouchure loses its seal"
Pair 22 [middle_cracked_g4_f4_v2, slightly_breathy]:
  "The attack cracks as the air splits between partials"
```

Distribution: Nominal 8 (pairs 02,03,07,08,12,13,19,20), Verbal 8
(pairs 04,05,14,15,17,18,21,22), Adjectival 6 (pairs 01,06,09,10,11,16).

---

### lora_v3 — Training History

#### Resume Bug (Critical — Fixed)

During the v3 training run, a resume from checkpoint produced a catastrophic CE
spike. Diagnosis: AdamW optimizer state was not saved with the checkpoint. On
resume, AdamW reinitializes fresh moment estimates (zeros). Zero moment estimates
mean the adaptive learning rate effectively applies full LR to all trained weights —
the model sees a large unexpected gradient step on weights that had already converged.

```
Epoch 20 (before resume): CE = 3.8088
Epoch 21 (first step after resume, fresh AdamW): CE = 33.27   ← spike
Epoch 25 (recovery): CE = 3.63   ← below pre-resume baseline
```

**Fix:** Save and restore optimizer state at every gate epoch:

```python
# At every gate:
torch.save(model.state_dict(),     f"checkpoints/lora_v3/epoch_{epoch:03d}.pt")
torch.save(optimizer.state_dict(), f"checkpoints/lora_v3/optimizer_{epoch:03d}.pt")

# On resume:
model.load_state_dict(torch.load(f"checkpoints/lora_v3/epoch_{epoch:03d}.pt"))
optimizer.load_state_dict(torch.load(f"checkpoints/lora_v3/optimizer_{epoch:03d}.pt"))
# DO NOT reinitialize optimizer
```

This fix is mandatory for all future LoRA training runs.

#### Gate History

| Epoch | train_ce | eval_ce | train_mg | Notes |
|---|---|---|---|---|
| 20 | 3.8088 | 3.8088 | — | Baseline (pre-resume) |
| 21 | — | — | — | CE spike 33.27 (resume bug) |
| 25 | 3.63 | — | — | Recovery, below baseline |
| 40 | 3.1995 | 4.2207 | 1.2061 | |
| 60 | 3.2748 | 4.1456 | 1.0292 | Best checkpoint |
| 75 | 3.2289 | — | 1.0246 | Stop — plateau confirmed |

**Best checkpoint: lora_v3/best = epoch 60, eval_ce = 4.1456**

#### Plateau Analysis

Training plateaued at approximately epoch 30. Train CE stabilized at 3.19–3.28
with no meaningful movement through epoch 75. Margin loss stabilized at 1.0–1.2
from epoch 25, never below 0.5.

KL regularizer: KL = 0.0000 throughout the entire run. The KL regularizer was
completely ineffective — it contributed zero loss signal. This does not affect
the outcome because the LoRA's purpose is vocabulary attraction, not closeness
to the reference distribution. For future alignment work (Phase 6/7), use DPO-LN
instead of KL — they address different problems.

#### Why the plateau is a data ceiling, not a training failure

21 training pairs (pair 22 held out for eval) cannot supply enough grammatical
variety for sentence-level predicate completion. The LoRA extracted everything
available from 21 pairs. Continuing past epoch 75 would not have improved eval_ce.

**Streaming verification result (lora_v3/best):**

```
Attractor words:          absent ✓
Trumpet vocabulary:       124 hits in 562 tokens (22% density) ✓
Space cycling:            absent ✓
▁Mr cycling:              post-window only, max_run=3, total=112
                          (LoRA free distribution, not a spec failure)
```

**Coherence assessment (Muse):** Not sentence-level coherent. Directional
vocabulary (▁above, ▁below, ▁directly, ▁centered) is appropriate register —
plausible pedagogical fragments, not assembled into sentences. This is the data
ceiling. More training epochs will not fix it. This is a data boundary.

---

### The Data Boundary and Why It Matters

The distinction between a data boundary and a training failure is important:

**Training failure:** The model could produce better output with the same data if
trained differently (different learning rate, more epochs, better loss function).

**Data ceiling:** The model has extracted everything the training data contains.
No training change will improve it because the information is not in the data.

lora_v3's plateau at epoch 30 after the resume spike consumed ~4 epochs is a data
ceiling. The training was not optimal but the ceiling was real regardless. Even
with a perfect training run, 21 pairs of grammatically similar sentences cannot
teach sentence-level structure.

The fix is more data. The expanded dataset (4,815 pairs) is what lora_v4 is
trained on.

---

### lora_v4 — Current Training Run

#### Expanded Dataset

4,815 labeled pairs assembled from three tracks:

| Label | Count | Source |
|---|---|---|
| slightly_breathy | 1,581 | Track A — shaped noise synthesis |
| breathy | 1,581 | Track A — shaped noise synthesis |
| flat | 814 | Track B — pitch shifting (−45c, −60c) |
| sharp | 814 | Track B — pitch shifting (+45c, +60c) |
| cracked | 25 | Track C — manual recordings |

Track B register breakdown: Low=180 source files, High=77, Mid=150 (capped).
High register pairs oversampled at 2.5× in the training loop to prevent
dilution by the larger Mid category.

#### Stratified Split

```python
random.seed(99)

eval_fractions = {
    "slightly_breathy": 0.05,
    "breathy":          0.05,
    "flat":             0.08,
    "sharp":            0.08,
    "cracked":          0.20,   # higher — only 25 pairs total
}
```

Split result:

| Label | Total | Eval | Train |
|---|---|---|---|
| breathy | 1,581 | 79 | 1,502 |
| cracked | 25 | 5 | 20 |
| flat | 814 | 65 | 749 |
| sharp | 814 | 65 | 749 |
| slightly_breathy | 1,581 | 79 | 1,502 |
| **TOTAL** | **4,815** | **293** | **4,522** |

#### Gate Schedule

Gate epochs: 20, 40, 60, 80, 100. Optimizer state saved at every gate.
Faber reports to Muse at every gate before continuing. Muse reviews trajectory
and issues continuation or stop instruction.

#### Success Criteria

- eval_ce < 4.1456 (must beat lora_v3/best)
- Streaming verification PASS (same criteria as v3)
- Trumpet vocabulary density ≥ 22%
- Coherence improvement: target sentence-level fragments, not just vocabulary
  scatter

**eval_ce target is necessary but not sufficient.** A lower eval_ce that produces
worse streaming coherence is not a success. Streaming verification is the final
gate.

#### Status

Training in progress. Gate reports pending.

---

### Inner Monologue Templates for Expanded Dataset

The expanded dataset uses simplified templates relative to the 22-pair set.
One template per label, not per individual recording:

**Track A — breathiness:**
```
slightly_breathy: "The tone carries a thin breath layer around the core"
breathy:          "The air splits around the tone, losing center"
```

**Track B — pitch variants:**
```
flat60c:   "The pitch drops severely below center"
flat45c:   "The pitch drops noticeably below center"
sharp45c:  "The pitch pushes noticeably above center"
sharp60c:  "The pitch pushes hard above center"
```

**Track C — cracks (register-assigned):**
```
crack_low_*:   "The note breaks apart as the embouchure loses its seal"
crack_mid_*:   "The attack cracks as the air splits between partials"
crack_high_*:  "The partial collapses as the aperture loses its seal"
```

All templates follow the same register rules as the 22-pair set: observational,
diagnostic, no encouragement, no prescription.

---

### KL Regularizer — Why It Does Not Work and What To Use Instead

The KL regularizer in lora_v3 computed the KL divergence between the LoRA's
output distribution and the base model's distribution, penalizing large divergence.
The intent was to prevent the LoRA from drifting too far from base Moshi behavior.

**Why it failed:** KL=0.0000 throughout the run. The regularizer never activated.
The likely cause: the LoRA's weight updates were small enough in magnitude that
the output distribution, when measured by KL, appeared unchanged — even though
the vocabulary distribution had shifted meaningfully.

**For Phase 6/7 alignment:** Use DPO-LN (length-normalized Direct Preference
Optimization), not KL regularizer. These solve different problems:

- KL regularizer: tries to keep the model close to a reference distribution.
  Failed because it measures the wrong thing for this application.
- DPO-LN: trains on relative preference pairs (preferred response vs. rejected
  response). Does not require absolute closeness to a reference. Stable for
  variable-length spoken dialogue. Confirmed effective by the Kyutai alignment
  paper (arXiv:2506.21463).

---

## Section 8: Dataset Expansion — Tracks A, B, C

---

### Why the dataset needed expanding

The original 22-pair dataset established the LoRA's vocabulary register but hit
a data ceiling at epoch 30. The ceiling was not a training problem — 21 pairs
simply cannot supply enough grammatical and acoustic variety for sentence-level
coherence. The fix required more data.

Three tracks were defined based on what could realistically be produced without
new professional recordings:

- **Track A:** Synthesize breathiness variants from existing clean recordings
- **Track B:** Synthesize pitch variants from existing clean recordings
- **Track C:** Manually record crack events (the only category that cannot
  be synthesized)

All three tracks source from Group 1 (1,810 personal recordings, fully
unrestricted license). The expanded dataset contains only commercially safe
material.

---

### Track A — Breathiness Synthesis

**Goal:** Produce slightly_breathy and breathy variants from clean recordings,
without recording new material.

**Source:** Group 1, 1,810 WAV files, 24kHz, 1–5 seconds, clean personal
recordings.

#### Pipeline

**Step 1 — HNR gate:**
Each source file is passed through `praat-parselmouth` to measure HNR
(harmonics-to-noise ratio). Only files with HNR ≥ 14dB pass. Files below
this threshold are already too noisy to use as clean sources — synthesizing
breathiness on top of an already-breathy file is acoustically incoherent.

**Step 2 — Shaped noise generation:**
White noise is generated at the source file's sample length, then filtered:

```python
# Butterworth low-pass filter, order 6, cutoff 3kHz
b, a = scipy.signal.butter(6, 3000 / (sample_rate / 2), btype='low')
shaped_noise = scipy.signal.filtfilt(b, a, white_noise)
shaped_noise = shaped_noise / np.max(np.abs(shaped_noise))  # normalize
```

The 3kHz cutoff shapes the noise to resemble breath — breath noise is
predominantly low-frequency turbulence, not broadband white noise. White
noise alone sounds electronic. Shaped noise sounds more like real air
turbulence. Known limitation: it still sounds more synthetic than real breath
noise. Flagged for revisit if LoRA training shows breathiness confusion.

**Step 3 — Adaptive mix level via binary search:**
Instead of a fixed mix level (which would produce different HNR results across
files with varying source HNR), the mix level is found per file via binary
search:

```python
# Target tiers:
# slightly_breathy: HNR 8–14dB
# breathy:          HNR < 8dB

# Binary search between -30dB and 0dB, 20 iterations max
# For each candidate level: mix, measure output HNR, adjust bounds
```

The binary search converged at approximately −11dB for slightly_breathy and
−7.5dB for breathy on the test files. These values varied per source file
depending on source HNR.

**Step 4 — Output HNR verification:**
The synthesized file's actual HNR is measured and verified to fall in the
target tier. Variants that fall outside their target tier are discarded.

**Output files:**
```
{stem}__slightly_breathy.wav
{stem}__breathy.wav
synthesis_log.json
```

**Result:** 3,162 pairs built (0 source files skipped by HNR gate — all
1,810 source files passed, producing two variants each, minus any that failed
output HNR verification).

The expected yield was 2,000–2,500. The higher actual yield (3,162) indicates
the Group 1 recordings were cleaner than the test sample suggested — more files
passed both the source HNR gate and the output verification than projected.

**Known limitation for future reference:** White noise synthesis sounds noisy
rather than naturally breathy. Real breath noise recorded as an additive signal
would be more acoustically accurate. If lora_v4 training shows confusion
between breathiness tiers, this is the first thing to revisit.

---

### Track B — Pitch Variant Pairs

**Goal:** Produce flat and sharp variants at ±45 and ±60 cents from clean
sustained-tone recordings.

**Source:** Same Group 1 pool, 1,810 files.

#### Pipeline

**Step 1 — F0 extraction:**
```python
f0, voiced_flag, voiced_probs = librosa.pyin(
    audio,
    fmin=58,     # Hz — below Bb2, lowest playable trumpet note
    fmax=1568,   # Hz — above G6, above normal trumpet range
    sr=sample_rate
)
```

F0 is extracted across the full file. The median of voiced frames only is used
for register classification — this ignores silence and noise frames.

**Step 2 — Register classification:**
Files are classified into three priority groups:

```
P1 Low:    median F0 ≤ 233Hz  (Bb3 and below)      — priority
P2 High:   median F0 ≥ 784Hz  (G5 and above)        — priority
P3 Mid:    duration ≥ 2.0s AND F0 std < 20¢          — sustained mid tones
All else:  skipped (scales, runs, arpeggios)
```

Scales and runs are correctly excluded by the F0 std criterion — a scale
played across an octave has F0 standard deviation in the hundreds of cents.
Only sustained tones qualify.

**Threshold tuning during development:**

| Parameter | Initial | Final | Reason |
|---|---|---|---|
| P3_MIN_DURATION_SEC | 3.0s | 2.0s | Recover clean short sustained tones |
| P3_MAX_F0_STD_CENTS | 15¢ | 20¢ | Recover borderline sustained tones at 16–19¢ |

The threshold relaxation recovered significantly more mid-register files than
the 5-file test run suggested — the full 1,810-file pool had far more sustained
tones in the 2.0–3.0s / 16–20¢ range than the sample indicated.

**Step 3 — Pitch shifting:**
Four shifts applied per qualifying file via `pyrubberband`:

```python
shifts = [-60, -45, +45, +60]  # cents

for cents in shifts:
    ratio = 2 ** (cents / 1200)
    shifted = pyrubberband.pitch_shift(audio, sample_rate, ratio)
    # write {stem}__flat60c.wav, etc.
```

Whole-file shift — the entire recording is shifted uniformly. The median F0
used for register classification is not affected by the shift (classification
happens before shifting).

**Hard limit: no shifts beyond ±60 cents.** Above this threshold the result
sounds like a different partial, not a sharp or flat variant of the same note.
This is acoustically incoherent as a training example.

**Output files:**
```
{stem}__flat60c.wav
{stem}__flat45c.wav
{stem}__sharp45c.wav
{stem}__sharp60c.wav
pitch_pairs_log.json
```

**Register breakdown of qualifying files:**

| Register | Source files | Shifted files |
|---|---|---|
| P1 Low (≤ Bb3) | 180 | 720 |
| P2 High (≥ G5) | 77 | 308 |
| P3 Mid sustained | 287 (capped at 150) | 600 |

High register (P2) was underrepresented at 77 source files — less than half
of Low, less than a third of uncapped Mid. This is a known gap. Mid was capped
at 150 source files to prevent it from swamping Low and High at training time.
High register pairs are oversampled at 2.5× in the lora_v4 training loop as
additional compensation.

**Total Track B pairs:** 1,628 (after stratification cap).

---

### Track C — Crack Recordings

**Goal:** Produce crack pair recordings. This is the only category that cannot
be synthesized — a crack event is a physical embouchure failure, and no
signal processing operation can convincingly produce one from a clean recording.

**Source:** New manual recordings by Burak. Instrument: Bb trumpet. Recording
conditions: same as Group 1 (dry room, close mic, mono, WAV 24kHz).

#### The Crack Mechanic

Two types of cracks were recorded:

**Embouchure tension release (Sessions 1 and 3):** The player allows embouchure
tension to relax mid-note, causing the lip aperture to lose its seal and the
note to drop to a lower partial.

**Attack air split (Session 2):** The attack air pressure splits between two
adjacent partials rather than locking onto one. The note begins and immediately
cracks to a different partial.

A usable crack: f0 must visibly discontinue or split in the spectrogram. Notes
that stay on the same partial throughout — even if unstable — are pitch variants,
not cracks.

#### Session Plan and Results

| # | Session | Start Note | Crack Target | Concert Pitch | Takes |
|---|---|---|---|---|---|
| 1 | Low | C4 | Bb3 | C4→Bb3 | 5 |
| 2 | Low | Bb3 | F3 | Bb3→F3 | 5 |
| 3 | Mid | G4 | F4 | G4→F4 | 5 |
| 4 | Mid | C5 | Bb4 | C5→Bb4 | 5 |
| 5 | High | F5 | D5 | F5→D5 | 5 |

Notes on Session 5: G5 was not achievable in the session — adapted to F5→D5
(minor third partial switch). Still valid high register crack with clear
spectrogram discontinuity.

All notes are written for Bb trumpet. Concert pitches are one whole step lower.

**Total raw takes:** 25

**Spectrogram verification:** All 25 takes verified in Audacity. All confirmed
as genuine cracks — visible f0 discontinuity on every take. 0 discarded.

**Final yield:** 25 crack pairs.

This is a 12.5× increase in crack representation over the original dataset
(2 pairs → 25 pairs). Crack events remain underrepresented relative to other
failure types, but 20 training pairs is sufficient to establish the register.

#### File naming convention

```
crack_low_{n}.wav    → low register cracks (C4→Bb3, Bb3→F3)
crack_mid_{n}.wav    → mid register cracks (G4→F4, C5→Bb4)
crack_high_{n}.wav   → high register cracks (F5→D5)
```

Register assignment in pair formation is by filename regex — not by F0 analysis.
The naming convention must be preserved if files are reorganized.

#### On crack synthesis

Crack events were investigated as a synthesis target before recording sessions
were scheduled. The investigation confirmed that synthesis is not viable:

- A crack requires a physical discontinuity in the lip seal — the f0 must jump
  instantaneously between partials
- No pitch-shifting operation produces this — it produces a glide, not a jump
- Amplitude envelope manipulation can produce abrupt note ends but not partial
  switches
- The only source for crack events is a player deliberately producing them

---

### Final Dataset Assembly

After all three tracks were formed into labeled pairs, the full dataset was
merged and shuffled:

```python
random.seed(42)
all_pairs = track_a_pairs + track_b_pairs + track_c_pairs
random.shuffle(all_pairs)
# written to pairs_expanded.json
```

**Final dataset: `pairs_expanded.json`, 4,815 pairs.**

**By label:**
```
slightly_breathy    1,581
breathy             1,581
flat                  814
sharp                 814
cracked                25
```

**Licensing:** The full expanded dataset derives from Group 1 (personal
recordings, fully unrestricted). The dataset is commercially safe. No
verify_pending files were used.

---

### Dependencies

```
OS:     Windows 11, WSL2
Python: 3.10
System: rubberband-cli (apt) — required for pyrubberband

pip:
  praat-parselmouth   — HNR measurement
  soundfile           — WAV read/write
  scipy               — noise filtering, signal processing
  numpy               — array operations
  librosa             — F0 extraction via pyin
  pyrubberband        — pitch shifting via rubberband-cli
```

---

## Section 9: RAG Layer — SPEC-RAG-v1

---

### What RAG means here

RAG stands for Retrieval-Augmented Generation. In the standard definition it
means: before the model generates a response, retrieve relevant passages from
a corpus and inject them as context. The model then generates with that context
available.

In TRUB.AI the term is used loosely. The v1 implementation is not a retrieval
system — it is a pre-authored lookup table. The "retrieval" is a dictionary
lookup by bucket pair. There is no vector search, no embedding model, no corpus
at runtime. The name reflects the intent (inject pedagogical context before
generation) rather than the mechanism.

The full retrieval architecture (Option B — dense vector search over a corpus)
is the target at corpus scale. v1 is the correct starting point given what the
corpus can support at launch.

---

### The Problem RAG Solves

After Phase 5b, the inner monologue had:
- A forced grammatical opening (PhraseConditioner prefix)
- A bridge token closing the prefix clause
- A 24-step bias-active free generation window

The free generation window produced trumpet vocabulary but not sentences. The
LoRA, trained on 22 pairs, had enough data to learn vocabulary register but not
enough grammatical variety to construct complete predicates. The [D]-window
output looked like:

```
▁end , ▁air ▁above per ▁Its ▁simultaneously , ▁air ▁below ▁center ▁air
```

Trumpet words present. Sentence structure absent.

The RAG layer fixes this by forcing a complete sentence continuation — 12
tokens that finish the grammatical clause started by the prefix. After the
forced completion, the LoRA generates continuation within the bias window.
The LoRA no longer needs to construct the sentence from scratch — it extends
an already-complete thought.

---

### Architecture Position

RAG inserts between the bridge token and the [D]-window:

```
[F] ▁Your ▁tone ▁center ▁is    ← PhraseConditioner forced prefix
[B] ▁spreading                  ← bridge token
[R] ▁slightly ▁at ▁the ▁phrase  ← RAG passage — forced, 12 tokens max
[D] ▁end ...                    ← LoRA free generation, logit-biased, 12 steps
    ...                         ← unsteered free generation
```

The [D]-window begins only after the RAG queue fully drains. The 24-step
BIAS_WINDOW_STEPS is split: first 12 steps are RAG-forced, remaining 12 are
bias-active free generation. The constant itself is unchanged.

---

### Spira's Pre-Spec Analysis: Four Options Evaluated

Before the spec was written, Spira evaluated four retrieval approaches and
ranked them by what the corpus could support at launch.

**Option A — Pre-computed lookup (9 entries):**
Write 9 passages offline, one per bucket pair. At phrase boundary, look up
by bucket pair. Zero runtime latency. No embedding model dependency.

**Option B — Dense vector retrieval (FAISS):**
Embed corpus passages offline into a FAISS index. At phrase boundary, embed
a query from the bucket pair and retrieve top-k passages semantically.
Meaningful at ≥200 passages. At 40–80 passages, semantic discrimination
over Option A is marginal.

**Option C — BM25 sparse retrieval:**
Index corpus with BM25, query with bucket expansion terms. Rejected: BM25
matches terms, not concepts. The symptom→cause→response mapping is semantic.
A passage about "air column velocity and embouchure aperture" is highly
relevant to (MED, LOW) but will not surface if the query terms are not
lexically present. BM25 cannot handle this application.

**Option D — Hybrid: pre-computed timing stubs + semantic slot fill:**
Fixed 9 stubs per bucket pair (like Option A), with an acoustic-specific
slot filled by retrieval. Correct long-term direction for large corpora.
Not viable at 40–80 passages — the slot composition problem is not solvable
cleanly at this corpus size.

**Ranking decision: Option A for launch, Option B at corpus expansion.**

The variation failure mode of Option A (every same-bucket phrase gets the
same passage) is real but bounded. At launch, the LoRA coherence gap is the
primary limitation, not passage variety. Option A's zero latency is a genuine
advantage — no hardware benchmarking required, no embedding model, no
retrieval competing with MERT on the inference device.

---

### Option B Migration Gates

Option B is the target architecture. Migration is gated on both conditions
being satisfied simultaneously — neither alone is sufficient:

**Gate 1 — Corpus size:** ≥200 usable passages, verified against register
and timing constraints. Passages from CC-BY YouTube pipeline count only after
per-video license verification is complete.

**Gate 2 — Latency benchmark on inference hardware:** Embedding forward pass
must be benchmarked on the actual deployment device — not the Modal H100, not
a generic benchmark from external sources. Procedure:
- Load candidate embedding model on deployment device
- Run 50 query embeddings of the Option B query form
- Record P50 and P95 latency
- Accept if P95 ≤ 500ms (one-third of the ~1.5s phrase duration budget)
- Reject if P95 > 500ms — evaluate a lighter model before re-benchmarking

Faber runs the benchmark and reports to Muse. Migration spec is not
commissioned until Muse confirms both gates are satisfied.

---

### The 9 Passages — Authorship and Constraints

**Hard constraints on every passage:**
- 12-token cap (enforced at authorship, not runtime — runtime truncation
  would corrupt the grammatical scaffold)
- Observational/diagnostic register: no encouragement, no prescriptions
- Continuation grammar: prefix + bridge + RAG passage must parse as a
  grammatical clause
- No second-person imperative ("Focus the air" is prescription —
  "The column loses focus" is observation)

**Authorship process:**
1. Spira drafts all 9 passages
2. Project owner reviews for pedagogical accuracy from Tonmeister background
3. Muse approves final content
4. Faber verifies token counts against loaded tokenizer before implementing

**Final approved passages:**

```python
RAG_PASSAGES: dict[tuple[str, str], str] = {
    ("LOW",  "LOW"):  "▁from a faster moving column",
    ("LOW",  "MED"):  "▁before the pitch finds center",
    ("LOW",  "HIGH"): "▁at the base, not at the lip",
    ("MED",  "LOW"):  "▁from a steadier column behind",
    ("MED",  "MED"):  "▁thinner toward the phrase end",
    ("MED",  "HIGH"): "▁but the column loses direction",
    ("HIGH", "LOW"):  "▁a narrower aperture to center",
    ("HIGH", "MED"):  "▁slightly at the phrase end",
    ("HIGH", "HIGH"): "▁centered and full through the phrase",
}
```

Note on (HIGH, HIGH): the original Spira draft was "▁centered and tone sounds
centered through the full phrase" — compound subject with verb agreement error.
Corrected to "▁centered and full through the phrase" before implementation.

**Token counts verified against loaded tokenizer:**

| Bucket | Tokens | Status |
|---|---|---|
| (LOW, LOW) | 6 | ✓ |
| (LOW, MED) | 6 | ✓ |
| (LOW, HIGH) | 9 | ✓ |
| (MED, LOW) | 8 | ✓ |
| (MED, MED) | 6 | ✓ |
| (MED, HIGH) | 6 | ✓ |
| (HIGH, LOW) | 6 | ✓ |
| (HIGH, MED) | 6 | ✓ |
| (HIGH, HIGH) | 6 | ✓ |

All 9 passages pass the 12-token cap. Maximum is 9 tokens (LOW, HIGH).

**Two tokenizer findings during verification:**

1. **Leading space token (id=260) on all passages.** Every passage begins
   with the bare space token as a SentencePiece artifact. Token counts above
   include this leading token. Decision: retain. The leading `▁` is the
   grammatical space join between the bridge token and the RAG passage — not
   an artifact to strip. Removing it would produce a join without whitespace.

2. **`▁steadier` fragments in (MED, LOW).** The word tokenizes as three
   pieces: `▁` + `stead` + `ier` (ids: 260, 9350, 2061). This is why
   (MED, LOW) has count=8 instead of 6. Decision: retain as-is. "From a
   steadier column" uses the comparative deliberately — "steadier" means the
   current column is insufficiently steady, which is the diagnostic content.
   Replacing with `▁steady` loses the comparative and weakens the observation.
   Three subword pieces decoding to a correctly spelled word is not a register
   problem.

---

### Implementation — `PhraseConditioner` Extension

The RAG queue is a third queue added to PhraseConditioner. Full implementation
detail is in Section 5. Summary of additions:

```python
class PhraseConditioner:
    RAG_MAX_TOKENS = 12     # new constant

    def __init__(self, ...):
        self._rag_queue: deque[int] = deque()   # new queue

    @property
    def is_active(self) -> bool:
        return (bool(self._queue) or
                bool(self._bridge_queue) or
                bool(self._rag_queue))   # must include rag_queue

    def next_token(self, device):
        if self._queue:        return ..., '[F]'
        if self._bridge_queue: return ..., '[B]'
        if self._rag_queue:    return ..., '[R]'   # new
        return None, ''
```

The runtime cap `[:self.RAG_MAX_TOKENS]` in `prime()` is a safety guard only.
Passages exceeding 12 tokens are a failure of the authorship process. If this
guard fires, Faber logs a warning and reports to Muse.

---

### Streaming Verification Result

SPEC-RAG-v1 streaming verification: PASS ✅

Full sequence for bucket (HIGH, MED), Phrase 1:

```
[F]▁Your [F]▁tone [F]▁center [F]▁is
[B]▁spreading
[R]▁spreading [R]▁ [R]▁slightly [R]▁at [R]▁the [R]▁phrase
[D]▁end , ▁air ▁above per ▁Its ▁simultaneously , ▁air ▁below ▁center ▁air
```

Decoded scaffold: "Your tone center is spreading slightly at the phrase end"

This is a grammatically complete, pedagogically correct observational sentence
for (HIGH, MED): the student has maintained pitch but tone focus is diffusing
toward phrase end. No prescription, no encouragement. Correct register.

**[R] display note:** The first [R] token displays as `▁spreading` — this is
the confirmed display artifact (one-step lag between forced input and displayed
prediction). All 6 RAG tokens were confirmed forced via ID-level queue drain
verification. Not a bug.

**Criteria results:**

| Criterion | Result |
|---|---|
| [R] tokens present after [B] | ✓ — 12 [R] tokens |
| All passage tokens forced | ✓ — verified by ID trace |
| [D]-window begins after RAG queue drains | ✓ |
| Full sequence grammatically coherent | ✓ — Muse assessment |
| Audio codebook stream unaffected | ✓ |
| Cross-phrase contamination | Zero |
| LaTeX in [D]-window | 0 |

---

### Corpus Development — Non-Blocking for v1

**Arban (IMSLP, public domain):**
PDF extraction via pdfplumber. Prose passages only — exercise notation
discarded. Estimated yield: 40–80 passages after filtering. Available for
Option B corpus preparation in parallel with v1 operation.

**CC-BY YouTube lesson transcripts — Phase 2:**
Per-video license verification required. The `license` field in YouTube
Data API metadata must return `creativeCommon` before any transcript enters
the corpus. Auto-generated transcripts on CC-BY videos inherit the video's
license. Manual transcripts require separate verification. This is a
corpus development workstream — does not block v1.

**Freesound pedagogical content — excluded from v1:**
Freesound licenses govern audio files. Associated text content (descriptions,
comments) has no uniform license framework. Requires per-entry legal review.
Excluded until reviewed.

**Option B migration trigger:** ≥200 verified passages AND latency benchmark
pass. Both conditions required simultaneously.

---

### Artifacts

No model checkpoints — RAG is a lookup table, not a trained component.

```
src/
    phrase_conditioner.py    ← modified — _rag_queue added
    rag_passages.py          ← new — RAG_PASSAGES dict, Muse-approved content

data/
    corpus/
        arban_passages.json        ← Faber-extracted, Muse-reviewed
        authored_passages.json     ← supplementary authored passages
```

`rag_passages.py` is a separate module from `phrase_conditioner.py`. The
conditioning table, bridge token IDs, and RAG passages are three separate
authored artifacts. Keeping them separate prevents coupling authorship changes
to implementation changes.

---

## Section 10: Diagnostics & Phase 6/7 Planning

---

### Diagnostic Philosophy

Every major component transition in TRUB.AI was followed by a streaming
diagnostic before the next phase began. The diagnostic is not a unit test —
it is a live streaming run that produces a token log and audio output, which
Muse assesses for coherence, register, and the absence of known failure modes.

This pattern was established because component-level verification (does the
conditioner produce the right embedding? does the forced token inject?) is
insufficient. The failures that mattered — LaTeX cycling, attractor words,
space token collapse — only appeared in the integrated streaming loop. Isolated
component tests would not have caught them.

---

### KL Verification — Phase 4

The first end-to-end diagnostic. Purpose: confirm that MERT conditioning
actually influences Moshi's inner monologue generation.

**Method:** Compare inner monologue token distributions under two conditions:
- Conditioned: condition_sum set from a real MERT trumpet embedding
- Unconditioned: condition_sum set to zeros

**Result: KL = 11.47**

This is a strong signal. KL near zero would indicate conditioning has no effect.
KL = 11.47 means the conditioned distribution is strongly different from the
unconditioned one. The MERT→TensorConditioner→condition_sum pipeline is doing
real work.

**What this confirmed:** The acoustic conditioning chain is live.
**What this did not confirm:** The conditioning is pointing in the right direction.
That required further streaming runs.

---

### Streaming Verification Criteria — Final Spec (§10)

These criteria apply to every streaming verification run. They are measured
over `BIAS_WINDOW_STEPS` (24 steps) — not 32. The 32-step criterion was a
spec error corrected when the BIAS_WINDOW_STEPS constant was established.

```
§10-1  LaTeX/academic tokens in [D]-window:   0
§10-2  Trumpet vocabulary tokens in [D]-window: ≥ 3
§10-3  Coherence:                              Muse assessment
§10-4  Forced prefix [F]/[B]/[R] present:     confirmed for all bucket pairs
§10-5  Audio codebook stream:                  continuous (no dropouts)

Additional checks:
  Attractor words:    absent
  Space cycling:      absent (max_run = 1)
  ▁Mr cycling:        post-window only (in-window hits = 0)
```

§10-3 (coherence) is always a Muse assessment — it cannot be automated. It
is the judgment of whether the inner monologue reads as pedagogically
appropriate for the bucket pair that triggered it.

---

### SPEC-DIAGNOSTIC-SPOKEN-v1 — The Spoken Output Diagnostic

**Purpose:** Determine whether Moshi's spoken output channel produces
teacher-appropriate speech when the inner monologue is richly scaffolded.

This was an empirical question that could not be answered by design. The
architecture supports session context injection — but whether the base model
would produce teacher-appropriate *speech* given pedagogical inner monologue
content was unknown until measured.

**What was injected:** A 47-token scaffold forced via a new [S] tag at session
start, before any phrase boundary:

```
"Student has played three phrases. First phrase: pitch low, tone low.
Second phrase: pitch low, tone medium. Third phrase: pitch medium, tone low.
Current observation: the air column loses support before the pitch stabilizes."
```

This simulates what a session state layer would inject — goal context,
phrase history, and current acoustic observation, all in the text channel.

**What was expected (optimistic case):** Moshi speaks something like:
"I can hear the air support dropping toward the end of your phrases —
let's work on keeping the column moving through the full note."

**What actually happened:**

The first five seconds of audio were non-audible — the scaffold tokens were
being forced through the inner monologue while the audio codebooks generated
incoherent output. The model had no speech primer, only dense technical context
in the diagnostic channel.

The remaining audio was described as "bigibigibigibigibigi" — the spoken output
channel entered a degenerate repetitive cycling state and had no exit mechanism.

The inner monologue token stream was functioning correctly throughout:
[S] tokens confirmed injected, [F]/[B]/[R]/[D] sequence firing at phrase
boundaries, trumpet vocabulary present, LaTeX absent.

**Diagnosis:** The inner monologue (semantic) channel is steerable. The audio
codebook (acoustic) channel is not — it runs on base Moshi behavior, and base
Moshi has no mechanism to generate teacher-appropriate speech from pedagogical
inner monologue content alone. Without alignment, it cycles.

---

### Understanding Why the Audio Cycled

This connects directly to Moshi's token architecture (covered in Section 2).

Moshi produces 17 token streams per timestep:
```
1  text codebook        — inner monologue (what we steer)
8  Moshi audio codebooks  — spoken output (what we cannot directly control)
8  user audio codebooks   — student input
```

Every intervention in the system — condition_sum, forced tokens, logit bias,
RAG passages — operates on the text codebook only. The 8 audio codebooks
run on base Moshi behavior. The base model's learned relationship between
inner monologue content and spoken audio was trained on conversational data,
not on trumpet pedagogy.

When the inner monologue contains dense technical pedagogical context with
no conversational speech primer, the audio codebooks have no pattern to
follow. They enter a degenerate state — the audio equivalent of ▁Mr cycling
in the text channel.

The Kyutai alignment paper (arXiv:2506.21463) confirms this independently:
computing DPO alignment loss over both text and audio tokens produces unstable
training. The text channel is the right alignment target. The audio codebooks
follow from the base model's learned text-to-audio mapping, which must be
shaped by alignment on the text side.

---

### What the Diagnostic Confirmed About Phase 6/7

The diagnostic answered the empirical question. The answer:

**DPO-LN alignment on the spoken output channel is not optional — it is the
primary gate for any interactive lesson capability.**

Building a session state layer, curriculum tracker, or interactive lesson
structure on top of an unaligned spoken output channel produces nothing useful.
The architecture supports these features. The base model does not yet produce
appropriate spoken output to carry them.

---

### Phase 6/7 — DPO-LN Alignment Plan

DPO-LN (Length-Normalized Direct Preference Optimization) is the alignment
method confirmed effective by the Kyutai paper. It operates on text-only token
probabilities — not audio codebooks — which is both stable and consistent with
the architecture.

**The alignment pipeline:**

```
1. Deploy current stack
        ↓
2. Collect student interactions (real or simulated)
        ↓
3. LLM judge evaluates each response on:
     - Factual accuracy (is the observation correct?)
     - Register (observational/diagnostic, not prescriptive)
     - Timing (was the response at the right moment relative to phrase boundary?)
        ↓
4. Form preference pairs:
     preferred  = correct observation, correct register, correct timing
     rejected   = cycling response, wrong register, wrong timing, stale response
        ↓
5. DPO-LN training on text-only token probabilities
        ↓
6. Verify spoken output quality post-alignment
```

**Timing as a first-class alignment signal:**

The Kyutai paper identifies timing as a first-class alignment dimension for
spoken dialogue — when a response is produced is as important as what it says.
For TRUB.AI this is especially critical:

- A response produced at the ACTIVE→TRAILING transition (student just finished
  a phrase) is contextually appropriate
- A response produced mid-phrase interrupts the student
- A response produced three phrases later is stale — it responds to playing
  the student no longer remembers

TAD already detects phrase boundaries. The alignment data must include timing
preference pairs:
```
preferred = response fires at ACTIVE→TRAILING transition
rejected  = response fires mid-phrase or more than one phrase late
```

This is architecturally supported — trublib provides the timing signal. It is
not yet collected as alignment data.

---

### Phase 6/7 — Session State Layer (After Alignment)

The interactive lesson capability — teacher remembers across phrases, adjusts
based on trajectory, sets goals — requires three layers that do not yet exist:

**Layer 1 (exists): Reactive feedback.**
Teacher hears a phrase and responds to what was heard. This is what the current
stack does. MERT encodes the acoustic content, conditioning steers the inner
monologue, the LoRA and RAG provide the scaffold.

**Layer 2 (not yet built): Session memory.**
Teacher remembers that three phrases ago the student was at LOW/LOW, now at
MED/MED. The response changes based on trajectory, not just current phrase.
Requires a session state object maintained outside Moshi and injected at
phrase boundaries:

```python
SessionState:
    current_goal: str           # e.g. "sustain lowest C, no valves"
    phrase_history: list        # [(bucket_pair, timestamp), ...]
    progression_criteria: dict  # {"LOW/LOW → MED/*": "move to next exercise"}
    session_arc: int            # phrase count in current exercise
```

This state is injected into the inner monologue text channel at each phrase
boundary, alongside the RAG passage. The injection point is already reserved —
the channel can carry multiple simultaneous inputs.

**Layer 3 (not yet built): Proactive direction.**
Teacher sets a goal. The system is no longer reactive — it directs what the
student should play next. This requires a curriculum layer: a sequence of
exercises, criteria for advancement, and language to instruct the next task.

**Why Layer 2 and 3 must wait for alignment:**

The diagnostic confirmed the spoken output channel cycles on dense inner
monologue content. Injecting session context (Layer 2) adds more content to
the channel that already produced cycling. Injecting curriculum direction
(Layer 3) is meaningless if the spoken output cannot carry it coherently.

Alignment must come first. Then verify spoken output quality. Then build
session state and curriculum on top of a verified foundation.

---

### The Kyutai Alignment Paper — Key Findings for TRUB.AI

**Paper:** "Aligning Spoken Dialogue Models from User Interactions"
arXiv:2506.21463, Kyutai/Cornell, ICML 2025.

**What it confirms:**
- Text-only probability estimation is the correct path for Moshi alignment.
  Our entire intervention stack operates on the text channel — independently
  arrived at the same conclusion.
- Inner monologue is treated as load-bearing for linguistic quality.
- DPO-LN is the stable method for variable-length spoken dialogue.

**Off-policy transfer:** Alignment data from one Moshi checkpoint can align
another checkpoint sharing the same architecture. This means interactions
collected under lora_v3 are valid training data for a DPO-LN run that aligns
lora_v4 — the alignment data does not need to be recollected from scratch
after each LoRA update.

**What does not change:**
- RAG layer is still the correct immediate next step (already implemented)
- Data boundary (21 pairs) was the binding constraint — addressed by lora_v4
- Sequencing: LoRA v4 → alignment → session state

---

### Current Production Stack — Complete Reference

| Component | Checkpoint / Module | Status |
|---|---|---|
| trublib (TAD) | PyPI: `trublib` | ✅ Production |
| TensorConditioner | Modal Volume: `/checkpoints/retrain_v13/best/tensor_conditioner.pt` as `ConditionerV12` | ✅ Production |
| LoRA | `lora_v3/best` in production streaming; `lora_v4/best` | 🔄 v4 training in progress |
| LogitBiasVector | Modal Volume: `/checkpoints/logit_bias/bias_v3.pt`, alpha=0.5 | ✅ Production |
| PhraseConditioner | `conditioner/phrase_conditioner.py` | ✅ Production |
| RAG layer | `RAG_PASSAGES` dict, inline in `conditioner/phrase_conditioner.py`, Option A | ✅ Production |
| Streaming script | `streaming/trubai_streaming_v14.py` | ✅ Current |
| Spoken output alignment | DPO-LN | ⏳ Phase 6/7 |
| Session state layer | — | ⏳ After alignment |
| Curriculum layer | — | ⏳ After session state |

---

### Open Items at Time of Writing

**Immediate:**
- lora_v4 training gate reports (Faber, every 20 epochs, Muse confirmation
  required before continuing)

**After lora_v4:**
- Streaming verification of lora_v4/best against §10 criteria
- Coherence assessment: does 4,815 pairs improve sentence-level structure
  over lora_v3's vocabulary scatter?

**Phase 6/7 prerequisites:**
- Deploy current stack for interaction collection
- Define LLM judge criteria: factual accuracy, register, timing
- Establish preference pair format including timing dimension
- DPO-LN training run on text-only token probabilities

**Corpus development (parallel, non-blocking):**
- Arban PDF extraction — 40–80 passages for Option B corpus
- CC-BY YouTube license verification pipeline
- Option B migration check: ≥200 passages AND latency benchmark on
  inference hardware

**Dataset licensing audit (non-blocking for current work):**
- Groups 6–11 (~265 files, verify_pending) need per-file license review
  before any public release. Not in current training pipeline.

---

### Non-Obvious Things to Know

A consolidated list of facts that are easy to lose between sessions and have
each cost real time at least once:

**Conditioner:**
- Always instantiate as `ConditionerV12`, not raw `TensorConditioner`
- Always inject `direction * output_scale.detach()`, not bare `direction`
  (bare direction is 37× signal reduction)
- Import from `moshi.conditioners.tensors`, not `moshi.conditioners`
- `output_scale` is `register_buffer`, not `nn.Parameter`

**LoRA:**
- Save and restore optimizer state at every gate checkpoint
- Never reinitialize AdamW on resume (resume bug, costs ~4 epochs)
- gap criterion from retrain_v2 (0.383) is not valid for diverse-trained
  conditioners — streaming verification is the only gate

**Streaming:**
- `BIAS_WINDOW_STEPS = 24` — not 32. The §10 criterion measures exactly
  this window. Do not hardcode 32 in any new script.
- Cross-phrase reset must zero `bias_steps_remaining` and call `prime()` —
  both, not just one
- `ConditionFuser` requires both `"sum"` and `"cross"` keys even when cross
  is empty

**Tokenizer:**
- `sp.encode(' ' + word)` produces id=260 as a leading artifact. The actual
  word ID is the second element.
- To verify bare space token: `sp.piece_to_id('▁')` returns 260.
  Do not use `sp.encode` for this.
- `embouchure` has no single-token form — 5 fragments. Excluded from
  W_POSITIVE_IDS.

**Phase 6/7:**
- DPO-LN, not KL regularizer. Different problems.
- Timing preference pairs are required — not optional.
- Session state layer comes after alignment is verified, not before.

---
