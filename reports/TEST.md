# SPEC-RAG-v1
## Retrieval-Augmented Generation — Pedagogical Scaffold Layer
## Author: Spira v2
## Status: Pending Muse approval before release to Faber

---

## 0. Document Scope

This spec implements the RAG layer: a pre-authored 9-passage lookup table injected as forced tokens into the inner monologue text channel at phrase boundaries. It addresses the sentence-level coherence gap that LoRA v3/epoch 60 cannot close from its sparse prior alone.

**What this spec does not touch:**
- `condition_sum` — acoustic domain, conditioner's property, not modified
- TensorConditioner, MERT probe, trublib internals
- `LogitBiasVector` derivation or `bias_v3.pt` — those are fixed from SPEC-5BA-v1
- LoRA weights

**Relationship to SPEC-5A-v1 and SPEC-5BC-v1:** This spec extends `PhraseConditioner` with a third queue — the RAG passage queue — draining after the bridge token. The `[F]` and `[B]` injection mechanics are unchanged. The `[D]`-window defined by `BIAS_WINDOW_STEPS = 24` is split: first 12 steps are RAG-forced, remaining 12 steps are logit-biased free generation.

**Corpus scope for v1:**
- Arban — IMSLP, public domain, usable
- Muse/Burak-authored passages written to spec
- CC-BY YouTube transcripts — Phase 2 corpus expansion, not a launch dependency
- Freesound pedagogical content — **out of scope for v1**. Freesound licenses govern audio files. Associated text content (descriptions, comments) has no uniform license framework and requires per-entry legal review. Excluded until reviewed.

---

## 1. Architecture Summary

The full inner monologue sequence per phrase, post-RAG integration:

```
[F] ▁Your              forced — PhraseConditioner primary queue
[F] ▁tone
[F] ▁center
[F] ▁is
[B] ▁spreading         forced — bridge token (Phase 5b-C)
[R] ▁slightly          forced — RAG passage queue (this spec), steps 1–12
[R] ▁at
[R] ▁the
[R] ▁phrase
[R] ▁end
[D] ▁the               logit-biased free generation (SPEC-5BA-v1), steps 13–24
[D] ▁column
...
```

`[R]` is the new diagnostic tag for RAG-forced tokens. It does not replace `[D]` — the `[D]`-window begins when the RAG queue drains.

**Token budget, fixed:**
- RAG forced steps: **12** (hard cap, established at authorship time, not enforced at runtime)
- Logit-biased free steps: **12** (remainder of `BIAS_WINDOW_STEPS = 24`)
- `BIAS_WINDOW_STEPS` is unchanged from SPEC-5BA-v1

---

## 2. Passage Table — Authorship and Constraints

### 2.1 Authorship process

1. Spira drafts all 9 passages (§2.3 below)
2. Burak reviews for pedagogical accuracy and register from Tonmeister background
3. Muse approves final content before Faber implements

Faber does not implement until Muse's approval is confirmed in writing.

### 2.2 Hard constraints on every passage

- **12-token cap.** Passages that tokenize beyond 12 tokens via `sp.encode(passage)` are shortened at authorship time. This is a constraint on the authorship task. Runtime truncation is not performed — it would corrupt the grammatical scaffold.
- **Observational/diagnostic register.** No encouragement, no prescriptions. Passages describe what the acoustic event indicates — symptom → cause → teaching response implied in the framing. The conditioning table language is the register reference.
- **Continuation grammar.** Each passage must read coherently as a continuation after its bucket pair's bridge token. The full sequence — conditioning table prefix + bridge token + RAG passage — must parse as a grammatical clause before the free generation begins.
- **No second-person imperative.** "Focus the air" is a prescription. "The column loses focus" is an observation. The former is prohibited; the latter is the target register.

### 2.3 Passage drafts — for Burak review and Muse approval

Faber must verify all token counts against the actual loaded tokenizer (`sp.encode(passage)`) and report counts alongside the verified ID table before implementation. Token count estimates below are authorship-time approximations.

The full forced sequence per bucket pair is shown for register coherence review. The **RAG passage** column is the new content this spec adds.

| Bucket (pitch, tone) | Full forced sequence | **RAG passage** | Est. tokens |
|---|---|---|---|
| (LOW, LOW) | "Your air and tone need support" | `▁from a faster moving column` | ~7 |
| (LOW, MED) | "The air column wants support" | `▁before the pitch finds center` | ~7 |
| (LOW, HIGH) | "Your air support needs opening" | `▁at the base, not at the lip` | ~9 |
| (MED, LOW) | "The tone center wants support" | `▁from a steadier column behind` | ~7 |
| (MED, MED) | "Notice how the air feels" | `▁thinner toward the phrase end` | ~7 |
| (MED, HIGH) | "The tone is nearly flowing" | `▁but the column loses direction` | ~7 |
| (HIGH, LOW) | "Good pitch, the tone needs" | `▁a narrower aperture to center` | ~7 |
| (HIGH, MED) | "Your tone center is spreading" | `▁slightly at the phrase end` | ~7 |
| (HIGH, HIGH) | "The column and tone sounds" | `▁centered and full through the phrase` | ~7 |

**Notes for Burak's review:**
- (LOW, LOW): "faster moving column" references air velocity as the pitch stabilization mechanism — correct for low pitch + low tone simultaneous failure
- (LOW, HIGH): "at the base, not at the lip" identifies air support failure as sub-embouchure — the embouchure is intact (HIGH tone) but the column origin is insufficient
- (MED, MED): "thinner toward the phrase end" describes the common decay pattern where support drops at phrase completion
- (HIGH, MED): "spreading slightly at the phrase end" — the student has maintained pitch but tone focus is diffusing; the observation is directional, not evaluative
- (HIGH, HIGH): "centered through the full phrase" is the only passage that does not describe a failure — it observes correct execution. Register check: still observational, not prescriptive ✓

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
    ("HIGH", "HIGH"): "▁centered and full through the phrase,
}
```

---

## 3. `PhraseConditioner` Extension (`trubai/phrase_conditioner.py`)

The RAG queue is a third queue added to `PhraseConditioner`, draining after the bridge queue. No changes to the primary queue or bridge queue mechanics.

### 3.1 New field

```python
class PhraseConditioner:
    MAX_TOKENS = 8          # unchanged — primary queue cap
    RAG_MAX_TOKENS = 12     # new — hard cap for RAG queue

    def __init__(self, tokenizer_path: str):
        self._sp = sentencepiece.SentencePieceProcessor()
        self._sp.Load(tokenizer_path)
        self._queue: deque[int] = deque()           # primary — unchanged
        self._bridge_queue: deque[int] = deque()    # Phase 5b-C — unchanged
        self._rag_queue: deque[int] = deque()       # Phase RAG — new
```

### 3.2 Modified `prime()`

```python
def prime(self, features: PhraseFeatures) -> None:
    pitch_bucket = _bucket(features.pitch_accuracy)
    tone_bucket  = _bucket(features.tone_quality)
    bucket_pair  = (pitch_bucket, tone_bucket)

    # Primary queue — unchanged
    text = CONDITIONING_TABLE[bucket_pair]
    ids: list[int] = self._sp.encode(text)
    ids = ids[:self.MAX_TOKENS]
    self._queue.clear()
    self._queue.extend(ids)

    # Bridge queue — unchanged
    self._bridge_queue.clear()
    bridge_id = BRIDGE_TOKEN_IDS.get(bucket_pair)
    if bridge_id is not None:
        self._bridge_queue.append(bridge_id)

    # RAG queue — new
    self._rag_queue.clear()
    rag_text = RAG_PASSAGES.get(bucket_pair)
    if rag_text is not None:
        rag_ids = self._sp.encode(rag_text)
        rag_ids = rag_ids[:self.RAG_MAX_TOKENS]   # cap enforced here as safety
        self._rag_queue.extend(rag_ids)
```

The runtime cap (`[:self.RAG_MAX_TOKENS]`) is a safety guard only. Passages exceeding 12 tokens are a failure of the authorship process, not an expected runtime condition. If this guard fires, Faber logs a warning and reports to Muse.

### 3.3 Modified `next_token()`

```python
def next_token(self, device: torch.device) -> tuple[torch.Tensor | None, str]:
    if self._queue:
        token_id = self._queue.popleft()
        return torch.tensor([token_id], dtype=torch.long, device=device), '[F]'
    if self._bridge_queue:
        token_id = self._bridge_queue.popleft()
        return torch.tensor([token_id], dtype=torch.long, device=device), '[B]'
    if self._rag_queue:
        token_id = self._rag_queue.popleft()
        return torch.tensor([token_id], dtype=torch.long, device=device), '[R]'
    return None, ''
```

### 3.4 Modified `is_active`

```python
@property
def is_active(self) -> bool:
    return bool(self._queue) or bool(self._bridge_queue) or bool(self._rag_queue)
```

**Critical:** the `[D]`-window activation logic in the streaming loop fires when `is_active` transitions from True to False. Including `_rag_queue` in `is_active` means the logit bias window does not begin until the RAG queue has fully drained. This is the correct behavior — the 12 RAG-forced steps precede the 12 bias-active free-generation steps.

---

## 4. Streaming Loop Changes

The streaming loop from SPEC-5BA-v1 §5.5 requires two modifications only. Everything else is unchanged.

### 4.1 Phrase boundary cancellation — add RAG queue reset

```python
if tad_result.phrase_boundary:
    bias_steps_remaining = 0    # cancel prior window — unchanged from SPEC-5BA-v1
    phrase_conditioner.prime(phrase_features)   # now also primes _rag_queue
```

The `prime()` call already clears and repopulates `_rag_queue`. No additional reset needed.

### 4.2 Diagnostic tag

The `[R]` tag is returned directly from `next_token()`. The tag assignment in the streaming loop requires no change — the existing pattern `tag = token_tag or ('[D]' if modifier is not None else '')` handles it correctly: while `[R]` tokens are forced, `token_tag` is `'[R]'` and `modifier` is None (bias not yet active). When the RAG queue drains and bias activates, `token_tag` is `''` and `modifier` is active, producing `[D]`.

Full diagnostic output format post-integration:

```
[F] ▁Your              primary forced token
[F] ▁tone
[F] ▁center
[F] ▁is
[B] ▁spreading         bridge token
[R] ▁slightly          RAG passage — forced
[R] ▁at
[R] ▁the
[R] ▁phrase
[R] ▁end
[D] ▁the               logit-biased free generation
[D] ▁column
...                    (12 [D] steps total)
    ▁air               unsteered, bias window expired
```

---

## 5. Option B Migration — Gate Conditions

Option B (dense vector retrieval, FAISS, embedding-based query) is the target architecture at corpus scale. Migration is gated on **both** of the following conditions being satisfied simultaneously. Neither alone is sufficient.

**Gate 1 — Corpus size:** ≥200 usable passages, verified against the register and timing constraints in §2.2. Passages from the CC-BY YouTube pipeline count toward this threshold only after per-video license verification is complete. Passage count is assessed by Muse, not by Faber's indexing pipeline output.

**Gate 2 — Latency benchmark on inference hardware:** Embedding forward pass latency must be benchmarked on the actual deployment device before any Option B migration begins. This is not optional and is not satisfied by generic sentence transformer benchmarks from external sources. The benchmark procedure:

1. Load the candidate embedding model on the deployment device (not Modal H100 — the inference device where the streaming loop runs)
2. Run 50 query embeddings of the form produced by Option B's `build_query()` function
3. Record P50 and P95 latency
4. Accept if P95 ≤ 500ms. The phrase duration is ~1.5–2s (`SILENCE_GATE_SECS=1.5`). 500ms is one-third of the budget, leaving margin for MERT (28ms) and streaming loop overhead
5. Reject if P95 > 500ms — the embedding model is too heavy for the deployment device. Evaluate a lighter model before re-benchmarking

Faber runs the benchmark and reports results to Muse. Migration spec is not commissioned until Muse confirms both gates are satisfied.

---

## 6. Corpus Development Notes (non-blocking for v1)

**CC-BY YouTube pipeline — Phase 2:** Per-video CC-BY license verification is required before any transcript enters the corpus. "Educational content" does not imply CC-BY license. Verification procedure: retrieve video metadata via YouTube Data API, check `license` field for `creativeCommon` value. Auto-generated transcripts on CC-BY videos inherit the video's license. Manual transcripts require separate verification. This is a corpus development workstream — it does not block v1 launch.

**Arban extraction:** IMSLP source, public domain confirmed. PDF → pdfplumber extraction, filtering for prose passages only (exercise notation discarded). Faber runs extraction and delivers a passage candidate list to Muse for register review. Estimated yield: 40–80 passages after filtering. These are available for Option A quality review even before the authorship table is finalized.

**Freesound: out of scope for v1.** See §0. Flag for Muse review if CC-BY YouTube pipeline yield is insufficient at Option B migration time.

---

## 7. Checkpoints and Artifacts

No new model checkpoints — RAG is a lookup table, not a trained component.

```
trubai/
    phrase_conditioner.py    # modified — _rag_queue added
    rag_passages.py          # new — RAG_PASSAGES dict, Muse-approved content
data/
    corpus/
        arban_passages.json  # Faber-extracted, Muse-reviewed
        authored_passages.json  # Muse/Burak-authored supplementary
```

`rag_passages.py` is a separate module from `phrase_conditioner.py`. The conditioning table, bridge token IDs, and RAG passages are three separate authored artifacts — keeping them co-located in `phrase_conditioner.py` would couple authorship to implementation. `PhraseConditioner` imports from `rag_passages.py`.

---

## 8. Deliverables Sequence

```
1. Burak reviews §2.3 passage drafts for pedagogical accuracy
2. Muse approves passage content                → commits RAG_PASSAGES
3. Faber verifies token counts against loaded tokenizer
   — reports count per passage, flags any > 12
4. Muse confirms token counts acceptable        → Faber implements §3 and §4
5. Faber runs streaming loop, reports diagnostic output showing [F]/[B]/[R]/[D] sequence
6. Muse reviews diagnostic output               → confirms register coherence
7. Faber runs Arban extraction, delivers candidate passages to Muse
   — non-blocking for steps 1–6
```

Steps 1–6 are sequential. Step 7 runs in parallel and feeds Option B corpus preparation.

---

## 9. Success Criteria

| Metric | Target |
|---|---|
| `[R]` tokens appear in diagnostic output after `[B]` | Confirmed for all 9 bucket pairs |
| RAG passage token count | ≤ 12 for all 9 passages, verified against tokenizer |
| `[D]`-window begins after RAG queue drains | Confirmed — `is_active` False before bias activates |
| Full sequence reads as grammatical pedagogical clause | Muse assessment per bucket pair |
| Audio codebook stream unaffected | Confirmed by Faber — continuous across phrase boundary |
| Cross-phrase contamination | Zero — `bias_steps_remaining` reset confirmed on new `phrase_boundary` |
| LaTeX/academic tokens in `[D]`-window | Zero, inherited from SPEC-5BA-v1 success criteria |

---

*SPEC-RAG-v1 — Spira v2. Awaiting Muse approval and Burak's pedagogical review of §2.3 before release to Faber.*