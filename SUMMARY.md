# TRUB.AI — Project Summary

---

## What It Is

TRUB.AI is a real-time AI trumpet teacher. You play, it listens, it responds —
like a real lesson. The system hears what you play, understands what is
happening musically, and tells you what it observes in a natural conversational
voice.

The goal is not a rigid exercise app that grades your performance. The goal is
a conversation. A teacher listens to a phrase, responds to what they heard, you
play again, the teacher listens again. That loop is what a real lesson is.

---

## The Core Design Decision

The obvious way to build a voice AI is: listen → transcribe to text → generate
text response → speak. Almost every voice assistant works this way.

This does not work for a music teacher.

When a student plays a note and it gets transcribed as "C", everything musically
important has been thrown away. Was the note flat or sharp? Was the tone breathy?
Did it crack at the attack? Was the air support consistent? None of this survives
transcription.

TRUB.AI does not transcribe the musical content. The trumpet audio is encoded
directly into the AI's understanding — pitch accuracy, tone quality, breathiness,
crack events are understood acoustically, not as words. The system responds to
what it hears, not to a label.

---

## How It Works

There are seven components that work together. Each one has a specific job.

**trublib** listens to the audio stream continuously and detects phrase
boundaries. It knows when you are playing, when you have just finished a phrase,
and when the room is silent. Every other component waits for trublib's signal
before doing anything.

**MERT** is a music understanding model. When trublib detects that a phrase has
just ended, MERT encodes the audio from that phrase into a compact mathematical
representation — a 768-number vector that captures the acoustic character of
what you just played.

**TensorConditioner** translates the MERT representation into a format that
Moshi (the conversation engine) can use. Think of it as a converter that bridges
two different languages. Getting this converter right took 13 training attempts —
the detailed history is in the technical log.

**Moshi** is the conversational AI at the center of everything. It listens and
speaks simultaneously — like a real conversation, not a walkie-talkie. Moshi
has an "inner monologue" channel that functions like a thought stream. This is
where all of the teaching context gets written — and it influences what Moshi
says out loud.

**PhraseConditioner** analyzes each phrase for two things: how accurate was the
pitch (low / medium / high), and how clean was the tone (low / medium / high).
From these two measurements it selects an opening sentence fragment and forces
it into Moshi's inner monologue. For example: pitch is high but tone is
spreading → forces "Your tone center is spreading" as the start of the thought.

**LogitBiasVector** is a set of guardrails applied to the inner monologue during
a 24-step window after each phrase. It suppresses words that should never appear
in a trumpet lesson (LaTeX symbols, academic jargon, random attractor words) and
gives a small boost to trumpet vocabulary (air, column, tone, pitch, aperture).

**RAG layer** completes the forced sentence. After the opening fragment and the
guardrails are active, the RAG layer forces 12 more tokens that finish the
thought into a grammatically complete observational sentence. The result:
"Your tone center is spreading slightly at the phrase end." After this scaffold,
the LoRA takes over and generates continuation.

**LoRA** is a small set of additional weights trained on top of Moshi that teach
it trumpet pedagogy vocabulary and register. It was trained on 4,815 labeled
examples of (audio, inner monologue sentence) pairs — recordings of trumpet
playing paired with one-sentence observational descriptions of what the playing
indicates.

---

## What the System Sounds Like Inside

For every phrase you play, the inner monologue sequence looks like this:

```
"Your tone center is" ← forced opening (9 tokens)
"spreading"           ← bridge token
"slightly at the      ← RAG scaffold (12 forced tokens)
 phrase end"
[free generation]     ← LoRA continues with trumpet vocabulary (12 steps)
[unsteered]           ← base model runs freely after bias window
```

Decoded: "Your tone center is spreading slightly at the phrase end."

That is a complete, grammatically correct, pedagogically accurate observation
for a phrase where pitch was stable but tone focus was diffusing at the end.
No prescription, no encouragement — just a diagnosis of what happened.

---

## The Dataset

The LoRA was trained on 4,815 labeled pairs assembled from three sources:

**Synthesized breathiness (3,162 pairs):** Clean recordings with shaped noise
mixed in at controlled levels to produce slightly breathy and breathy variants.

**Pitch-shifted variants (1,628 pairs):** Sustained-tone recordings shifted by
±45 and ±60 cents to simulate flat and sharp playing.

**Crack recordings (25 pairs):** Manually recorded embouchure crack events —
the only failure type that cannot be synthesized. Burak recorded these
deliberately across low, mid, and high register.

---

## What Works and What Does Not Yet

**Working:**
The inner monologue side is fully operational. For every phrase, the system
produces a grammatically complete, trumpet-vocabulary-rich observational sentence
in the inner monologue. No LaTeX, no attractor words, no cycling within the
active window.

**Not yet working:**
The spoken output — what Moshi actually says out loud — is not aligned. A
diagnostic run where rich pedagogical context was injected into the inner
monologue produced incoherent cycling audio rather than teacher speech. The
inner monologue channel is steerable. The spoken output channel runs on base
Moshi behavior and needs alignment training before it will produce coherent
teacher speech.

**The alignment work (Phase 6/7):** The plan is to collect real student
interactions, judge each response for accuracy, register, and timing, form
preference pairs (good response vs. bad response), and train Moshi to prefer
the good ones. This technique is called DPO-LN and was confirmed effective
by a Kyutai research paper published at ICML 2025.

---

## The Longer-Term Goal

The interactive lesson you would expect from a real teacher — where the teacher
remembers what you have been working on across phrases, adjusts feedback based
on your trajectory, and sets goals for what to practice next — is the target.

The architecture supports it. The session memory and curriculum layers are not
yet built, but the injection point for them (Moshi's inner monologue text
channel) is already in place. These features come after the spoken output
channel is aligned and verified.

The ten-to-fifteen year horizon is a full AI trumpet teacher that a beginner
can practice with between lessons — one that responds to the actual sound of
playing, remembers where you were last session, and adjusts what it works on
as you improve.

---

## Current Status

| Component | Status |
|---|---|
| trublib — phrase detection | ✅ Complete, on PyPI |
| TensorConditioner — acoustic bridge | ✅ Production |
| LoRA v4 — vocabulary training | 🔄 Training in progress |
| LogitBiasVector — guardrails | ✅ Production |
| PhraseConditioner — phrase analysis | ✅ Production |
| RAG layer — sentence scaffold | ✅ Production |
| Spoken output alignment (DPO-LN) | ⏳ Phase 6/7 |
| Session memory and curriculum | ⏳ After alignment |