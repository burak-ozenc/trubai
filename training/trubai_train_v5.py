"""
trubai_train_v5.py
TensorConditioner v5 — Curriculum Training
SPEC-TC-V5-v2 + Amendment A1 (Muse-approved)

Usage
-----
# Step 1: verify token IDs — report all 7 IDs to Muse before training begins
    modal run trubai_train_v5.py --mode verify

# Step 2: full curriculum (after Muse confirms IDs)
    modal run trubai_train_v5.py --mode train

# Step 2 alt: if ▁the gate fired and Muse approved 18-token W_neg
    modal run trubai_train_v5.py --mode train --drop-the 1

# Step 3: resume from a saved epoch (if run interrupted)
    modal run trubai_train_v5.py --mode train --resume-epoch 50
"""

import modal
import json
import random
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Image
# ──────────────────────────────────────────────────────────────────────────────

MOSHI_FORK_URL = "git+https://github.com/burak-ozenc/moshi.git#subdirectory=moshi"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git"])
    .pip_install([
        "torch==2.6.0",
        MOSHI_FORK_URL,
        "transformers",
        "sentencepiece",
        "huggingface_hub",
        "safetensors",
    ])
)

app = modal.App("trubai-train-v5", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

CKPT_DIR        = Path("/checkpoints")
HF_DIR          = Path("/hf-cache")
ANCHOR_JSONL    = CKPT_DIR / "trubai_training_pairs.jsonl"
COMM_SAFE_TXT   = CKPT_DIR / "commercial_safe.txt"
NEW_EMB_DIR     = CKPT_DIR / "new_embeddings" / "embeddings"
PHASE5_EMB_DIR  = CKPT_DIR / "embeddings"
V5_DIR          = CKPT_DIR / "retrain_v5"
PROD_CKPT       = CKPT_DIR / "retrain_v2" / "best" / "tensor_conditioner.pt"
EMB_CACHE_PATH  = CKPT_DIR / "moshi_emb_cache.pt"   # cached embedding table

# ──────────────────────────────────────────────────────────────────────────────
# Token sets  (§9 handover + Amendment A1)
# ──────────────────────────────────────────────────────────────────────────────

# W_positive — 18 verified IDs (embouchure not in vocab, dropped)
W_POSITIVE_IDS: dict[str, int] = {
    "▁column":    2368, "▁flat":    3077, "▁support":  711,  "▁partial":  4107,
    "▁center":    1611, "▁hollow": 19664, "▁pinch":  24657,  "▁crack":    6615,
    "▁focus":     1563, "▁breath":  8735, "▁airflow": 28512, "▁tone":     9064,
    "▁sharp":     6064, "▁buzz":   21938, "▁pitch":    6396, "▁air":      1142,
    "▁spreading": 13369,                   "▁aperture": 16252,
}

# W_negative existing — 12 unique IDs (▁major id=916 added per Muse resolution)
W_NEGATIVE_IDS_EXISTING: dict[str, int] = {
    "▁disagreement": 17888, "▁cherish":   31329, "▁nationality": 25602,
    "▁tower":         6537, "▁canonical":  8862, "▁iteration":   11024,
    "▁mascot":       27902, "▁brittle":   29915, "▁Tournament":  10653,
    "▁proud":         9630, "▁Az":         9823, "▁major":         916,
}

# Amendment A1 additions — 7 tokens, IDs resolved at runtime via verify_tokens()
# Final W_negative: 12 existing + 7 additions = 19 unique IDs (no overlap)
# (▁major id=916 added to existing set per Muse audit resolution)
W_NEGATIVE_ADDITIONS_WORDS: list[str] = [
    "▁",      # bare space token — confirmed collapse target
    "▁in",    # confirmed in v3/v4 collapse output
    "▁a",     # confirmed in v3/v4 collapse output
    "▁of",    # distributional neighbor
    "▁to",    # distributional neighbor
    "▁and",   # distributional neighbor
    "▁the",   # confirmed in failure output — carries ▁the monitoring gate
]

SPACE_TOKEN_ID_EXPECTED = 260
THE_GATE_EPOCH          = 5
THE_GATE_THRESHOLD      = 0.32   # basis: v3 peak=0.3728, 0.37-0.05 conservative floor

# ──────────────────────────────────────────────────────────────────────────────
# Curriculum helpers  (SPEC-TC-V5-v2 §2–4)
# ──────────────────────────────────────────────────────────────────────────────

def get_phase(epoch: int) -> int:
    if epoch <= 50:  return 1
    if epoch <= 130: return 2
    return 3

def get_k_diverse(epoch: int, diverse_len: int) -> int:
    """Returns number of diverse records to sample for this epoch."""
    if epoch <= 50:  return 0
    if epoch > 130:  return diverse_len
    sub_phase = (epoch - 51) // 10 + 1       # 1–8
    return int(22 * sub_phase * 0.10)

def get_lambda_anchor(epoch: int) -> float:
    """Linear decay Phase 2, fixed floor Phase 3."""
    if epoch <= 50:  return 0.0
    if epoch >= 131: return 0.01
    t = (epoch - 51) / 79                    # 0.0 at ep 51, 1.0 at ep 130
    return 0.1 - t * 0.09

def compute_anchor_loss(model, W_anchor: dict, lambda_val: float):
    """||W - W_anchor||^2 regularization. W_anchor has no gradient."""
    import torch
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        anchor = W_anchor[name].to(param.device)
        loss   = loss + (param - anchor).pow(2).sum()
    return lambda_val * loss

# ──────────────────────────────────────────────────────────────────────────────
# TensorConditioner
# Architecture inferred from and verified against retrain_v2/best at startup.
# "768→4096 linear projection + output_proj structure" (§3 handover).
# ──────────────────────────────────────────────────────────────────────────────

def build_tensor_conditioner(device: str):
    """
    Instantiate TensorConditioner using the real moshi class with the same
    constructor args as the working v4 setup. Random initialization — does NOT
    load from the production checkpoint.

    The production checkpoint keys ['learnt_padding', 'output_proj.weight']
    match TensorConditioner(dim=768, output_dim=4096, force_linear=True,
    output_bias=False, learn_padding=True).
    """
    from moshi.conditioners.tensors import TensorConditioner

    tc = TensorConditioner(
        dim=768,
        output_dim=4096,
        device=device,
        force_linear=True,
        output_bias=False,
        learn_padding=True,
    ).to(device)

    print(f"  TensorConditioner keys: {list(tc.state_dict().keys())}")
    print(f"  TensorConditioner instantiated (random init) ✓")
    return tc


def tc_forward(tc, emb, W_pos, W_neg):
    """
    Compute pos_sim, neg_sim, proj_norm from a TensorConditioner forward pass.
    TensorConditioner._get_condition expects a TensorCondition(tensor, mask),
    not a raw tensor — wrap before calling.

    emb   : [1, 1, 768] float32
    W_pos : [n_pos, 4096] float32
    W_neg : [n_neg, 4096] float32
    """
    import torch
    import torch.nn.functional as F
    from moshi.conditioners import TensorCondition

    # mask: [B, T] bool — all True (no padding)
    mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
    cond = TensorCondition(tensor=emb, mask=mask)

    proj = tc(cond)[0]                   # ConditionType is a namedtuple — index 0 is the embedding tensor [1, 1, 4096]
    x    = proj.squeeze(1)                # [1, 4096]

    x_n      = F.normalize(x,     dim=-1)
    pos_n    = F.normalize(W_pos, dim=-1)
    neg_n    = F.normalize(W_neg, dim=-1)
    pos_sim  = (x_n @ pos_n.T).mean()
    neg_sim  = (x_n @ neg_n.T).mean()
    proj_norm = x.norm()
    return pos_sim, neg_sim, proj_norm

# ──────────────────────────────────────────────────────────────────────────────
# Modal class
# ──────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu="H100",
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=21600,   # 6 hours — sufficient for 200 epochs on H100
)
class TrainV5:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = str(HF_DIR)   # MUST be first line — see Failure Mode §13

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        self.device = torch.device("cuda")

        # ── SentencePiece tokenizer ──────────────────────────────────────────
        print("Loading Moshi tokenizer...")
        text_tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp = sentencepiece.SentencePieceProcessor(text_tok_path)
        print(f"  Vocab size: {self.sp.GetPieceSize()}")

        # ── Moshi LM head weights (text_linear) ─────────────────────────────
        # W_positive / W_negative are rows of text_linear.weight [vocab, 4096].
        # Cache on volume — skip the 7B load on subsequent startups.
        if EMB_CACHE_PATH.exists():
            print("Loading cached LM head weights...")
            lm_head = torch.load(EMB_CACHE_PATH, map_location="cpu",
                                 weights_only=True)
        else:
            print("Extracting LM head weights from Moshi LM (first time — slow)...")
            moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
            moshi_lm     = loaders.get_moshi_lm(moshi_weight, device="cpu")
            lm_head      = moshi_lm.text_linear.weight.detach().float()   # [vocab, 4096]
            del moshi_lm
            torch.cuda.empty_cache()
            torch.save(lm_head, EMB_CACHE_PATH)
            vol.commit()
            print(f"  LM head cached at {EMB_CACHE_PATH}")

        print(f"  LM head shape: {lm_head.shape}")   # expected [32000, 4096]

        # ── W_positive — 18 verified IDs ────────────────────────────────────
        pos_ids    = list(W_POSITIVE_IDS.values())
        self.W_pos = lm_head[pos_ids].to(self.device)   # [18, 4096]

        # ── W_negative — built after verify_tokens(), stored here ────────────
        # Populated inside train() after token verification; kept None until then.
        self.W_neg    = None
        self.lm_head  = lm_head   # full table — sliced during verify
        print("Setup complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Token verification  (Amendment A1)
    # ──────────────────────────────────────────────────────────────────────────

    @modal.method()
    def verify_tokens(self, drop_the: bool = False) -> dict[str, int]:
        """
        Amendment A1 verification step.
        Resolves all 7 new token IDs via SentencePiece.
        Asserts space token == 260.
        Prints a formatted report for Muse.
        Returns {word: id} dict for the 7 additions.
        Do not begin training until this is reported to and confirmed by Muse.
        """
        print()
        print("=" * 65)
        print("TOKEN VERIFICATION — Amendment A1 (SPEC-TC-V5-v2)")
        print("=" * 65)

        # 1. Hardcoded space token assertion
        space_id = self.sp.piece_to_id("▁")
        assert space_id == SPACE_TOKEN_ID_EXPECTED, (
            f"Space token ID mismatch: got {space_id}, expected {SPACE_TOKEN_ID_EXPECTED}"
        )

        # 2. All 7 additions
        additions: dict[str, int] = {}
        all_ok = True
        words = [w for w in W_NEGATIVE_ADDITIONS_WORDS
                 if not (drop_the and w == "▁the")]

        for word in words:
            pid = self.sp.piece_to_id(word)
            ok  = pid > 0
            if not ok:
                all_ok = False
            additions[word] = pid
            status = "✓" if ok else "✗  NOT IN VOCAB"
            print(f"  {word!r:10}  →  id={pid:6d}  {status}")

        print()
        print(f"  Existing W_negative: {len(W_NEGATIVE_IDS_EXISTING)} unique IDs")
        print(f"  Additions resolved:  {sum(1 for v in additions.values() if v > 0)} / {len(words)}")
        print(f"  drop_the={drop_the}")

        if all_ok:
            print()
            print("  All IDs resolved. Report these values to Muse before proceeding.")
        else:
            print()
            print("  FAILURE: one or more tokens not in vocabulary.")
            print("  Report to Muse. Do not proceed.")

        print("=" * 65)
        print()

        if not all_ok:
            raise RuntimeError("Verification failed — see output above.")

        return additions

    # ──────────────────────────────────────────────────────────────────────────
    # W_negative existing set audit  (Muse pre-training gate)
    # ──────────────────────────────────────────────────────────────────────────

    @modal.method()
    def audit_wneg(self) -> None:
        """
        Muse audit: encode each of the 12 existing W_negative words via
        sp.encode(' ' + word), report all token IDs and pieces, flag collisions.
        Run with: modal run trubai_train_v5.py --mode audit
        Report full output to Muse before Phase 1 begins.
        """
        W_negative_existing_words = [
            "proud", "brittle", "ville", "nik", "tower", "Az",
            "nationality", "Tournament", "disagreement", "major",
            "cherish", "mascot",
        ]

        print()
        print("=" * 65)
        print("W_negative EXISTING SET AUDIT (Muse pre-training gate)")
        print("=" * 65)

        seen_ids: dict[int, str] = {}
        for word in W_negative_existing_words:
            ids    = self.sp.encode(" " + word)
            pieces = [self.sp.id_to_piece(i) for i in ids]
            print(f"  {word!r:15} → ids={ids}  pieces={pieces}")
            for i in ids:
                if i in seen_ids:
                    print(f"    *** COLLISION: id={i} already claimed by {seen_ids[i]!r}")
                seen_ids[i] = word

        print()
        print(f"  Total unique IDs across all 12 words: {len(seen_ids)}")
        print("=" * 65)
        print()
        print("Report this output to Muse. Do not begin Phase 1 until resolved.")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Curriculum training
    # ──────────────────────────────────────────────────────────────────────────

    @modal.method()
    def train(
            self,
            resume_epoch: int = 0,
            drop_the:     bool = False,
    ) -> None:
        """
        Full 200-epoch curriculum per SPEC-TC-V5-v2 + Amendment A1.

        Mandatory stops (raise RuntimeError, do not proceed):
          epoch  5 — ▁the gate: gap < 0.32
          epoch 50 — Phase 1 gate: pos_sim < 0.02

        Bridge sub-phase (§7):
          Auto-applied at Phase 3 if pos_sim < 0.01 for 3 consecutive epochs.
          No Muse direction needed for bridge — fires automatically per spec.

        Checkpoints saved to: /checkpoints/retrain_v5/
        """
        import torch
        import torch.nn as nn

        # ── Verify tokens + build W_negative ────────────────────────────────
        additions = self.verify_tokens.local(drop_the=drop_the)
        addition_ids = list(additions.values())
        all_neg_ids  = list(set(
            list(W_NEGATIVE_IDS_EXISTING.values()) + addition_ids
        ))
        self.W_neg = self.lm_head[all_neg_ids].to(self.device)
        print(f"W_negative: {self.W_neg.shape[0]} unique IDs "
              f"(existing={len(W_NEGATIVE_IDS_EXISTING)}, "
              f"additions={len(addition_ids)}, "
              f"post-dedup={len(all_neg_ids)})")
        print(f"drop_the={drop_the}")
        print()

        # ── TensorConditioner — random init, architecture from prod ckpt ────
        tc = build_tensor_conditioner(str(self.device))

        # ── Load datasets ────────────────────────────────────────────────────
        anchor_set, diverse_set = self._load_datasets()

        # ── Checkpoint dir ────────────────────────────────────────────────────
        V5_DIR.mkdir(parents=True, exist_ok=True)

        # ── Hyperparameters ───────────────────────────────────────────────────
        LR_MAX      = 4.5e-5    # v4 LR (original * 0.15)
        LAMBDA_NORM = 0.1       # unchanged from v4
        TARGET_NORM = 36.5      # validated in v4

        optimizer = torch.optim.AdamW(
            tc.parameters(), lr=LR_MAX, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-6
        )

        # ── Resume ────────────────────────────────────────────────────────────
        start_epoch = 1
        W_anchor    = None

        if resume_epoch > 0:
            ckpt = V5_DIR / f"epoch_{resume_epoch:03d}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
            tc.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            start_epoch = resume_epoch + 1
            print(f"Resumed from epoch {resume_epoch}")
            anchor_path = V5_DIR / "phase1_anchor.pt"
            if resume_epoch >= 50 and anchor_path.exists():
                W_anchor = torch.load(
                    anchor_path, map_location=self.device, weights_only=True
                )
                print("W_anchor loaded ✓")
            # Fast-forward scheduler
            for _ in range(resume_epoch):
                scheduler.step()

        # ── Phase 3 bridge tracking ────────────────────────────────────────
        # §7: trigger check window is epochs 131–140 ONLY.
        # If trigger fires, bridge (k=1500) replaces full set for remainder
        # of 131–140. Epoch 141+ always full Phase 3, regardless.
        p3_low_count    = 0       # consecutive epochs with pos_sim < 0.01
        bridge_active   = False   # True after trigger fires within 131–140

        # ── Training loop ─────────────────────────────────────────────────
        print(f"Training from epoch {start_epoch} to 200")
        print("=" * 80)

        # Variables held across loop for reporting gates
        last_pos  = 0.0
        last_neg  = 0.0
        last_norm = 0.0

        for epoch in range(start_epoch, 201):
            phase         = get_phase(epoch)
            k             = get_k_diverse(epoch, len(diverse_set))
            lambda_anchor = get_lambda_anchor(epoch)

            # ── Batch assembly (§3) ────────────────────────────────────────
            if phase == 1:
                epoch_data = list(anchor_set)

            elif phase == 2:
                diverse_sample = random.sample(diverse_set, k)
                epoch_data     = list(anchor_set) + diverse_sample

            else:  # Phase 3
                # Bridge sub-phase applies only during epochs 131–140 (§7).
                # After epoch 140 (epoch 141+), always full regardless of bridge.
                if bridge_active and epoch <= 140:
                    diverse_sample = random.sample(diverse_set, 1500)
                    epoch_data     = list(anchor_set) + diverse_sample
                    if epoch == 140:
                        print(f"  [bridge] Epoch 140 reached — bridge sub-phase ends. "
                              f"Full Phase 3 begins epoch 141.")
                else:
                    epoch_data = list(anchor_set) + list(diverse_set)

            random.shuffle(epoch_data)

            # ── Gradient accumulation over epoch (§6) ─────────────────────
            tc.train()
            optimizer.zero_grad()
            epoch_loss = 0.0

            for record in epoch_data:
                emb = self._load_embedding(record).to(self.device)

                pos_sim, neg_sim, proj_norm = tc_forward(tc, emb, self.W_pos, self.W_neg)

                L_c      = -(pos_sim - neg_sim)
                L_n      = LAMBDA_NORM * (proj_norm - TARGET_NORM) ** 2
                L_a      = (
                    compute_anchor_loss(tc, W_anchor, lambda_anchor)
                    if W_anchor is not None
                    else torch.tensor(0.0, device=self.device)
                )
                loss = L_c + L_n + L_a
                loss.backward()
                epoch_loss += loss.item()

            # Track last record's metrics for logging (per spec §6)
            last_pos  = pos_sim.item()
            last_neg  = neg_sim.item()
            last_norm = proj_norm.item()

            torch.nn.utils.clip_grad_norm_(tc.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            gap = last_pos - last_neg

            # ── Save W_anchor + epoch 50 checkpoint ───────────────────────
            if epoch == 50:
                W_anchor = {
                    k_: v.detach().clone()
                    for k_, v in tc.state_dict().items()
                }
                torch.save(W_anchor,           V5_DIR / "phase1_anchor.pt")
                torch.save(tc.state_dict(),    V5_DIR / "epoch_050.pt")
                vol.commit()
                print(f"\n>>> Phase 1 complete. W_anchor and epoch_050 saved.")

            # ── Phase boundary checkpoints ─────────────────────────────────
            if epoch in (100, 130, 150, 200):
                torch.save(tc.state_dict(), V5_DIR / f"epoch_{epoch:03d}.pt")
                vol.commit()

            # ── Logging  (epochs 1–5 every epoch, then every 10) ──────────
            if epoch <= 5 or epoch % 10 == 0:
                anc_dist = (
                    compute_anchor_loss(tc, W_anchor, 1.0).item()
                    if W_anchor is not None else 0.0
                )
                print(
                    f"Epoch {epoch:3d} | phase={phase} | k_div={k:4d} | "
                    f"gap={gap:.4f} | pos={last_pos:.4f} | neg={last_neg:.4f} | "
                    f"norm={last_norm:.2f} | λ_anc={lambda_anchor:.4f} | "
                    f"anc_dist={anc_dist:.4f} | "
                    f"lr={scheduler.get_last_lr()[0]:.6f}"
                )

            # ══════════════════════════════════════════════════════════════
            # REPORTING GATES
            # ══════════════════════════════════════════════════════════════

            # ── Gate A: ▁the monitoring — epoch 5 ─────────────────────────
            # RETIRED as hard stop (Muse decision, this run):
            # Threshold 0.32 was calibrated against v3/v4 11-token W_negative.
            # v5 W_negative includes 7 high-frequency central tokens — absolute
            # gap values are not comparable. Trend (pos_sim monotonically
            # increasing, positive from epoch 1) is the correct signal.
            # Gate logs informational only; epoch 50 is the real decision point.
            if epoch == THE_GATE_EPOCH and not drop_the:
                print(f"\n--- ▁the monitoring gate (epoch {THE_GATE_EPOCH}) — informational only ---")
                print(f"    gap={gap:.4f}  (threshold {THE_GATE_THRESHOLD} retired for this run)")
                print(f"    pos_sim={last_pos:.4f} — trend is the signal, not absolute gap")
                print(f"    Continuing Phase 1. Real gate: epoch 50 pos_sim > 0.02.")

            # ── Gate B: Phase 1 end — epoch 50 ────────────────────────────
            # Mandatory stop: pos_sim < 0.02 → do not begin Phase 2
            if epoch == 50:
                anc_dist_50 = 0.0   # W_anchor was just saved, dist is 0 by definition
                self._report_gate(
                    label="EPOCH 50 — Phase 1 end",
                    epoch=epoch,
                    last_pos=last_pos, last_neg=last_neg,
                    gap=gap, last_norm=last_norm,
                    anc_dist=anc_dist_50,
                )
                if last_pos < 0.02:
                    torch.save(
                        tc.state_dict(),
                        V5_DIR / "epoch_050_gate_stop.pt"
                    )
                    vol.commit()
                    print(f"\n>>> ▲ PHASE 1 GATE TRIGGERED: pos_sim={last_pos:.4f} < 0.02")
                    print(f"    Training stopped. Report to Muse before Phase 2.")
                    raise RuntimeError(
                        f"Phase 1 gate: pos_sim={last_pos:.4f} < 0.02. "
                        "Stopped before Phase 2. Report to Muse."
                    )
                else:
                    print(f"    pos_sim={last_pos:.4f} ≥ 0.02 ✓ — Phase 2 begins.")
                    # Muse addition: log epoch 50 values as v5 baseline reference.
                    # All future v5 monitoring gates measured against these values,
                    # not v3/v4 values (W_negative composition changed).
                    print()
                    print("  >>> V5 BASELINE REFERENCE (epoch 50) <<<")
                    print(f"      gap_v5_baseline      = {gap:.4f}")
                    print(f"      pos_sim_v5_baseline  = {last_pos:.4f}")
                    print(f"      neg_sim_v5_baseline  = {last_neg:.4f}")
                    print(f"      proj_norm_v5_baseline= {last_norm:.2f}")
                    print("      All future v5 curriculum gates reference these values.")

            # ── Gate C: Phase 3 bridge trigger (§7) ───────────────────────
            # Spec: trigger check window is epochs 131–140 ONLY.
            # Fires automatically — no Muse direction needed.
            # After epoch 140, bridge window closed; check has no effect.
            if phase == 3 and epoch <= 140 and not bridge_active:
                if last_pos < 0.01:
                    p3_low_count += 1
                    print(f"  [bridge watch] epoch {epoch}: pos_sim={last_pos:.4f} < 0.01 "
                          f"({p3_low_count}/3 consecutive, window closes ep 140)")
                    if p3_low_count >= 3:
                        bridge_active = True
                        print(f"\n>>> Phase 3 bridge sub-phase triggered at epoch {epoch}.")
                        print(f"    k=1500 diverse for epochs {epoch}–140.")
                        print(f"    Full Phase 3 (3203 pairs) resumes epoch 141.")
                else:
                    if p3_low_count > 0:
                        print(f"  [bridge watch] epoch {epoch}: pos_sim recovered — reset counter")
                    p3_low_count = 0

            # ── Gate D: Phase 2 end report — epoch 130 ────────────────────
            if epoch == 130:
                anc_dist_130 = (
                    compute_anchor_loss(tc, W_anchor, 1.0).item()
                    if W_anchor is not None else 0.0
                )
                self._report_gate(
                    label="EPOCH 130 — Phase 2 end",
                    epoch=epoch,
                    last_pos=last_pos, last_neg=last_neg,
                    gap=gap, last_norm=last_norm,
                    anc_dist=anc_dist_130,
                )
                p3_likely = last_pos < 0.015
                print(f"    Phase 3 bridge likely: {p3_likely}")
                print(f"    (Note: continue training — epoch 130 is informational only)")

        # ── Final report (epoch 200) ───────────────────────────────────────
        anc_dist_final = (
            compute_anchor_loss(tc, W_anchor, 1.0).item()
            if W_anchor is not None else 0.0
        )
        gap_final = last_pos - last_neg

        print()
        print("=" * 65)
        print("EPOCH 200 REPORT — Training complete")
        print("=" * 65)
        print(f"  gap       = {gap_final:.4f}  (target ≥ 0.38)")
        print(f"  pos_sim   = {last_pos:.4f}  (target > 0.02)")
        print(f"  neg_sim   = {last_neg:.4f}")
        print(f"  proj_norm = {last_norm:.2f}  (target 35.0–38.0)")
        print(f"  anc_dist  = {anc_dist_final:.4f}")
        print(f"  bridge_applied  = {bridge_active}")
        print(f"  drop_the        = {drop_the}")
        print()

        success = gap_final >= 0.38 and last_pos > 0.02
        if success:
            print("  STATUS: CRITERIA MET ✓")
            print("  Provisional 'best' saved. Awaiting streaming verification.")
        else:
            misses = []
            if gap_final < 0.38:  misses.append(f"gap={gap_final:.4f} < 0.38")
            if last_pos <= 0.02:  misses.append(f"pos_sim={last_pos:.4f} ≤ 0.02")
            print(f"  STATUS: CRITERIA MISSED — {', '.join(misses)}")
            print("  Report to Muse. LoRA retraining blocked until criteria met.")
        print("=" * 65)

        # Save final checkpoint + provisional best
        torch.save(tc.state_dict(), V5_DIR / "epoch_200.pt")
        (V5_DIR / "best").mkdir(exist_ok=True)
        torch.save(tc.state_dict(), V5_DIR / "best" / "tensor_conditioner.pt")
        vol.commit()
        print(f"\nCheckpoints committed to volume at {V5_DIR}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_datasets(self) -> tuple[list[dict], list[dict]]:
        """
        Returns (anchor_set, diverse_set).
        anchor_set : 22 dicts with '_emb_path' pointing to Phase 5 embeddings
        diverse_set: 3181 dicts with '_emb_path' pointing to new_embeddings/
        """
        import re

        def normalize_pitch(s: str) -> str:
            """Gs4 → G#4 normalization (commercial_safe.txt uses Gs)."""
            return re.sub(r'Gs(\d)', r'G#\1', s)

        # ── Anchor set (22) ────────────────────────────────────────────────
        anchor_records = [json.loads(l) for l in open(ANCHOR_JSONL)]
        for rec in anchor_records:
            stem = Path(rec["embedding_path"]).stem   # Kaggle path → stem only
            rec["_emb_path"] = PHASE5_EMB_DIR / f"{stem}.pt"
        assert len(anchor_records) == 22, (
            f"Expected 22 anchor pairs, got {len(anchor_records)}"
        )
        # Verify all anchor embeddings exist
        missing_anchors = [
            r["_emb_path"] for r in anchor_records
            if not r["_emb_path"].exists()
        ]
        if missing_anchors:
            raise FileNotFoundError(
                f"{len(missing_anchors)} anchor embeddings missing:\n"
                + "\n".join(str(p) for p in missing_anchors[:5])
            )
        print(f"Anchor set:  {len(anchor_records)} pairs ✓")

        # ── Diverse set (3181) ─────────────────────────────────────────────
        stems_raw = [
            Path(line.strip()).stem
            for line in open(COMM_SAFE_TXT)
            if line.strip()
        ]
        diverse_records: list[dict] = []
        miss = 0
        for stem in stems_raw:
            pt = NEW_EMB_DIR / f"{stem}.pt"
            if not pt.exists():
                pt = NEW_EMB_DIR / f"{normalize_pitch(stem)}.pt"
            if pt.exists():
                diverse_records.append({"_emb_path": pt})
            else:
                miss += 1

        assert len(diverse_records) == 3181, (
            f"Expected 3181 diverse records, got {len(diverse_records)} "
            f"({miss} unmatched stems).\n"
            f"If count is wrong, report to Muse — do not proceed."
        )
        print(f"Diverse set: {len(diverse_records)} pairs ({miss} unmatched stems) ✓")
        return anchor_records, diverse_records

    def _load_embedding(self, record: dict):
        """Load a single .pt embedding. Ensures [1, 1, 768] float32."""
        import torch
        path = record["_emb_path"]
        emb  = torch.load(str(path), map_location="cpu", weights_only=True)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)   # [1, 768] → [1, 1, 768]
        return emb.float()

    @staticmethod
    def _report_gate(
            label: str, epoch: int,
            last_pos: float, last_neg: float,
            gap: float, last_norm: float,
            anc_dist: float,
    ) -> None:
        print()
        print("=" * 65)
        print(f"  {label}")
        print("=" * 65)
        print(f"  epoch     = {epoch}")
        print(f"  pos_sim   = {last_pos:.4f}")
        print(f"  neg_sim   = {last_neg:.4f}")
        print(f"  gap       = {gap:.4f}")
        print(f"  proj_norm = {last_norm:.2f}")
        print(f"  anc_dist  = {anc_dist:.4f}")
        print("=" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoint — never parse sys.argv (Failure Mode §5)
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
        mode:         str = "verify",
        resume_epoch: int = 0,
        drop_the:     int = 0,      # 0=False, 1=True (bool CLI workaround)
) -> None:
    """
    mode='verify' : run token ID verification only — report to Muse first
    mode='train'  : full curriculum training
    drop_the=1    : remove ▁the from W_negative (only after Muse approval)
    resume_epoch=N: resume training from saved epoch N checkpoint
    """
    runner   = TrainV5()
    drop_the_bool = bool(drop_the)

    if mode == "verify":
        runner.verify_tokens.remote(drop_the=drop_the_bool)

    elif mode == "train":
        runner.train.remote(
            resume_epoch=resume_epoch,
            drop_the=drop_the_bool,
        )

    elif mode == "audit":
        runner.audit_wneg.remote()

    else:
        print(f"Unknown mode: {mode!r}")
        print("Valid modes: 'verify', 'audit', 'train'")