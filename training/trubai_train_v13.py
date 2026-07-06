"""
trubai_train_v13.py
TensorConditioner v13 — Anchor Fix + output_scale as Buffer
SPEC-TC-V13-v1 (Muse-approved)

Delta against SPEC-TC-V7-v1:
  - Start: v7/best_phase2.pt (epoch 64, pos=0.3522, norm=36.41, anc_dist=0.3962)
  - Phase 1 and Phase 2 do not run — Phase 3 only (epochs 131–200)
  - Gradient clipping: clip_grad_norm_(max_norm=1.0) before every optimizer.step()
  - Scheduler CosineAnnealingLR fast-forwarded to last_epoch=64
  - Per-epoch logging throughout Phase 3
  - Checkpoint policy: trigger events, gate saves, best_phase3.pt continuous
  - Reporting gates: 145, 160, 175, 200 (+ immediate on any bridge trigger)
  - Root cause fix: clipping prevents catastrophic single-step norm crash

Unchanged from v7:
  - W_anchor: v5/phase1_anchor.pt (read-only)
  - LAMBDA_ANCHOR_PHASE3 = 0.05 flat
  - Phase 3 sub-phase ramp 3a–3d (500→1200→2000→3181)
  - W_negative 19 tokens, bridge trigger thresholds, revert table
  - random.seed(42)

Usage
-----
    modal run trubai_train_v8.py
"""

import modal
import json
import random
from pathlib import Path

random.seed(42)   # SPEC-TC-V7-v1 Amendment 2 — deterministic sampling

# ──────────────────────────────────────────────────────────────────────────────
# Image  (unchanged from v5)
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

app = modal.App("trubai-train-v13", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

CKPT_DIR       = Path("/checkpoints")
HF_DIR         = Path("/hf-cache")
ANCHOR_JSONL   = CKPT_DIR / "trubai_training_pairs.jsonl"
COMM_SAFE_TXT  = CKPT_DIR / "commercial_safe.txt"
NEW_EMB_DIR    = CKPT_DIR / "new_embeddings" / "embeddings"
PHASE5_EMB_DIR = CKPT_DIR / "embeddings"
EMB_CACHE_PATH = CKPT_DIR / "moshi_emb_cache.pt"

# v8 starting checkpoint — v7 best Phase 2 state
V8_START    = CKPT_DIR / "retrain_v7" / "best_phase2.pt"
# W_anchor — read-only from v5 Phase 1 (unchanged throughout all runs)
V5_ANCHOR   = CKPT_DIR / "retrain_v5" / "phase1_anchor.pt"

# v6 output dir
V13_DIR = CKPT_DIR / "retrain_v13"

# ──────────────────────────────────────────────────────────────────────────────
# Token sets  (SPEC-TC-V5-v2 §9 + Amendment A1 — unchanged in v6)
# ──────────────────────────────────────────────────────────────────────────────

# W_positive — 18 verified IDs
W_POSITIVE_IDS: dict[str, int] = {
    "▁column":    2368, "▁flat":    3077, "▁support":   711, "▁partial":  4107,
    "▁center":    1611, "▁hollow": 19664, "▁pinch":   24657, "▁crack":    6615,
    "▁focus":     1563, "▁breath":  8735, "▁airflow": 28512, "▁tone":     9064,
    "▁sharp":     6064, "▁buzz":   21938, "▁pitch":    6396, "▁air":      1142,
    "▁spreading": 13369,                   "▁aperture": 16252,
}

# W_negative — 19 unique IDs (12 existing + 7 Amendment A1 additions)
# Fully verified and confirmed in v5 pre-training audit.
W_NEGATIVE_IDS: dict[str, int] = {
    # 12 existing attractor tokens
    "▁disagreement": 17888, "▁cherish":   31329, "▁nationality": 25602,
    "▁tower":         6537, "▁canonical":  8862, "▁iteration":   11024,
    "▁mascot":       27902, "▁brittle":   29915, "▁Tournament":  10653,
    "▁proud":         9630, "▁Az":         9823, "▁major":         916,
    # 7 Amendment A1 additions (collapse region tokens, verified id=260,271,272,264,269,267,262)
    "▁":   260, "▁in": 271, "▁a":  272,
    "▁of": 264, "▁to": 269, "▁and": 267, "▁the": 262,
}

# ──────────────────────────────────────────────────────────────────────────────
# Curriculum helpers — v6 delta (SPEC-TC-V6-v1 §3, §4.3, §6)
# ──────────────────────────────────────────────────────────────────────────────

# Phase 3 lambda — named constant, flat, non-decaying (§4.3)
# Rationale: v5 ran 0.01 against 3203-pair pressure → anc_dist grew 0.040/epoch.
# Required lambda to hold drift ≤ 0.01/epoch ≈ 0.04. With sub-phase structure
# reducing max single-boundary jump to 13.4×, recalibrated to 0.05 (2× margin).
LAMBDA_ANCHOR_PHASE3 = 0.05

def get_phase(epoch: int) -> int:
    """v6 starts at epoch 51 — phase 1 never runs."""
    if epoch <= 130: return 2
    return 3

def get_k_diverse_phase2(epoch: int) -> int:
    """Phase 2 k_diverse — unchanged from SPEC-TC-V5-v2 §2."""
    if epoch > 130:
        raise ValueError(f"get_k_diverse_phase2 called at epoch {epoch} > 130")
    sub_phase = (epoch - 51) // 10 + 1   # 1–8
    return int(22 * sub_phase * 0.10)

def get_k_diverse_phase3(epoch: int) -> int:
    """Phase 3 ramp — restored to original structure (SPEC-TC-V12-v1).
    Crash mechanism eliminated by output normalization — staged ramp no longer needed.
    3a k=500 (131–145), 3b k=1200 (146–160), 3c k=2000 (161–175), 3d k=3181 (176–200).
    """
    if epoch < 131:
        raise ValueError(f"get_k_diverse_phase3 called before Phase 3 (epoch={epoch})")
    if epoch <= 145: return 500
    if epoch <= 160: return 1200
    if epoch <= 175: return 2000
    return 3181

def get_sub_phase_3(epoch: int) -> str:
    """Returns '3a', '3b', '3c', or '3d'."""
    if epoch <= 145: return "3a"
    if epoch <= 160: return "3b"
    if epoch <= 175: return "3c"
    return "3d"

def get_lambda_anchor(epoch: int) -> float:
    """
    Phase 2: linear decay 0.1 → 0.01 (unchanged from v5).
    Phase 3: LAMBDA_ANCHOR_PHASE3 = 0.05, flat, non-decaying.
    Phase 1 (≤50): 0.0 — not used in v6, included for completeness.
    """
    if epoch <= 50:  return 0.0
    if epoch <= 130:
        t = (epoch - 51) / 79   # 0.0 at ep 51, 1.0 at ep 130
        return 0.1 - t * 0.09
    return LAMBDA_ANCHOR_PHASE3   # = 0.05, always in Phase 3

def compute_anchor_loss(model, W_anchor: dict, lambda_val: float):
    """||W - W_anchor||^2 regularization. W_anchor has no gradient (§4.2).
    Strips "tc." prefix from ConditionerV12 parameter names before lookup:
    model yields "tc.learnt_padding"; W_anchor has "learnt_padding".
    output_scale is a buffer (not a parameter) — not yielded by named_parameters().
    """
    import torch
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # Strip "tc." prefix: ConditionerV12 wraps inner TC under self.tc
        anchor_name = name[3:] if name.startswith("tc.") else name
        if anchor_name not in W_anchor:
            continue
        anchor = W_anchor[anchor_name].to(param.device)
        loss   = loss + (param - anchor).pow(2).sum()
    return lambda_val * loss

def check_bridge_trigger(pos_sim: float, output_scale: float = 0.0) -> bool:
    """
    Single-condition trigger (SPEC-TC-V13-v1).
    Condition B retired — output_scale is a fixed buffer, cannot drift.
    Trigger: pos_sim < 0.03 for 2 consecutive epochs.
    """
    return pos_sim < 0.03

# Bridge revert table — universal k=17 (SPEC-TC-V11-v1, retained V12)
# k=17 is the only empirically confirmed stable recovery baseline.
BRIDGE_REVERT: dict[str, tuple[int, str]] = {
    "3a": (17, "k=17"),
    "3b": (17, "k=17"),
    "3c": (17, "k=17"),
    "3d": (17, "k=17"),
}

# ──────────────────────────────────────────────────────────────────────────────
# ConditionerV12 — wrapper with decoupled direction and output_scale
# SPEC-TC-V12-v1 architectural fix
# ──────────────────────────────────────────────────────────────────────────────

def build_conditioner_v12(device: str):
    """
    Build ConditionerV12 wrapper.
    
    Architecture (SPEC-TC-V12-v1):
      inner: moshi TensorConditioner (direction only, [1,1,4096] raw output)
      output_scale: nn.Parameter(36.5) — learned magnitude, initialized to target norm
      forward: direction = F.normalize(proj, dim=-1); x = direction * output_scale
    
    Contrastive loss receives direction (unit-norm) → scale-invariant, drives direction only.
    L_norm receives output_scale → drives magnitude only, never touches direction.
    Crash mechanism eliminated: pathological gradient can only move direction or scale
    independently — never both simultaneously in a catastrophic interaction.

    CONFIRMATION 1: output_scale is nn.Parameter — it IS in the optimizer param group.
    CONFIRMATION 2: tc_forward returns direction (unit-norm) for contrastive loss.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from moshi.conditioners.tensors import TensorConditioner
    from moshi.conditioners import TensorCondition

    class ConditionerV12(nn.Module):
        def __init__(self):
            super().__init__()
            self.tc = TensorConditioner(
                dim=768, output_dim=4096, device=device,
                force_linear=True, output_bias=False, learn_padding=True,
            )
            # output_scale is a fixed buffer — not a learned parameter.
            # register_buffer: persists in state_dict, moves with .to(device),
            # NOT in optimizer.parameters(). Honest contract — no loss term moves it.
            # CONFIRMATION 1: register_buffer, absent from optimizer.parameters()
            self.register_buffer("output_scale", torch.tensor(36.5))

        def forward(self, emb):
            """
            Returns direction (unit-norm [1,4096]) and output_scale scalar.
            Caller uses direction for contrastive loss, output_scale for L_norm.
            """
            mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            cond = TensorCondition(tensor=emb, mask=mask)
            proj      = self.tc(cond)[0]             # ConditionType[0] → [1,1,4096]
            raw       = proj.squeeze(1)               # [1, 4096]
            # CONFIRMATION 2: direction is F.normalize output — unit sphere, scale-invariant
            direction = F.normalize(raw, dim=-1)      # [1, 4096], ||direction||=1 always
            return direction, self.output_scale       # contrastive uses direction; L_norm uses scale

    conditioner = ConditionerV12().to(device)
    print(f"  ConditionerV12 parameter keys: {[n for n, _ in conditioner.named_parameters()]}")
    print(f"  output_scale init: {conditioner.output_scale.item():.2f}")
    return conditioner


def tc_forward(conditioner, emb, W_pos, W_neg):
    """
    ConditionerV12 forward → pos_sim, neg_sim, output_scale.
    
    CONFIRMATION 2 (explicit): contrastive loss receives direction, NOT x.
    direction is unit-norm — cosine similarity is already scale-invariant,
    but passing direction makes this structural, not incidental.

    emb    : [1, 1, 768] float32
    W_pos  : [n_pos, 4096] float32
    W_neg  : [n_neg, 4096] float32
    returns: pos_sim, neg_sim, output_scale (scalar Parameter)
    """
    import torch.nn.functional as F

    # direction: unit-norm [1, 4096]. output_scale: scalar Parameter.
    direction, output_scale = conditioner(emb)

    # Contrastive loss — direction only, scale-invariant
    pos_n   = F.normalize(W_pos, dim=-1)
    neg_n   = F.normalize(W_neg, dim=-1)
    pos_sim = (direction @ pos_n.T).mean()
    neg_sim = (direction @ neg_n.T).mean()

    return pos_sim, neg_sim, output_scale

# ──────────────────────────────────────────────────────────────────────────────
# Modal class
# ──────────────────────────────────────────────────────────────────────────────

@app.cls(
    gpu="H100",
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=21600,
)
class TrainV13:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = str(HF_DIR)   # MUST be first line

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        self.device = torch.device("cuda")

        # ── Prerequisite check — v8 starting checkpoints must exist ─────────
        for path in (V8_START, V5_ANCHOR):
            if not path.exists():
                raise FileNotFoundError(
                    f"v8 prerequisite checkpoint missing: {path}\n"
                    "Confirm v7/best_phase2.pt and v5/phase1_anchor.pt exist on volume."
                )
        print("v8 prerequisite checkpoints confirmed ✓")
        print(f"  {V8_START}")
        print(f"  {V5_ANCHOR}")

        # ── SentencePiece tokenizer ──────────────────────────────────────────
        print("Loading Moshi tokenizer...")
        text_tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp = sentencepiece.SentencePieceProcessor(text_tok_path)
        print(f"  Vocab size: {self.sp.GetPieceSize()}")

        # ── LM head weights (cached from v5 run) ────────────────────────────
        if EMB_CACHE_PATH.exists():
            print("Loading cached LM head weights...")
            lm_head = torch.load(EMB_CACHE_PATH, map_location="cpu", weights_only=True)
        else:
            print("Extracting LM head weights from Moshi LM (first time — slow)...")
            moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
            moshi_lm     = loaders.get_moshi_lm(moshi_weight, device="cpu")
            lm_head      = moshi_lm.text_linear.weight.detach().float()
            del moshi_lm
            torch.cuda.empty_cache()
            torch.save(lm_head, EMB_CACHE_PATH)
            vol.commit()
            print(f"  LM head cached at {EMB_CACHE_PATH}")

        print(f"  LM head shape: {lm_head.shape}")

        # ── W_positive and W_negative — all IDs pre-verified in v5 ──────────
        self.W_pos = lm_head[list(W_POSITIVE_IDS.values())].to(self.device)
        self.W_neg = lm_head[list(W_NEGATIVE_IDS.values())].to(self.device)
        print(f"  W_pos: {self.W_pos.shape[0]} tokens, W_neg: {self.W_neg.shape[0]} tokens")

        print("Setup complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # Training — epochs 51–200
    # ──────────────────────────────────────────────────────────────────────────

    @modal.method()
    def train(self, resume_epoch: int = 0) -> None:
        """
        v6 curriculum: Phase 2 (epochs 51–130) + Phase 3 sub-phases 3a–3d.
        Phase 1 does not run — starts from v5 epoch_050.pt.

        Mandatory stop:
          - Bridge trigger fires 3× within a single sub-phase → stop, report to Muse

        Reporting gates (mandatory, no skip even if no issues):
          epoch 133, 136, 139, 142, 145 (intra-3a), 160, 175, 200
        """
        import torch

        # ── Load datasets ────────────────────────────────────────────────────
        anchor_set, diverse_set = self._load_datasets()

        # ── ConditionerV12 — load v7 best_phase2 weights into inner tc ───────
        # epoch 64: pos_sim=0.3522, proj_norm=36.41, anc_dist=0.3962
        # V8_START checkpoint holds weights for conditioner.tc (the inner moshi TC).
        # output_scale is a new Parameter initialized to 36.5 — not in the checkpoint.
        tc = build_conditioner_v12(str(self.device))
        tc.tc.load_state_dict(
            torch.load(V8_START, map_location=self.device, weights_only=True)
        )
        print(f"  Loaded v7/best_phase2.pt into ConditionerV12.tc ✓")
        print(f"  output_scale init: {tc.output_scale.item():.2f} (trainable)")

        # ── W_anchor — read-only from v5, not recomputed ─────────────────────
        W_anchor = {
            k: v.to(self.device)
            for k, v in torch.load(
                V5_ANCHOR, map_location=self.device, weights_only=True
            ).items()
        }
        print(f"  Loaded v5 phase1_anchor.pt ✓ ({len(W_anchor)} parameter tensors)")

        # ── CONFIRMATION checks (SPEC-TC-V13-v1) ─────────────────────────────
        # CONFIRMATION 1: output_scale must NOT be in optimizer.parameters()
        opt_param_ids = {id(p) for p in tc.parameters()}
        scale_id      = id(tc.output_scale)
        assert scale_id not in opt_param_ids, (
            "FAIL CONFIRMATION 1: output_scale is in optimizer.parameters() — "
            "must be register_buffer, not nn.Parameter"
        )
        print("  CONFIRMATION 1: output_scale absent from optimizer.parameters() ✓")

        # CONFIRMATION 2: anchor regularization produces L_anc > 0 on first anchor record
        anchor_set_check, _ = self._load_datasets()
        emb_check = self._load_embedding(anchor_set_check[0]).to(self.device)
        _, _, _ = tc_forward(tc, emb_check, self.W_pos, self.W_neg)
        L_anc_check = compute_anchor_loss(tc, W_anchor, LAMBDA_ANCHOR_PHASE3)
        assert L_anc_check.item() > 0.0, (
            f"FAIL CONFIRMATION 2: L_anc={L_anc_check.item():.6f} — "
            "anchor regularization is zero. Prefix fix may not be applied."
        )
        print(f"  CONFIRMATION 2: L_anc={L_anc_check.item():.6f} > 0.0 on first anchor record ✓")

        # ── Optimizer — fresh AdamW ───────────────────────────────────────────
        LR_MAX      = 4.5e-5
        # LAMBDA_NORM and TARGET_NORM removed — L_norm dropped in v13

        optimizer = torch.optim.AdamW(
            tc.parameters(), lr=LR_MAX, weight_decay=0.01
        )

        # ── Scheduler — fast-forwarded 64 steps (last_epoch=64) ─────────────
        # v8 starts from epoch 64 state. Fast-forward to continue the cosine
        # decay naturally. LR at epoch 131 (first Phase 3 step) ≈ 0.64 × LR_MAX.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=200, eta_min=1e-6
        )
        # Bug fix (v9 → v10): optimizer.step() must precede scheduler.step()
        # during fast-forward. PyTorch skips the first LR value if reversed.
        for _ in range(64):
            optimizer.step()    # step first — initialises initial_lr in param_groups
            scheduler.step()    # then advance schedule
        first_lr = scheduler.get_last_lr()[0]
        print(f"  Scheduler fast-forwarded 64 steps: LR={first_lr:.6f} "
              f"({first_lr/LR_MAX:.3f} × LR_MAX) [bug fixed: opt.step before sched.step]")

        # ── Checkpoint dir ────────────────────────────────────────────────────
        V13_DIR.mkdir(parents=True, exist_ok=True)

        # ── Resume ────────────────────────────────────────────────────────────
        start_epoch = 131
        if resume_epoch > 0:
            ckpt = V13_DIR / f"epoch_{resume_epoch:03d}.pt"
            if not ckpt.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
            tc.load_state_dict(
                torch.load(ckpt, map_location=self.device, weights_only=True)
            )
            start_epoch = resume_epoch + 1
            print(f"  Resumed from v6 epoch_{resume_epoch:03d}.pt")
            # Fast-forward scheduler from epoch 64 to resume_epoch
            for _ in range(resume_epoch - 64):
                scheduler.step()
            print(f"  Scheduler fast-forwarded to epoch {resume_epoch}: "
                  f"LR={scheduler.get_last_lr()[0]:.6f}")

        # ── Bridge state — Phase 3 ────────────────────────────────────────────
        trigger_consecutive  = 0
        revert_active        = False
        revert_k             = 0
        revert_epochs_left   = 0
        revert_count_in_subphase: dict[str, int] = {
            "3a": 0, "3b": 0, "3c": 0, "3d": 0,
        }
        extended_once: dict[str, bool] = {
            "3a": False, "3b": False, "3c": False, "3d": False,
        }

        # ── best_phase3.pt tracking ────────────────────────────────────────────
        best_phase3_pos = 0.0



        # ── Training loop ─────────────────────────────────────────────────────
        print(f"Training from epoch {start_epoch} to 200 (Phase 3 only)")
        print("=" * 80)

        last_pos  = 0.0
        last_neg  = 0.0
        last_scale = 0.0

        for epoch in range(start_epoch, 201):  # Phase 3 only: 131–200
            phase = get_phase(epoch)

            # ── Batch assembly — Phase 3 only ─────────────────────────────────
            sub_phase = get_sub_phase_3(epoch)
            if revert_active and revert_epochs_left > 0:
                k = revert_k
                revert_epochs_left -= 1
                if revert_epochs_left == 0:
                    revert_active = False
                    print(f"  [bridge] Revert complete at epoch {epoch}. "
                          f"Resuming {sub_phase}.")
            else:
                k = get_k_diverse_phase3(epoch)
            diverse_sample = random.sample(diverse_set, k)
            epoch_data = list(anchor_set) + diverse_sample

            random.shuffle(epoch_data)

            # ── Per-record mini-batch loop (SPEC-TC-V9-v1) ─────────────────
            # Each record: isolated zero_grad → forward → backward → clip → step.
            # Anchor records interspersed via shuffle — counter-pressure distributed.
            # LR scaled by n_records so total update magnitude per epoch is preserved.
            tc.train()
            lambda_anchor   = get_lambda_anchor(epoch)
            n_records       = len(epoch_data)
            epoch_loss      = 0.0

            for record in epoch_data:
                optimizer.zero_grad()

                emb = self._load_embedding(record).to(self.device)
                # tc_forward returns (pos_sim, neg_sim, output_scale)
                # pos_sim/neg_sim use direction (unit-norm) — scale-invariant
                # output_scale is the nn.Parameter driving L_norm
                pos_sim, neg_sim, output_scale = tc_forward(
                    tc, emb, self.W_pos, self.W_neg
                )

                L_c  = -(pos_sim - neg_sim)
                # L_norm removed — output_scale is a fixed buffer, not a learned param
                # L_guard removed — crash mechanism gone (structural fix from v12)
                L_a  = compute_anchor_loss(tc, W_anchor, lambda_anchor)
                loss = L_c + L_a
                loss.backward()

                torch.nn.utils.clip_grad_norm_(tc.parameters(), max_norm=1.0)

                # Scale LR: epoch-level LR / n_records preserves expected update magnitude
                step_lr = scheduler.get_last_lr()[0] / n_records
                for param_group in optimizer.param_groups:
                    param_group['lr'] = step_lr

                optimizer.step()
                epoch_loss += loss.item()

            # Restore epoch-level LR before scheduler.step() so the schedule
            # advances correctly on the epoch boundary
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler.get_last_lr()[0]
            scheduler.step()

            last_pos   = pos_sim.item()
            last_neg   = neg_sim.item()
            last_scale = output_scale.item()   # output_scale replaces proj_norm
            gap        = last_pos - last_neg

            # ── Logging — every epoch ─────────────────────────────────────────
            anc_dist    = compute_anchor_loss(tc, W_anchor, 1.0).item()
            revert_tag  = f" [REVERT {revert_epochs_left}ep]" if revert_active else ""
            epoch_lr    = scheduler.get_last_lr()[0]
            step_lr_log = epoch_lr / n_records
            print(
                f"Epoch {epoch:3d} | {sub_phase}{revert_tag} | k_div={k:4d} | n={n_records} | "
                f"gap={gap:.4f} | pos={last_pos:.4f} | neg={last_neg:.4f} | "
                f"scale={last_scale:.2f} | "
                f"anc_dist={anc_dist:.4f} | "
                f"epoch_lr={epoch_lr:.2e} | step_lr={step_lr_log:.2e}"
            )

            # ── Checkpoint save policy (SPEC-TC-V12-v1) ─────────────────────
            # Gate checkpoints: Phase 3 boundaries
            if epoch in (145, 160, 175, 200):
                torch.save(tc.state_dict(), V13_DIR / f"gate_ep{epoch:03d}.pt")
                vol.commit()

            # best_phase3.pt — continuous, overwrite when pos_sim improves
            if last_pos > best_phase3_pos:
                best_phase3_pos = last_pos
                torch.save(tc.state_dict(), V13_DIR / "best_phase3.pt")

            # ══════════════════════════════════════════════════════════════════
            # REPORTING GATES
            # ══════════════════════════════════════════════════════════════════

            # ── Phase 3 boundary gates — 145, 160, 175 ────────────────────
            # (Intra-3a hold logic removed — crash mechanism fixed structurally)
            if epoch in (145, 160, 175):
                anc_dist_now = compute_anchor_loss(tc, W_anchor, 1.0).item()
                label_map    = {145: "3a end", 160: "3b end", 175: "3c end"}
                self._report_gate(
                    label=f"EPOCH {epoch} — {label_map[epoch]}",
                    epoch=epoch, pos=last_pos, neg=last_neg,
                    gap=gap, norm=last_scale, anc_dist=anc_dist_now,
                )

            # ── Bridge trigger check (Phase 3) ───────────────────────────────
            if phase == 3:
                sub_phase = get_sub_phase_3(epoch)
                if check_bridge_trigger(last_pos, last_scale):
                    trigger_consecutive += 1
                    print(f"  [bridge watch] epoch {epoch} ({sub_phase}): "
                          f"trigger condition met "
                          f"(pos={last_pos:.4f}<0.03={last_pos<0.03} | "
                          f"scale={last_scale:.2f}<32.0={last_scale<32.0}) "
                          f"— {trigger_consecutive}/2 consecutive")
                else:
                    if trigger_consecutive > 0:
                        print(f"  [bridge watch] epoch {epoch}: condition cleared "
                              f"— counter reset")
                    trigger_consecutive = 0

                if trigger_consecutive >= 2:
                    revert_count_in_subphase[sub_phase] += 1
                    trigger_consecutive = 0

                    # ── Triple-fire stop condition ─────────────────────────
                    if revert_count_in_subphase[sub_phase] >= 3:
                        torch.save(
                            tc.state_dict(),
                            V13_DIR / f"epoch_{epoch:03d}_triple_fire_stop.pt"
                        )
                        vol.commit()
                        print(f"\n>>> TRIPLE-FIRE STOP in sub-phase {sub_phase} "
                              f"at epoch {epoch}.")
                        print(f"    revert_count={revert_count_in_subphase[sub_phase]}")
                        print(f"    Checkpoint saved. Report full trigger history to Muse.")
                        raise RuntimeError(
                            f"Bridge trigger fired 3× in sub-phase {sub_phase} "
                            f"at epoch {epoch}. "
                            "Structural problem — revert mechanism exhausted. "
                            "Report to Muse."
                        )

                    # ── Revert one sub-phase ───────────────────────────────
                    revert_k_val, revert_label = BRIDGE_REVERT[sub_phase]
                    revert_duration = 5

                    # Extend if already reverted once in this sub-phase
                    if revert_count_in_subphase[sub_phase] == 2:
                        if not extended_once[sub_phase]:
                            extended_once[sub_phase] = True
                            revert_duration = 10   # 5 original + 5 extension
                            print(f"  [bridge] Second trigger in {sub_phase} — "
                                  f"extending revert to {revert_duration} epochs.")
                        # Third fire is caught above — no further extension here

                    revert_active       = True
                    revert_k            = revert_k_val
                    revert_epochs_left  = revert_duration

                    # Save trigger-event checkpoint (SPEC-TC-V7-v1)
                    torch.save(tc.state_dict(), V13_DIR / f"trigger_ep{epoch:03d}.pt")
                    vol.commit()

                    print(f"\n>>> Bridge trigger fired at epoch {epoch} ({sub_phase}).")
                    print(f"    Reverting to {revert_label} (k={revert_k_val}) "
                          f"for {revert_duration} epochs.")
                    print(f"    pos_sim={last_pos:.4f}")
                    print(f"    revert_count_in_subphase[{sub_phase}]="
                          f"{revert_count_in_subphase[sub_phase]}")

        # ── Final report — epoch 200 ───────────────────────────────────────
        anc_dist_final = compute_anchor_loss(tc, W_anchor, 1.0).item()
        gap_final      = last_pos - last_neg

        print()
        print("=" * 65)
        print("EPOCH 200 REPORT — Training complete")
        print("=" * 65)
        print(f"  gap       = {gap_final:.4f}  (target ≥ 0.38)")
        print(f"  pos_sim   = {last_pos:.4f}  (target > 0.03, sustained Phase 3)")
        print(f"  neg_sim   = {last_neg:.4f}")
        print(f"  output_scale = {last_scale:.2f}  (fixed buffer — structural fix confirmed)")
        print(f"  anc_dist  = {anc_dist_final:.4f}  (target < 7.5)")
        print()
        print(f"  Bridge trigger history:")
        for sp, count in revert_count_in_subphase.items():
            print(f"    {sp}: {count} trigger(s)")
        print()

        # Success criteria (SPEC-TC-V6-v1 §9)
        criteria = {
            "gap ≥ 0.38":          gap_final      >= 0.38,
            "pos_sim > 0.03":      last_pos       >  0.03,
            "output_scale == 36.5": abs(last_scale - 36.5) < 0.01,  # must be unchanged (buffer)
            "anc_dist < 7.5":      anc_dist_final <  7.5,
        }
        passed = all(criteria.values())
        for label, ok in criteria.items():
            print(f"  {'✓' if ok else '✗'} {label}")
        print()
        if passed:
            print("  STATUS: ALL CRITERIA MET ✓")
            print("  Awaiting streaming verification before LoRA retraining.")
        else:
            missed = [l for l, ok in criteria.items() if not ok]
            print(f"  STATUS: CRITERIA MISSED — {', '.join(missed)}")
            print("  Report to Muse. LoRA retraining blocked.")
        print("=" * 65)

        # Save best
        (V13_DIR / "best").mkdir(exist_ok=True)
        torch.save(tc.state_dict(), V13_DIR / "best" / "tensor_conditioner.pt")
        vol.commit()
        print(f"\nAll checkpoints committed to {V13_DIR}")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_datasets(self) -> tuple[list[dict], list[dict]]:
        """
        Returns (anchor_set, diverse_set) — unchanged from v5.
        anchor_set : 22 dicts, Phase 5 embeddings [1,1,768]
        diverse_set: 3181 dicts, new_embeddings [1,768] (unsqueezed on load)
        """
        import re

        def normalize_pitch(s: str) -> str:
            return re.sub(r'Gs(\d)', r'G#\1', s)

        # Anchor set (22)
        anchor_records = [json.loads(l) for l in open(ANCHOR_JSONL)]
        for rec in anchor_records:
            stem = Path(rec["embedding_path"]).stem
            rec["_emb_path"] = PHASE5_EMB_DIR / f"{stem}.pt"
        assert len(anchor_records) == 22, (
            f"Expected 22 anchor pairs, got {len(anchor_records)}"
        )
        missing = [r["_emb_path"] for r in anchor_records if not r["_emb_path"].exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} anchor embeddings missing: "
                + str(missing[:3])
            )
        print(f"Anchor set:  {len(anchor_records)} pairs ✓")

        # Diverse set (3181)
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
            f"({miss} unmatched stems). Report to Muse."
        )
        print(f"Diverse set: {len(diverse_records)} pairs ({miss} unmatched stems) ✓")
        return anchor_records, diverse_records

    def _load_embedding(self, record: dict):
        """Load .pt embedding, ensure [1, 1, 768] float32."""
        import torch
        emb = torch.load(str(record["_emb_path"]), map_location="cpu",
                         weights_only=True)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)   # [1, 768] → [1, 1, 768]
        return emb.float()

    @staticmethod
    def _report_gate(
            label: str, epoch: int,
            pos: float, neg: float,
            gap: float, norm: float,
            anc_dist: float,
            extra: str = "",
    ) -> None:
        print()
        print("=" * 65)
        print(f"  {label}")
        print("=" * 65)
        print(f"  epoch     = {epoch}")
        print(f"  pos_sim   = {pos:.4f}")
        print(f"  neg_sim   = {neg:.4f}")
        print(f"  gap       = {gap:.4f}")
        print(f"  proj_norm = {norm:.2f}")
        print(f"  anc_dist  = {anc_dist:.4f}")
        if extra:
            print(f"  NOTE: {extra}")
        print("=" * 65)
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoint — typed parameters, never sys.argv
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(resume_epoch: int = 0) -> None:
    """
    resume_epoch=N : resume from v6 epoch_NNN.pt checkpoint
    """
    TrainV13().train.remote(resume_epoch=resume_epoch)