# TRUB.AI — TensorConditioner Retraining v3
# Full commercial_safe dataset (3,342 embeddings).
# Loss, token sets, hyperparameters: identical to Phase 5 run 2 (retrain_v2/best).
#
# Changes from Phase 5 script:
#   - Loads from /checkpoints/new_embeddings/ (3342 files) instead of 22-pair set
#   - Embeddings are [1,768] from Task 3 pipeline — unsqueezed to [1,1,768] on load
#   - CKPT_EVERY = 10 (was 20)
#   - Early stop if proj_norm > 55.0 at any checkpoint
#   - Saves to retrain_v3/ — retrain_v2/best untouched
#   - Epoch target: 120 minimum (script runs to 200, stop manually after 120 if gap stable)
#
# Do not touch: LoRA, 22-pair training pairs, moshi fork, streaming loop.
# ─────────────────────────────────────────────────────────────────────────────

import modal
import numpy as np
import json
import typing as tp
from pathlib import Path

app = modal.App("trubai-retrain-conditioner-v3")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg", "unzip"])
    .pip_install([
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "moshi",
        "transformers",
        "sentencepiece",
        "huggingface_hub",
    ])
)

volume   = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config — identical to Phase 5 run 2 except paths and CKPT_EVERY
# ─────────────────────────────────────────────────────────────────────────────

# New embeddings uploaded to Modal volume
EMBEDDINGS_ZIP      = "/checkpoints/new_embeddings.zip"
EMBEDDINGS_UNZIPPED = "/checkpoints/new_embeddings"
PARTITION_TXT       = "/checkpoints/commercial_safe.txt"

# Output — retrain_v2/best stays untouched
CKPT_DIR = "/checkpoints/retrain_v3"

MERT_DIM  = 768
MOSHI_DIM = 4096

EPOCHS             = 200        # run to 200; Muse minimum is 120
LR                 = 3e-4
WEIGHT_DECAY       = 0.01
LAMBDA_CONTRASTIVE = 0.7
LAMBDA_DIVERSITY   = 0.3
MARGIN             = 0.3
VIRTUAL_BATCH      = 8
LOG_EVERY          = 10
CKPT_EVERY         = 10         # Muse spec: save every 10
NORM_STOP          = 55.0       # early stop threshold — flag to Muse if exceeded
L_NORM_TARGET      = 36.5       # output norm anchor (original proj norm)
L_NORM_LAMBDA      = 0.1        # norm penalty weight

# ── Token sets — exact Phase 5 strings, do not modify ────────────────────────
TRUMPET_WORDS = [
    " air", " column", " tone", " breath", " aperture",
    " center", " embouchure", " cracked", " flat", " sharp",
    " breathy", " pitch", " partial", " buzz", " focus",
    " pinched", " spreading", " hollow", " airflow", " support",
]
# embouchure not in Moshi vocab → silently dropped → 18 resolved tokens

ATTRACTOR_WORDS = [
    " proud", " brittle", " ville", " nationality",
    " Tournament", " disagreement", " Az", " tower",
    " mascot", " cherish", " canonical", " iteration",
]
# 12 words → 11 unique token ids after dedup+filter


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
    timeout=7200,
)
class ConditionerRetrainerV3:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"

        import torch
        import subprocess
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners.tensors import TensorConditioner
        from moshi.conditioners import TensorCondition

        self.device = "cuda"

        # ── Unzip new embeddings if not already done ──────────────────────────
        emb_dir = Path(EMBEDDINGS_UNZIPPED)
        if not emb_dir.exists():
            print(f"Unzipping {EMBEDDINGS_ZIP} → {emb_dir}")
            emb_dir.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                ["unzip", "-q", EMBEDDINGS_ZIP, "-d", str(emb_dir)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"unzip failed: {result.stderr}")
            print(f"Unzip complete.")
        else:
            print(f"Embeddings already unzipped at {emb_dir}")

        # ── Load commercial_safe file list ────────────────────────────────────
        # commercial_safe.txt contains full paths from Kaggle — we only need stems
        with open(PARTITION_TXT) as f:
            raw_lines = [l.strip() for l in f if l.strip()]

        # Extract stems — match against .pt files in the unzipped dir
        safe_stems = set(Path(l).stem for l in raw_lines)
        print(f"commercial_safe.txt: {len(safe_stems)} stems")

        # ── Discover .pt files, filter to commercial_safe ─────────────────────
        # Unzip may produce a nested subdir — search recursively
        all_pt = list(emb_dir.rglob("*.pt"))
        print(f"Total .pt files found: {len(all_pt)}")

        safe_pt = [p for p in all_pt if p.stem in safe_stems]
        print(f"Matched to commercial_safe: {len(safe_pt)}")

        if len(safe_pt) == 0:
            # Fallback: stems might have chunk suffix — try prefix match
            safe_pt = [p for p in all_pt
                       if any(p.stem.startswith(s) for s in safe_stems)
                       or p.stem in safe_stems]
            print(f"After prefix fallback: {len(safe_pt)}")

        if len(safe_pt) == 0:
            raise RuntimeError(
                "No .pt files matched commercial_safe.txt. "
                "Check that stems in commercial_safe.txt match .pt filenames."
            )

        # ── Load all embeddings into memory ───────────────────────────────────
        print(f"Loading {len(safe_pt)} embeddings into memory...")
        self.embeddings = []
        failed = []
        for pt_path in sorted(safe_pt):
            try:
                emb = torch.load(
                    str(pt_path), map_location=self.device, weights_only=False
                ).float()
                # Task 3 pipeline produces [1, 768] — unsqueeze to [1, 1, 768]
                # to match TensorCondition expected shape
                if emb.dim() == 2:          # [1, 768]
                    emb = emb.unsqueeze(1)  # → [1, 1, 768]
                elif emb.dim() == 3:        # already [1, 1, 768] — Phase 5 format
                    pass
                else:
                    raise ValueError(f"Unexpected embedding shape {emb.shape}")
                self.embeddings.append(emb)
            except Exception as e:
                failed.append((str(pt_path), str(e)))

        if failed:
            print(f"WARNING: {len(failed)} files failed to load:")
            for f, e in failed[:5]:
                print(f"  {f}: {e}")

        print(f"Loaded {len(self.embeddings)} embeddings ✓")

        # ── Load base Moshi LM head weights ───────────────────────────────────
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        base_model   = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        self.lm_head_weights = base_model.text_linear.weight.detach().float()
        print(f"LM head weights: {self.lm_head_weights.shape}")

        # ── Text tokenizer ────────────────────────────────────────────────────
        text_tok_path = hf_hub_download(
            loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME
        )
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(text_tok_path)

        # ── Build W_positive / W_negative — identical to Phase 5 ─────────────
        def words_to_weight_rows(words: list, label: str):
            import torch.nn.functional as F
            ids = []
            for w in words:
                w_stripped = w.strip()
                piece_id   = self.text_tokenizer.piece_to_id(f"▁{w_stripped}")
                if piece_id > 0:
                    ids.append(piece_id)
                else:
                    toks = self.text_tokenizer.encode(w_stripped)
                    for t in toks:
                        if 0 < t < self.lm_head_weights.shape[0]:
                            ids.append(t)
                            break
            ids = list(set(ids) - {0, 260})
            print(f"{label} ({len(ids)} tokens):")
            for i in ids:
                print(f"  '{self.text_tokenizer.id_to_piece(i)}' → {i}")
            return self.lm_head_weights[ids]

        self.W_positive   = words_to_weight_rows(TRUMPET_WORDS,  "W_positive")
        self.W_negative   = words_to_weight_rows(ATTRACTOR_WORDS, "W_negative")

        import torch.nn.functional as F
        self.W_pos_normed = F.normalize(self.W_positive, dim=1)
        self.W_neg_normed = F.normalize(self.W_negative, dim=1)

        # Verify token counts match Phase 5
        assert self.W_positive.shape[0] == 18, \
            f"W_positive: expected 18 tokens, got {self.W_positive.shape[0]}"
        assert self.W_negative.shape[0] == 11, \
            f"W_negative: expected 11 tokens, got {self.W_negative.shape[0]}"
        print(f"Token counts verified: W_pos={self.W_positive.shape[0]}, W_neg={self.W_negative.shape[0]} ✓")

        # ── TensorConditioner ─────────────────────────────────────────────────
        from moshi.conditioners.tensors import TensorConditioner
        self.tensor_conditioner = TensorConditioner(
            dim=768, output_dim=4096, device=self.device,
            force_linear=True, output_bias=False, learn_padding=True,
        ).to(self.device).float()

        print("Setup complete. Ready to train.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _project(self, emb):
        import torch
        from moshi.conditioners import TensorCondition
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        tc   = TensorCondition(tensor=emb, mask=mask)
        out  = self.tensor_conditioner(tc)
        return out.condition.squeeze()  # [4096]

    def _contrastive_loss(self, projection):
        import torch
        import torch.nn.functional as F
        proj      = projection
        L_norm    = (proj.norm() - L_NORM_TARGET) ** 2
        proj_norm = F.normalize(proj.unsqueeze(0), dim=1)  # [1, 4096]
        pos_sim   = (proj_norm @ self.W_pos_normed.T).squeeze(0)
        neg_sim   = (proj_norm @ self.W_neg_normed.T).squeeze(0)
        L_pull    = -pos_sim.mean()
        L_push    = torch.relu(neg_sim + MARGIN).mean()
        return L_pull + L_push + L_NORM_LAMBDA * L_norm

    def _diversity_loss(self, projections):
        import torch.nn.functional as F
        norms      = F.normalize(projections, dim=1)
        sim_matrix = norms @ norms.T
        B          = projections.shape[0]
        import torch
        mask       = ~torch.eye(B, dtype=torch.bool, device=self.device)
        return sim_matrix[mask].mean()

    def _geometric_check(self):
        import torch
        import torch.nn.functional as F
        self.tensor_conditioner.eval()
        pos_sims, neg_sims, proj_norms = [], [], []
        pos_centroid = F.normalize(self.W_positive.mean(0), dim=0)
        neg_centroid = F.normalize(self.W_negative.mean(0), dim=0)

        # Sample up to 200 embeddings for the check — full 3342 every epoch is slow
        import random
        sample = random.sample(self.embeddings, min(200, len(self.embeddings)))

        with torch.no_grad():
            for emb in sample:
                proj      = self._project(emb)
                proj_norms.append(proj.norm().item())
                proj_norm = F.normalize(proj, dim=0)
                pos_sims.append((proj_norm * pos_centroid).sum().item())
                neg_sims.append((proj_norm * neg_centroid).sum().item())

        self.tensor_conditioner.train()
        return (
            float(np.mean(pos_sims)),
            float(np.mean(neg_sims)),
            float(np.mean(proj_norms)),
            float(np.mean(pos_sims)) > float(np.mean(neg_sims)),
        )

    # ── Training ───────────────────────────────────────────────────────────────

    @modal.method()
    def train(self) -> dict:
        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW
        import random

        Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)

        optimizer = AdamW(
            self.tensor_conditioner.parameters(),
            lr=LR, weight_decay=WEIGHT_DECAY,
        )

        best_geo_gap  = -999.0
        best_epoch    = -1
        checkpoint_table = []   # rows for Muse report: epoch|proj_norm|pos_sim|neg_sim|gap
        norm_stop_fired  = False

        print(f"\nTensorConditioner retraining v3")
        print(f"  Embeddings     : {len(self.embeddings)}")
        print(f"  Epochs         : {EPOCHS} (Muse minimum: 120)")
        print(f"  LR             : {LR}")
        print(f"  λ_contrastive  : {LAMBDA_CONTRASTIVE}")
        print(f"  λ_diversity    : {LAMBDA_DIVERSITY}")
        print(f"  Margin         : {MARGIN}")
        print(f"  Norm target    : {L_NORM_TARGET}  λ_norm={L_NORM_LAMBDA}")
        print(f"  CKPT_EVERY     : {CKPT_EVERY}")
        print(f"  Norm stop      : {NORM_STOP}")
        print()
        print(f"  {'Epoch':>6} | {'proj_norm':>9} | {'pos_sim':>8} | {'neg_sim':>8} | {'gap':>8}")
        print(f"  {'-'*52}")

        for epoch in range(1, EPOCHS + 1):
            self.tensor_conditioner.train()
            indices = list(range(len(self.embeddings)))
            random.shuffle(indices)

            epoch_losses = []

            for batch_start in range(0, len(indices), VIRTUAL_BATCH):
                batch_idx   = indices[batch_start:batch_start + VIRTUAL_BATCH]
                batch_embs  = [self.embeddings[i] for i in batch_idx]
                projections = [self._project(e) for e in batch_embs]
                proj_stack  = torch.stack(projections, dim=0)  # [B, 4096]

                L_contrastive = torch.stack(
                    [self._contrastive_loss(p) for p in projections]
                ).mean()
                L_diversity   = self._diversity_loss(proj_stack)
                loss          = LAMBDA_CONTRASTIVE * L_contrastive + LAMBDA_DIVERSITY * L_diversity

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.tensor_conditioner.parameters(), max_norm=1.0
                )
                optimizer.step()
                epoch_losses.append(loss.item())

            pos_mean, neg_mean, norm_mean, passed = self._geometric_check()
            geo_gap   = pos_mean - neg_mean
            mean_loss = float(np.mean(epoch_losses))

            row = {
                "epoch":     epoch,
                "proj_norm": round(norm_mean, 2),
                "pos_sim":   round(pos_mean,  4),
                "neg_sim":   round(neg_mean,  4),
                "gap":       round(geo_gap,   4),
            }

            # Log every CKPT_EVERY epochs and epoch 1
            if epoch % LOG_EVERY == 0 or epoch == 1:
                print(f"  {epoch:>6} | {norm_mean:>9.2f} | {pos_mean:>8.4f} | "
                      f"{neg_mean:>8.4f} | {geo_gap:>8.4f}  {'✓' if passed else '✗'}")
                checkpoint_table.append(row)

            # Early stop: proj_norm exceeded threshold
            if norm_mean > NORM_STOP:
                msg = (f"⚠ NORM STOP at epoch {epoch}: proj_norm={norm_mean:.2f} > {NORM_STOP}. "
                       f"Space token collapse risk. Flagging to Muse.")
                print(msg)
                norm_stop_fired = True
                volume.commit()
                return {
                    "status":           "norm_stop",
                    "stopped_at_epoch": epoch,
                    "proj_norm":        round(norm_mean, 2),
                    "norm_stop_threshold": NORM_STOP,
                    "checkpoint_table": checkpoint_table,
                    "message": msg,
                }

            # Save best
            if geo_gap > best_geo_gap:
                best_geo_gap = geo_gap
                best_epoch   = epoch
                best_path    = Path(CKPT_DIR) / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.tensor_conditioner.to(torch.bfloat16).state_dict(),
                    best_path / "tensor_conditioner.pt",
                    )
                self.tensor_conditioner.float()

            # Periodic checkpoint + volume flush
            if epoch % CKPT_EVERY == 0:
                ckpt_path = Path(CKPT_DIR) / f"epoch_{epoch:04d}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.tensor_conditioner.to(torch.bfloat16).state_dict(),
                    ckpt_path / "tensor_conditioner.pt",
                    )
                self.tensor_conditioner.float()
                volume.commit()
                print(f"  → checkpoint saved: epoch_{epoch:04d}  "
                      f"(best so far: epoch {best_epoch}, gap={best_geo_gap:.4f})")

        # ── Final report ───────────────────────────────────────────────────────
        best_state = torch.load(
            Path(CKPT_DIR) / "best" / "tensor_conditioner.pt",
            map_location=self.device, weights_only=False,
            )
        self.tensor_conditioner.load_state_dict(best_state)
        self.tensor_conditioner.float()

        # Full-set geometric check on best checkpoint
        self.tensor_conditioner.eval()
        pos_centroid = F.normalize(self.W_positive.mean(0), dim=0)
        neg_centroid = F.normalize(self.W_negative.mean(0), dim=0)
        pos_all, neg_all, norm_all = [], [], []
        with torch.no_grad():
            for emb in self.embeddings:
                proj      = self._project(emb)
                norm_all.append(proj.norm().item())
                proj_norm = F.normalize(proj, dim=0)
                pos_all.append((proj_norm * pos_centroid).sum().item())
                neg_all.append((proj_norm * neg_centroid).sum().item())

        pos_final  = float(np.mean(pos_all))
        neg_final  = float(np.mean(neg_all))
        norm_final = float(np.mean(norm_all))
        gap_final  = pos_final - neg_final

        volume.commit()

        print(f"\n{'='*60}")
        print(f"Training complete.")
        print(f"  Best epoch     : {best_epoch}")
        print(f"  Best gap       : {best_geo_gap:.4f}  (Phase 5 reference: 0.383)")
        print(f"  pos_sim (final): {pos_final:.4f}")
        print(f"  neg_sim (final): {neg_final:.4f}")
        print(f"  gap (final)    : {gap_final:.4f}")
        print(f"  proj_norm      : {norm_final:.2f}  (target ~36.5)")
        print(f"  {'✓ PASS' if gap_final > 0 else '✗ FAIL'}")
        print(f"\nCheckpoint: {CKPT_DIR}/best/tensor_conditioner.pt")
        print(f"\nCheckpoint table (every {LOG_EVERY} epochs):")
        print(f"  {'epoch':>6} | {'proj_norm':>9} | {'pos_sim':>8} | {'neg_sim':>8} | {'gap':>8}")
        print(f"  {'-'*52}")
        for row in checkpoint_table:
            print(f"  {row['epoch']:>6} | {row['proj_norm']:>9.2f} | "
                  f"{row['pos_sim']:>8.4f} | {row['neg_sim']:>8.4f} | {row['gap']:>8.4f}")

        return {
            "status":           "complete",
            "best_epoch":       best_epoch,
            "best_geo_gap":     round(best_geo_gap, 4),
            "pos_final":        round(pos_final, 4),
            "neg_final":        round(neg_final, 4),
            "gap_final":        round(gap_final, 4),
            "proj_norm_final":  round(norm_final, 2),
            "passed":           gap_final > 0,
            "phase5_reference": {"gap": 0.383, "proj_norm": 37.82, "epoch": 104},
            "checkpoint_table": checkpoint_table,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    result = ConditionerRetrainerV3().train.remote()

    print("\n" + "="*60)
    if result["status"] == "norm_stop":
        print("⚠ TRAINING STOPPED — proj_norm exceeded threshold")
        print(f"  Stopped at epoch : {result['stopped_at_epoch']}")
        print(f"  proj_norm        : {result['proj_norm']}")
        print(f"  Threshold        : {result['norm_stop_threshold']}")
        print(f"  → FLAG TO MUSE BEFORE PROCEEDING")
    else:
        print("RETRAINING COMPLETE")
        print("="*60)
        print(f"  Best epoch    : {result['best_epoch']}")
        print(f"  Gap           : {result['gap_final']:.4f}  "
              f"(Phase 5 ref: {result['phase5_reference']['gap']})")
        print(f"  proj_norm     : {result['proj_norm_final']:.2f}  (target ~36.5)")
        print(f"  Passed        : {result['passed']}")
        print()
        print(f"  Checkpoint table (epoch | proj_norm | pos_sim | neg_sim | gap):")
        for row in result["checkpoint_table"]:
            print(f"    {row['epoch']:>6} | {row['proj_norm']:>9.2f} | "
                  f"{row['pos_sim']:>8.4f} | {row['neg_sim']:>8.4f} | {row['gap']:>8.4f}")