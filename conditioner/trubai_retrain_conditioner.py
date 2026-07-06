# TRUB.AI — TensorConditioner Retraining, Phase 1
# Contrastive + diversity loss against LM head weight rows.
# No CE loss. No LoRA. No Moshi forward pass during training.
#
# Scholar spec:
#   - Pull projection toward LM head rows for trumpet vocabulary tokens
#   - Push projection away from LM head rows for attractor tokens
#   - Diversity penalty across virtual batch of 8 to fix embedding collapse
#   - Validate geometrically before running streaming session
#
# After this passes geometric check → Phase 2: retrain LoRA from scratch.
# ─────────────────────────────────────────────────────────────────────────────

import modal
import numpy as np
import json
import typing as tp
from pathlib import Path
from dataclasses import dataclass

app = modal.App("trubai-retrain-conditioner")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg"])
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
# Config
# ─────────────────────────────────────────────────────────────────────────────

JSONL_PATH     = "/checkpoints/trubai_training_pairs.jsonl"
EMBEDDINGS_DIR = "/checkpoints/embeddings"
CKPT_DIR       = "/checkpoints/retrain_v2"

MERT_DIM   = 768
MOSHI_DIM  = 4096
EPOCHS     = 200
LR         = 3e-4
WEIGHT_DECAY = 0.01
LAMBDA_CONTRASTIVE = 0.7   # weight of contrastive loss
LAMBDA_DIVERSITY   = 0.3   # weight of diversity loss
MARGIN     = 0.3           # attractor push margin
VIRTUAL_BATCH = 8          # pairs per diversity computation
LOG_EVERY  = 20            # epochs between prints
CKPT_EVERY = 20

# Trumpet vocabulary — content words a teacher uses, no articles
# Space prefix (▁) is how SentencePiece encodes word-initial subwords
TRUMPET_WORDS = [
    " air", " column", " tone", " breath", " aperture",
    " center", " embouchure", " cracked", " flat", " sharp",
    " breathy", " pitch", " partial", " buzz", " focus",
    " pinched", " spreading", " hollow", " airflow", " support",
]

# Attractor words confirmed from diagnostic (both with and without LoRA)
ATTRACTOR_WORDS = [
    " proud", " brittle", " ville", " nationality",
    " Tournament", " disagreement", " Az", " tower",
    " mascot", " cherish", " canonical", " iteration",
]


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
    timeout=3600,
)
class ConditionerRetrainer:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = "/hf-cache"

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners.tensors import TensorConditioner
        from moshi.conditioners import (
            TensorCondition, ConditionFuser,
            ConditionProvider, ConditionAttributes,
        )

        self.device = "cuda"

        # ── Load base Moshi (text_linear.weight extraction only) ──────────────
        mimi_weight  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        base_model = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)

        # ── Extract LM head weights — frozen for entire training ──────────────
        # text_linear: Linear(4096 → 32000), weight shape [32000, 4096]
        self.lm_head_weights = base_model.text_linear.weight.detach().float()
        # [32000, 4096] on cuda
        print(f"LM head weights: {self.lm_head_weights.shape}")

        # ── Text tokenizer — needed to map words → token ids ──────────────────
        text_tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(text_tok_path)

        # ── Build positive / negative weight matrices ─────────────────────────
        def words_to_weight_rows(words: list) -> "torch.Tensor":
            ids = []
            for w in words:
                # SentencePiece uses ▁ (U+2581) as word-initial prefix.
                # piece_to_id() returns 0 (unk) if the piece doesn't exist.
                # Try ▁word first, fall back to encoding and taking first token.
                w_stripped = w.strip()
                piece_id = self.text_tokenizer.piece_to_id(f"▁{w_stripped}")
                if piece_id > 0:
                    ids.append(piece_id)
                else:
                    # Fallback: encode and take the first non-zero token
                    toks = self.text_tokenizer.encode(w_stripped)
                    for t in toks:
                        if 0 < t < self.lm_head_weights.shape[0]:
                            ids.append(t)
                            break
            ids = list(set(ids) - {0, 260})  # deduplicate, remove unk and bare ▁
            resolved = [(self.text_tokenizer.id_to_piece(i), i) for i in ids]
            print(f"  → {len(ids)} unique token ids from {len(words)} words:")
            for piece, pid in resolved:
                print(f"      '{piece}' → {pid}")
            return self.lm_head_weights[ids]  # [N, 4096]

        print("Trumpet token ids:")
        self.W_positive = words_to_weight_rows(TRUMPET_WORDS)   # [~20, 4096]
        print("Attractor token ids:")
        self.W_negative = words_to_weight_rows(ATTRACTOR_WORDS) # [~12, 4096]

        # L2-normalize for cosine similarity
        import torch.nn.functional as F
        self.W_pos_normed = F.normalize(self.W_positive, dim=1)  # [P, 4096]
        self.W_neg_normed = F.normalize(self.W_negative, dim=1)  # [N, 4096]

        print(f"W_positive: {self.W_positive.shape} | W_negative: {self.W_negative.shape}")

        # ── TensorConditioner — the only thing we're training ─────────────────
        self.tensor_conditioner = TensorConditioner(
            dim=768, output_dim=4096, device=self.device,
            force_linear=True, output_bias=False, learn_padding=True,
        ).to(self.device).float()
        # Note: train in float32 for numerical stability; cast to bfloat16 when saving

        # ── Load all 22 embeddings into memory ────────────────────────────────
        with open(JSONL_PATH) as f:
            self.records = [json.loads(l) for l in f]

        self.embeddings = []  # list of [1,1,768] float32 tensors on cuda
        for record in self.records:
            name = Path(record["embedding_path"]).name
            emb  = torch.load(
                str(Path(EMBEDDINGS_DIR) / name),
                map_location=self.device, weights_only=False,
            ).float()  # [1,1,768]
            self.embeddings.append(emb)

        print(f"Loaded {len(self.embeddings)} embeddings ✓")
        print("Setup complete. Ready to train.")

    # ── Projection helper ──────────────────────────────────────────────────────

    def _project(self, emb):
        """
        Wrap raw [1,1,768] embedding in TensorCondition and run TensorConditioner forward.
        Returns [4096] float32 tensor.
        TensorConditioner.forward expects a TensorCondition object, not a raw tensor.
        """
        import torch
        from moshi.conditioners import TensorCondition
        mask = torch.ones(1, 1, dtype=torch.bool, device=self.device)
        tc   = TensorCondition(tensor=emb, mask=mask)
        out  = self.tensor_conditioner(tc)   # ConditionType(condition, mask)
        return out.condition.squeeze()        # [4096]

    # ── Loss functions ─────────────────────────────────────────────────────────

    def _contrastive_loss(self, projection):
        """
        projection: [4096] float32
        Pull toward trumpet LM head rows, push away from attractor rows.
        Norm penalty (λ=0.1) anchors output scale to original checkpoint (~36.5).
        """
        import torch, torch.nn.functional as F

        proj = projection  # [4096]

        # Output norm constraint — anchor to original projection output norm
        L_norm = (proj.norm() - 36.5) ** 2

        proj_norm = F.normalize(proj.unsqueeze(0), dim=1)  # [1, 4096]

        # Pull: maximize mean cosine similarity to trumpet vocabulary
        pos_sim = (proj_norm @ self.W_pos_normed.T).squeeze(0)  # [P]
        L_pull  = -pos_sim.mean()

        # Push: penalize attractor similarity above -margin
        neg_sim = (proj_norm @ self.W_neg_normed.T).squeeze(0)  # [N]
        L_push  = torch.relu(neg_sim + MARGIN).mean()

        return L_pull + L_push + 0.1 * L_norm

    def _diversity_loss(self, projections):
        """
        projections: [B, 4096] float32
        Penalize off-diagonal pairwise cosine similarity.
        B = virtual batch size (8).
        """
        import torch
        import torch.nn.functional as F
        norms      = F.normalize(projections, dim=1)         # [B, 4096]
        sim_matrix = norms @ norms.T                         # [B, B]
        B          = projections.shape[0]
        mask       = ~torch.eye(B, dtype=torch.bool, device=self.device)
        return sim_matrix[mask].mean()

    # ── Geometric validation ───────────────────────────────────────────────────

    def _geometric_check(self):
        """
        For each embedding, compute projection and measure:
          - mean cosine sim to W_positive centroid
          - mean cosine sim to W_negative centroid
        Returns (pos_mean, neg_mean, passed: bool)
        """
        import torch
        import torch.nn.functional as F

        self.tensor_conditioner.eval()
        pos_sims, neg_sims, proj_norms = [], [], []
        pos_centroid = F.normalize(self.W_positive.mean(0), dim=0)  # [4096]
        neg_centroid = F.normalize(self.W_negative.mean(0), dim=0)  # [4096]

        with torch.no_grad():
            for emb in self.embeddings:
                proj = self._project(emb)            # [4096]
                proj_norms.append(proj.norm().item())
                proj_norm = F.normalize(proj, dim=0) # [4096]
                pos_sims.append((proj_norm * pos_centroid).sum().item())
                neg_sims.append((proj_norm * neg_centroid).sum().item())

        pos_mean  = float(np.mean(pos_sims))
        neg_mean  = float(np.mean(neg_sims))
        norm_mean = float(np.mean(proj_norms))
        passed    = pos_mean > neg_mean

        self.tensor_conditioner.train()
        return pos_mean, neg_mean, norm_mean, passed

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

        best_geo_gap   = -999.0   # pos_mean - neg_mean, higher is better
        best_epoch     = -1
        geo_history    = []
        loss_history   = []

        print(f"\nStarting TensorConditioner retraining")
        print(f"  Epochs: {EPOCHS} | LR: {LR} | λ_contrastive: {LAMBDA_CONTRASTIVE} | λ_diversity: {LAMBDA_DIVERSITY}")
        print(f"  W_positive: {self.W_positive.shape[0]} tokens | W_negative: {self.W_negative.shape[0]} tokens\n")

        for epoch in range(1, EPOCHS + 1):
            self.tensor_conditioner.train()
            indices = list(range(len(self.embeddings)))
            random.shuffle(indices)

            epoch_losses = []

            # Virtual batches of VIRTUAL_BATCH for diversity loss
            for batch_start in range(0, len(indices), VIRTUAL_BATCH):
                batch_idx  = indices[batch_start : batch_start + VIRTUAL_BATCH]
                batch_embs = [self.embeddings[i] for i in batch_idx]

                # Forward: all embeddings in virtual batch
                projections = []
                for emb in batch_embs:
                    proj = self._project(emb)  # [4096]
                    projections.append(proj)

                projections_stacked = torch.stack(projections, dim=0)  # [B, 4096]

                # Contrastive loss: mean over batch
                L_contrastive = torch.stack([
                    self._contrastive_loss(p) for p in projections
                ]).mean()

                # Diversity loss: across virtual batch
                L_diversity = self._diversity_loss(projections_stacked)

                loss = LAMBDA_CONTRASTIVE * L_contrastive + LAMBDA_DIVERSITY * L_diversity

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.tensor_conditioner.parameters(), max_norm=1.0
                )
                optimizer.step()

                epoch_losses.append(loss.item())

            mean_loss = float(np.mean(epoch_losses))
            loss_history.append(round(mean_loss, 4))

            # Geometric check every epoch
            pos_mean, neg_mean, norm_mean, passed = self._geometric_check()
            geo_gap = pos_mean - neg_mean
            geo_history.append({
                "epoch":     epoch,
                "pos_mean":  round(pos_mean, 4),
                "neg_mean":  round(neg_mean, 4),
                "gap":       round(geo_gap, 4),
                "proj_norm": round(norm_mean, 2),
                "passed":    passed,
            })

            if epoch % LOG_EVERY == 0 or epoch == 1:
                print(f"Epoch {epoch:4d}/{EPOCHS} | loss: {mean_loss:.4f} | "
                      f"pos: {pos_mean:.4f} | neg: {neg_mean:.4f} | "
                      f"gap: {geo_gap:.4f} | proj_norm: {norm_mean:.1f} | "
                      f"{'✓' if passed else '✗'}")

            # Save best checkpoint by geometric gap
            if geo_gap > best_geo_gap:
                best_geo_gap = geo_gap
                best_epoch   = epoch
                best_path    = Path(CKPT_DIR) / "best"
                best_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.tensor_conditioner.to(torch.bfloat16).state_dict(),
                    best_path / "tensor_conditioner.pt",
                    )
                self.tensor_conditioner.float()  # back to float32 for training

            # Periodic checkpoint
            if epoch % CKPT_EVERY == 0:
                ckpt_path = Path(CKPT_DIR) / f"epoch_{epoch:04d}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                torch.save(
                    self.tensor_conditioner.to(torch.bfloat16).state_dict(),
                    ckpt_path / "tensor_conditioner.pt",
                    )
                self.tensor_conditioner.float()
                volume.commit()  # flush to Modal volume
                print(f"  → Checkpoint saved: epoch_{epoch:04d}")

        # Final geometric check with best checkpoint
        print(f"\n{'='*60}")
        print(f"Training complete.")
        print(f"Best epoch: {best_epoch} | Best geo gap: {best_geo_gap:.4f}")

        # Reload best for final report
        best_state = torch.load(
            Path(CKPT_DIR) / "best" / "tensor_conditioner.pt",
            map_location=self.device, weights_only=False,
            )
        self.tensor_conditioner.load_state_dict(best_state)
        self.tensor_conditioner.float()

        pos_final, neg_final, norm_final, passed_final = self._geometric_check()
        print(f"Final geometric check (best ckpt):")
        print(f"  pos_sim (trumpet):   {pos_final:.4f}")
        print(f"  neg_sim (attractor): {neg_final:.4f}")
        print(f"  gap:                 {pos_final - neg_final:.4f}")
        print(f"  proj_norm (mean):    {norm_final:.2f}  (target: ~36.5)")
        if norm_final > 60:
            print(f"  ⚠ proj_norm still high — consider increasing λ_norm to 0.3")
        print(f"  {'✓ PASS — proceed to Phase 2 LoRA retraining' if passed_final else '✗ FAIL — do not proceed, report to Scholar'}")

        # Per-embedding breakdown for Scholar
        print(f"\nPer-embedding similarities (best checkpoint):")
        import torch.nn.functional as F
        pos_centroid = F.normalize(self.W_positive.mean(0), dim=0)
        neg_centroid = F.normalize(self.W_negative.mean(0), dim=0)
        self.tensor_conditioner.eval()
        with torch.no_grad():
            for i, (emb, record) in enumerate(zip(self.embeddings, self.records)):
                proj      = self._project(emb)              # [4096]
                proj_norm = F.normalize(proj, dim=0)         # [4096]
                ps = (proj_norm * pos_centroid).sum().item()
                ns = (proj_norm * neg_centroid).sum().item()
                obs = record.get("observation", {})
                print(f"  [{i:2d}] {obs.get('tone_quality','?'):18s} "
                      f"{obs.get('note','?'):4s} {obs.get('register','?'):7s} | "
                      f"pos={ps:.4f} neg={ns:.4f} gap={ps-ns:.4f}")

        volume.commit()

        return {
            "best_epoch":     best_epoch,
            "best_geo_gap":   round(best_geo_gap, 4),
            "pos_final":      round(pos_final, 4),
            "neg_final":      round(neg_final, 4),
            "proj_norm_final": round(norm_final, 2),
            "passed":         passed_final,
            "geo_history":    geo_history,
            "loss_history":   loss_history,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    result = ConditionerRetrainer().train.remote()

    print("\n" + "="*60)
    print("RETRAINING COMPLETE")
    print("="*60)
    print(f"Best epoch:   {result['best_epoch']}")
    print(f"Geo gap:      {result['best_geo_gap']} (pos - neg cosine sim)")
    print(f"pos_sim:      {result['pos_final']}")
    print(f"neg_sim:      {result['neg_final']}")
    print(f"Passed:       {result['passed']}")

    print("\nGeo history (every logged epoch):")
    for entry in result["geo_history"]:
        if entry["epoch"] % 20 == 0 or entry["epoch"] == 1:
            print(f"  epoch {entry['epoch']:4d} | gap={entry['gap']:+.4f} | "
                  f"pos={entry['pos_mean']:.4f} neg={entry['neg_mean']:.4f} "
                  f"{'✓' if entry['passed'] else '✗'}")

    if result["passed"]:
        print("\n✓ Geometric check passed.")
        print("  Next step: run Phase 2 — retrain LoRA from scratch against corrected TensorConditioner.")
        print(f"  Best checkpoint: /checkpoints/retrain_v2/best/tensor_conditioner.pt")
    else:
        print("\n✗ Geometric check failed. Report to Scholar before proceeding.")