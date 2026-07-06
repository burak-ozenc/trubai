"""
trubai_outlier_diagnostic.py
Identify the pathological outlier embedding in diverse_set.

Method (per Muse spec):
1. Seed random with 42 (matches v7/v8 exactly)
2. Replay random calls through epochs 51–54 to advance random state
3. Print the 2 file paths drawn at epoch 55 (k=2, sub-phase 2.1)
4. Run isolated forward-backward for each against the loss function
5. Report gradient norm for each — outlier will be orders of magnitude larger

Usage:
    modal run trubai_outlier_diagnostic.py
"""

import modal
import json
import random
from pathlib import Path

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

app   = modal.App("trubai-outlier-diagnostic", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

CKPT_DIR       = Path("/checkpoints")
HF_DIR         = Path("/hf-cache")
ANCHOR_JSONL   = CKPT_DIR / "trubai_training_pairs.jsonl"
COMM_SAFE_TXT  = CKPT_DIR / "commercial_safe.txt"
NEW_EMB_DIR    = CKPT_DIR / "new_embeddings" / "embeddings"
PHASE5_EMB_DIR = CKPT_DIR / "embeddings"
EMB_CACHE_PATH = CKPT_DIR / "moshi_emb_cache.pt"
V7_START       = CKPT_DIR / "retrain_v7" / "best_phase2.pt"
V5_ANCHOR      = CKPT_DIR / "retrain_v5" / "phase1_anchor.pt"

# Token sets — unchanged from v7/v8
W_POSITIVE_IDS: dict[str, int] = {
    "▁column":    2368, "▁flat":    3077, "▁support":   711, "▁partial":  4107,
    "▁center":    1611, "▁hollow": 19664, "▁pinch":   24657, "▁crack":    6615,
    "▁focus":     1563, "▁breath":  8735, "▁airflow": 28512, "▁tone":     9064,
    "▁sharp":     6064, "▁buzz":   21938, "▁pitch":    6396, "▁air":      1142,
    "▁spreading": 13369,                   "▁aperture": 16252,
}
W_NEGATIVE_IDS: dict[str, int] = {
    "▁disagreement": 17888, "▁cherish":   31329, "▁nationality": 25602,
    "▁tower":         6537, "▁canonical":  8862, "▁iteration":   11024,
    "▁mascot":       27902, "▁brittle":   29915, "▁Tournament":  10653,
    "▁proud":         9630, "▁Az":         9823, "▁major":         916,
    "▁":   260, "▁in": 271, "▁a":  272,
    "▁of": 264, "▁to": 269, "▁and": 267, "▁the": 262,
}

LAMBDA_ANCHOR_PHASE3 = 0.05
LAMBDA_NORM          = 0.1
TARGET_NORM          = 36.5


def get_k_diverse_phase2(epoch: int) -> int:
    sub_phase = (epoch - 51) // 10 + 1
    return int(22 * sub_phase * 0.10)


def build_tensor_conditioner(device: str):
    from moshi.conditioners.tensors import TensorConditioner
    return TensorConditioner(
        dim=768, output_dim=4096, device=device,
        force_linear=True, output_bias=False, learn_padding=True,
    ).to(device)


def tc_forward(tc, emb, W_pos, W_neg):
    import torch
    import torch.nn.functional as F
    from moshi.conditioners import TensorCondition

    mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
    cond = TensorCondition(tensor=emb, mask=mask)
    proj  = tc(cond)[0]
    x     = proj.squeeze(1)
    x_n   = F.normalize(x,     dim=-1)
    pos_n = F.normalize(W_pos, dim=-1)
    neg_n = F.normalize(W_neg, dim=-1)
    return (x_n @ pos_n.T).mean(), (x_n @ neg_n.T).mean(), x.norm()


def compute_anchor_loss(model, W_anchor, lambda_val):
    import torch
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        anchor = W_anchor[name].to(param.device)
        loss   = loss + (param - anchor).pow(2).sum()
    return lambda_val * loss


@app.cls(
    gpu="H100",
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=1800,
)
class Diagnostic:

    @modal.enter()
    def setup(self):
        import os
        os.environ["HF_HOME"] = str(HF_DIR)
        import torch
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders

        self.device = torch.device("cuda")

        print("Loading LM head weights...")
        lm_head = torch.load(EMB_CACHE_PATH, map_location="cpu", weights_only=True)
        self.W_pos = lm_head[list(W_POSITIVE_IDS.values())].to(self.device)
        self.W_neg = lm_head[list(W_NEGATIVE_IDS.values())].to(self.device)
        print(f"  W_pos={self.W_pos.shape}, W_neg={self.W_neg.shape}")
        print("Setup complete.")

    @modal.method()
    def run(self) -> None:
        import torch

        # ── Load datasets ─────────────────────────────────────────────────────
        anchor_set, diverse_set = self._load_datasets()

        # ── Replay random sequence from seed=42 through epochs 51–54 ─────────
        # Must exactly mirror the training loop's random calls:
        #   each epoch: random.sample(diverse_set, k) + random.shuffle(epoch_data)
        # Epoch 55 (sub-phase 2.1, k=2) is the first draw that caused the crash.
        random.seed(42)

        for epoch in range(51, 55):   # replay epochs 51, 52, 53, 54
            k = get_k_diverse_phase2(epoch)
            sampled = random.sample(diverse_set, k)
            epoch_data = list(anchor_set) + sampled
            random.shuffle(epoch_data)

        # Now at exactly the random state before epoch 55's draw
        k_ep55 = get_k_diverse_phase2(55)   # = 2
        epoch55_diverse = random.sample(diverse_set, k_ep55)

        print()
        print("=" * 65)
        print("EPOCH 55 DIVERSE DRAW (seed=42, k=2)")
        print("=" * 65)
        for i, rec in enumerate(epoch55_diverse):
            print(f"  [{i}] {rec['_emb_path']}")
        print()

        # ── Load checkpoints ──────────────────────────────────────────────────
        tc = build_tensor_conditioner(str(self.device))
        tc.load_state_dict(
            torch.load(V7_START, map_location=self.device, weights_only=True)
        )
        tc.train()

        W_anchor = {
            k: v.to(self.device)
            for k, v in torch.load(
                V5_ANCHOR, map_location=self.device, weights_only=True
            ).items()
        }

        # ── Isolated gradient norm per diverse embedding ───────────────────────
        print("Isolated gradient norm measurement:")
        print("(Each embedding gets its own zero_grad → forward → backward)")
        print()

        for i, rec in enumerate(epoch55_diverse):
            emb = self._load_embedding(rec).to(self.device)

            tc.zero_grad()
            pos_sim, neg_sim, proj_norm = tc_forward(tc, emb, self.W_pos, self.W_neg)
            L_c = -(pos_sim - neg_sim)
            L_n = LAMBDA_NORM * (proj_norm - TARGET_NORM) ** 2
            L_a = compute_anchor_loss(tc, W_anchor, LAMBDA_ANCHOR_PHASE3)
            loss = L_c + L_n + L_a
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                tc.parameters(), max_norm=float('inf')   # measure, don't clip
            ).item()

            print(f"  [{i}] {rec['_emb_path'].name}")
            print(f"       pos_sim={pos_sim.item():.4f}  neg_sim={neg_sim.item():.4f}  "
                  f"proj_norm={proj_norm.item():.2f}")
            print(f"       loss={loss.item():.4f}  grad_norm={grad_norm:.2f}")
            print()

        # ── Also check epoch 131's first diverse draw (v8 first crash) ────────
        # Advance random state past epoch 55's shuffle and epochs 56–130
        epoch55_data = list(anchor_set) + epoch55_diverse
        random.shuffle(epoch55_data)

        for epoch in range(56, 131):
            k = get_k_diverse_phase2(epoch)
            sampled = random.sample(diverse_set, k)
            epoch_data = list(anchor_set) + sampled
            random.shuffle(epoch_data)

        # Epoch 131: Phase 3, k=500
        epoch131_diverse = random.sample(diverse_set, 500)

        print("=" * 65)
        print("EPOCH 131 DIVERSE DRAW — gradient norm scan (k=500)")
        print("Scanning for any embedding with grad_norm > 100")
        print("=" * 65)

        outliers_131 = []
        for idx, rec in enumerate(epoch131_diverse):
            emb = self._load_embedding(rec).to(self.device)
            tc.zero_grad()
            pos_sim, neg_sim, proj_norm = tc_forward(tc, emb, self.W_pos, self.W_neg)
            L_c = -(pos_sim - neg_sim)
            L_n = LAMBDA_NORM * (proj_norm - TARGET_NORM) ** 2
            L_a = compute_anchor_loss(tc, W_anchor, LAMBDA_ANCHOR_PHASE3)
            loss = L_c + L_n + L_a
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                tc.parameters(), max_norm=float('inf')
            ).item()
            if gn > 100:
                outliers_131.append((idx, rec['_emb_path'], gn))
                print(f"  OUTLIER [{idx}] {rec['_emb_path'].name}  grad_norm={gn:.2f}")

        if not outliers_131:
            print("  No outliers above 100 found in epoch 131 draw.")
        else:
            print(f"\n  Total outliers found: {len(outliers_131)}")

        print()
        print("=" * 65)
        print("Report full output to Muse before v9 begins.")
        print("=" * 65)

    def _load_datasets(self) -> tuple[list[dict], list[dict]]:
        import re

        def normalize_pitch(s: str) -> str:
            return re.sub(r'Gs(\d)', r'G#\1', s)

        anchor_records = [json.loads(l) for l in open(ANCHOR_JSONL)]
        for rec in anchor_records:
            stem = Path(rec["embedding_path"]).stem
            rec["_emb_path"] = PHASE5_EMB_DIR / f"{stem}.pt"
        assert len(anchor_records) == 22
        print(f"Anchor set:  {len(anchor_records)} pairs ✓")

        stems_raw = [
            Path(line.strip()).stem
            for line in open(COMM_SAFE_TXT) if line.strip()
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
            f"Expected 3181 diverse records, got {len(diverse_records)}"
        )
        print(f"Diverse set: {len(diverse_records)} pairs ({miss} unmatched) ✓")
        return anchor_records, diverse_records

    def _load_embedding(self, record: dict):
        import torch
        emb = torch.load(str(record["_emb_path"]), map_location="cpu",
                         weights_only=True)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        return emb.float()


@app.local_entrypoint()
def main() -> None:
    Diagnostic().run.remote()