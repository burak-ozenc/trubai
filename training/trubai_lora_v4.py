"""
trubai_lora_v4.py
SPEC-LORA-v4 — LoRA Retraining on Expanded Dataset (4,815 pairs)

Dataset:      pairs_train.json (4,522 pairs) / pairs_eval.json (293 pairs)
Architecture: identical to lora_v3 — rank 8, last 4 layers, Q/K/V/O
Objective:    CE loss + margin loss (KL removed — ineffective in v3)
Optimizer:    AdamW lr=5e-5 wd=0.1, optimizer state saved at every gate
Gates:        epochs 20, 40, 60, 80, 100 — stop and report at each
Plateau:      stop if eval_ce does not improve ≥0.02 across 3 consecutive gates
Divergence:   stop if eval_ce > 6.0 at any gate (excluding epoch 20)

High register oversampling: HIGH_REGISTER_WEIGHT = 2.5
Track A pairs (register=None): weight = 1.0

Resume:
    modal run trubai_lora_v4.py::main --resume-epoch 20

Faber does not continue past any gate without Muse confirmation.
"""

import modal
import json
import random
from pathlib import Path

MOSHI_FORK_URL = "git+https://github.com/burak-ozenc/moshi.git#subdirectory=moshi"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["git", "ffmpeg"])
    .pip_install([
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers==4.46.3",
        "huggingface_hub==0.26.5",
        "sentencepiece",
        "safetensors",
        "peft",
    ])
    .pip_install(MOSHI_FORK_URL, force_build=True)
)

app      = modal.App("trubai-lora-v4", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)
data_vol = modal.Volume.from_name("trubai-data",         create_if_missing=True)

CKPT_DIR    = Path("/checkpoints")
HF_DIR      = Path("/hf-cache")
DATA_DIR    = Path("/data/data")   # volume stores files under data/ prefix
V13_CKPT    = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_V4_DIR = CKPT_DIR / "lora_v4"

TRAIN_JSON = DATA_DIR / "pairs_train.json"
EVAL_JSON  = DATA_DIR / "pairs_eval.json"
EMB_DIR    = CKPT_DIR / "embeddings_v4"   # MERT embeddings cached per pair

# ── Hyperparameters ────────────────────────────────────────────────────────────
LORA_RANK            = 8
LORA_ALPHA           = 16
LORA_LR              = 5e-5
LORA_WD              = 0.1
EPOCHS               = 100
MARGIN               = 2.0
GATE_EPOCHS          = {20, 40, 60, 80, 100}
FREE_GEN_TOKENS      = 3
HIGH_REGISTER_WEIGHT = 2.5   # §2 oversampling

HARD_NEGATIVES = [" Gmina", " pgfplots", " martingale"]

# Plateau / divergence criteria (§4)
PLATEAU_MIN_IMPROVE  = 0.02   # minimum improvement to not count as no-improve
PLATEAU_CONSECUTIVE  = 3      # gates without improvement → stop
DIVERGENCE_THRESHOLD = 6.0    # eval_ce above this → stop immediately


def build_conditioner_v12(device: str):
    import torch, torch.nn as nn, torch.nn.functional as F
    from moshi.conditioners.tensors import TensorConditioner
    from moshi.conditioners import TensorCondition

    class ConditionerV12(nn.Module):
        def __init__(self):
            super().__init__()
            self.tc = TensorConditioner(
                dim=768, output_dim=4096, device=device,
                force_linear=True, output_bias=False, learn_padding=True,
            )
            self.register_buffer("output_scale", torch.tensor(36.5))

        def forward(self, emb):
            mask = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            cond = TensorCondition(tensor=emb, mask=mask)
            proj = self.tc(cond)[0]
            return F.normalize(proj.squeeze(1), dim=-1), self.output_scale

        def condition_sum_vector(self, emb):
            direction, scale = self.forward(emb)
            return direction * scale.detach()

    return ConditionerV12().to(device)


@app.cls(
    gpu="H100",
    volumes={
        "/checkpoints": vol,
        "/hf-cache":    hf_cache,
        "/data":        data_vol,
    },
    timeout=86400,   # 24h — embedding extraction + full gate epoch
)
class LoRATrainV4:

    @modal.enter()
    def setup(self):
        import torch._dynamo
        torch._dynamo.config.disable = True

        import os
        os.environ["HF_HOME"] = str(HF_DIR)

        import torch, sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from moshi.conditioners import ConditionProvider, ConditionFuser
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModel

        self.device = torch.device("cuda")

        # ── Tokenizer ─────────────────────────────────────────────────────────
        tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp  = sentencepiece.SentencePieceProcessor(tok_path)

        # ── Moshi LM ──────────────────────────────────────────────────────────
        print("Loading Moshi LM...")
        moshi_weight  = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm      = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )
        self._moshi_weight = moshi_weight

        # ── ConditionerV12 — frozen ────────────────────────────────────────────
        print("Loading v13 conditioner...")
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False
        frozen = sum(p.numel() for p in self.conditioner.parameters())
        print(f"  ConditionerV12 frozen ✓ ({frozen:,} params)")

        # ── ConditionProvider / Fuser ──────────────────────────────────────────
        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

        # ── LoRA ──────────────────────────────────────────────────────────────
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        self.moshi_model = get_peft_model(moshi_lm, lora_config)
        self.moshi_model.print_trainable_parameters()
        base = self.moshi_model.get_base_model()
        base.condition_provider = self.cp
        base.fuser               = self.fuser

        # ── MERT (for embedding extraction) ───────────────────────────────────
        print("Loading MERT...")
        self.mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True, cache_dir=str(HF_DIR),
        ).to(self.device).eval()

        # ── Hard negatives ─────────────────────────────────────────────────────
        self.hard_neg_ids = [self.sp.encode(s) for s in HARD_NEGATIVES]

        # ── Reference LM head for KL (kept for parity — KL weight = 0) ────────
        # KL was ineffective in v3 (KL=0.0 throughout). Excluded from loss.
        # Reference weights not loaded — saves memory.

        # ── Load dataset JSON only — embeddings loaded/extracted separately ──────
        print("Loading dataset JSON...")
        with open(TRAIN_JSON) as f:
            self.train_raw = json.load(f)
        with open(EVAL_JSON) as f:
            self.eval_raw = json.load(f)
        print(f"  Train: {len(self.train_raw)} pairs | Eval: {len(self.eval_raw)} pairs")

        # ── Weighted sampler weights (by register field) ───────────────────────
        self.train_weights = [
            HIGH_REGISTER_WEIGHT if p.get("register") == "High" else 1.0
            for p in self.train_raw
        ]
        n_high = sum(1 for w in self.train_weights if w == HIGH_REGISTER_WEIGHT)
        print(f"  High register pairs: {n_high} (weight={HIGH_REGISTER_WEIGHT})")

        # train_data / eval_data populated by extract_embeddings() or _ensure_embeddings()
        self.train_data = None
        self.eval_data  = None

        print("Setup complete.")

    def _load_or_extract(self, raw_pairs: list, split_name: str) -> list:
        """Load cached embeddings or extract via MERT. Returns enriched list."""
        import torch, torchaudio
        import os

        result = []
        n_extracted = 0
        n_cached    = 0

        for pair in raw_pairs:
            stem     = Path(pair["file"]).stem
            emb_path = EMB_DIR / f"{stem}.pt"

            if emb_path.exists():
                emb = torch.load(str(emb_path), map_location="cpu", weights_only=False)
                n_cached += 1
            else:
                # Audio file expected at /data/audio/<filename>
                audio_path = DATA_DIR / "audio" / pair["file"]
                if not audio_path.exists():
                    print(f"  WARNING: audio not found: {audio_path} — skipping")
                    continue

                wav, sr = torchaudio.load(str(audio_path))
                if sr != 24000:
                    wav = torchaudio.functional.resample(wav, sr, 24000)
                wav = wav.mean(0).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out = self.mert(wav.float())
                    emb = out.last_hidden_state.mean(1, keepdim=True).cpu()  # [1,1,768]

                torch.save(emb, str(emb_path))
                n_extracted += 1

            ids = self.sp.encode(pair["inner_monologue"])
            result.append({
                "emb":      emb.float(),
                "ids":      ids,
                "text":     pair["inner_monologue"],
                "label":    pair.get("label"),
                "register": pair.get("register"),
            })

        print(f"  {split_name}: {len(result)} pairs loaded "
              f"({n_cached} cached, {n_extracted} extracted)")
        if n_extracted > 0:
            vol.commit()
        return result

    @modal.method()
    def train(self, resume_epoch: int = 0) -> None:
        import torch
        import random as _random

        # Load embeddings from cache (extraction must have run first)
        self._ensure_embeddings()

        LORA_V4_DIR.mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.moshi_model.parameters() if p.requires_grad],
            lr=LORA_LR, weight_decay=LORA_WD,
        )

        # ── Resume bug fix: restore optimizer state ────────────────────────────
        if resume_epoch > 0:
            adapter_path = LORA_V4_DIR / f"epoch_{resume_epoch:03d}"
            opt_path     = LORA_V4_DIR / f"optimizer_{resume_epoch:03d}.pt"
            if not adapter_path.exists():
                raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")
            self.moshi_model.load_adapter(str(adapter_path), adapter_name="default")
            print(f"Adapter restored from epoch {resume_epoch} ✓")
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(str(opt_path), map_location=self.device, weights_only=True)
                )
                print(f"Optimizer state restored from {opt_path} ✓")
            else:
                print(f"WARNING: optimizer state not found at {opt_path}")
                print(f"  CE spike expected on first epoch — report to Muse immediately.")

        start_epoch      = resume_epoch + 1 if resume_epoch > 0 else 1
        best_eval_ce     = float("inf")
        best_eval_epoch  = 0
        no_improve_gates = 0   # consecutive gates without ≥0.02 improvement

        print(f"Training epochs {start_epoch}–{EPOCHS}  |  gates: {sorted(GATE_EPOCHS)}")
        print(f"HIGH_REGISTER_WEIGHT={HIGH_REGISTER_WEIGHT}  |  "
              f"PLATEAU_CONSECUTIVE={PLATEAU_CONSECUTIVE}")
        print("=" * 70)

        for epoch in range(start_epoch, EPOCHS + 1):
            self.moshi_model.train()

            # Weighted sampling — sample indices with replacement per epoch
            indices = _random.choices(
                range(len(self.train_data)),
                weights=self.train_weights,
                k=len(self.train_data),
            )

            epoch_ce = 0.0
            epoch_mg = 0.0

            for idx in indices:
                item = self.train_data[idx]
                optimizer.zero_grad()

                emb      = item["emb"].to(self.device)
                cond_vec = self.conditioner.condition_sum_vector(
                    emb.to(torch.bfloat16)
                ).detach()

                L_ce = self._compute_ce(item["ids"], cond_vec)

                # Margin loss — hard negatives
                L_mg = torch.tensor(0.0, device=self.device)
                for neg_ids in self.hard_neg_ids:
                    L_ce_neg = self._compute_ce(neg_ids, cond_vec)
                    L_mg += torch.clamp(MARGIN - (L_ce_neg - L_ce), min=0.0)
                L_mg = L_mg / max(len(self.hard_neg_ids), 1)

                loss = L_ce + L_mg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.moshi_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

                epoch_ce += L_ce.item()
                epoch_mg += L_mg.item()

            n = len(indices)
            print(f"Epoch {epoch:3d} | CE={epoch_ce/n:.4f} | Mg={epoch_mg/n:.4f}")

            if epoch in GATE_EPOCHS:
                # ── Eval ──────────────────────────────────────────────────────
                eval_ce = self._eval_ce()

                # ── Save adapter + optimizer state (mandatory at every gate) ──
                gate_path = LORA_V4_DIR / f"epoch_{epoch:03d}"
                self.moshi_model.save_pretrained(str(gate_path))
                torch.save(
                    optimizer.state_dict(),
                    LORA_V4_DIR / f"optimizer_{epoch:03d}.pt",
                    )

                # ── Best checkpoint ────────────────────────────────────────────
                improved = eval_ce < best_eval_ce - PLATEAU_MIN_IMPROVE
                if eval_ce < best_eval_ce:
                    best_eval_ce    = eval_ce
                    best_eval_epoch = epoch
                    self.moshi_model.save_pretrained(str(LORA_V4_DIR / "best"))

                vol.commit()

                # ── Plateau tracking ───────────────────────────────────────────
                if improved:
                    no_improve_gates = 0
                else:
                    no_improve_gates += 1

                plateau_str = (
                    "N/A" if epoch == 20 else
                    f"{no_improve_gates} consecutive no-improve"
                    if no_improve_gates < PLATEAU_CONSECUTIVE
                    else "STOP"
                )

                # ── Free-gen gate sample ───────────────────────────────────────
                sample = self._free_gen_sample()

                # ── Gate report (bring to Muse) ────────────────────────────────
                print()
                print("=" * 70)
                print(f"GATE REPORT — EPOCH {epoch} — BRING TO MUSE")
                print("=" * 70)
                print(f"  Epoch:               {epoch}")
                print(f"  train_ce:            {epoch_ce/n:.4f}")
                print(f"  eval_ce:             {eval_ce:.4f}")
                print(f"  margin_loss:         {epoch_mg/n:.4f}")
                print(f"  Best eval_ce so far: {best_eval_ce:.4f} (epoch {best_eval_epoch})")
                print(f"  Optimizer state:     ✓ saved")
                print(f"  Plateau status:      {plateau_str}")
                print(f"  Free-gen sample:     {sample}")
                print("=" * 70)
                print()
                print("STOP — awaiting Muse confirmation before continuing.")
                print("Resume: modal run trubai_lora_v4.py::main "
                      f"--resume-epoch {epoch}")
                print("=" * 70)

                # ── Stop conditions ────────────────────────────────────────────
                if no_improve_gates >= PLATEAU_CONSECUTIVE and epoch > 20:
                    print(f"\nPLATEAU: eval_ce has not improved ≥{PLATEAU_MIN_IMPROVE} "
                          f"across {PLATEAU_CONSECUTIVE} consecutive gates.")
                    print("Stopping. Report plateau to Muse.")
                    return

                if eval_ce > DIVERGENCE_THRESHOLD and epoch > 20:
                    print(f"\nDIVERGENCE: eval_ce={eval_ce:.4f} exceeds "
                          f"threshold {DIVERGENCE_THRESHOLD}.")
                    print("Stopping immediately. Report to Muse.")
                    return

                # Gate hard stop — the run stops here and requires manual resume
                # after Muse confirmation. This is enforced by the design:
                # each gate saves state cleanly, then the run exits.
                return

        print()
        print("=" * 70)
        print(f"Training complete. Best eval_ce = {best_eval_ce:.4f} "
              f"(epoch {best_eval_epoch})")

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_ce(self, token_ids: list, cond_vec=None) -> "torch.Tensor":
        import torch, torch.nn.functional as F
        if len(token_ids) < 2:
            return torch.tensor(0.0, device=self.device)
        base       = self.moshi_model.get_base_model()
        vocab_size = base.emb[0].weight.shape[0]
        ids_t      = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        ids_t      = ids_t.clamp(0, vocab_size - 1)
        x = base.emb[0](ids_t[:-1]).unsqueeze(0).to(torch.bfloat16)
        if cond_vec is not None:
            x = x + cond_vec.unsqueeze(1).to(x.dtype)
        h      = base.transformer(x)
        logits = base.text_linear(h).squeeze(0).float()
        return F.cross_entropy(logits, ids_t[1:])

    def _eval_ce(self) -> float:
        import torch
        self.moshi_model.eval()
        total = 0.0
        with torch.no_grad():
            for item in self.eval_data:
                emb      = item["emb"].to(self.device)
                cond_vec = self.conditioner.condition_sum_vector(emb.to(torch.bfloat16))
                loss     = self._compute_ce(item["ids"], cond_vec)
                total   += loss.item()
        self.moshi_model.train()
        return total / max(len(self.eval_data), 1)

    def _free_gen_sample(self) -> str:
        """Greedy 3-token sample from first eval pair."""
        import torch
        self.moshi_model.eval()
        try:
            item   = self.eval_data[0]
            base   = self.moshi_model.get_base_model()
            vocab  = base.emb[0].weight.shape[0]
            emb    = item["emb"].to(self.device)
            cv     = self.conditioner.condition_sum_vector(emb.to(torch.bfloat16))
            ids    = item["ids"][:4]
            ids_t  = torch.tensor(ids, dtype=torch.long, device=self.device).clamp(0, vocab-1)
            x      = base.emb[0](ids_t).unsqueeze(0).to(torch.bfloat16)
            x      = x + cv.unsqueeze(1).to(x.dtype)
            tokens = []
            with torch.no_grad():
                for _ in range(FREE_GEN_TOKENS):
                    h      = base.transformer(x)
                    logits = base.text_linear(h[:, -1:]).squeeze().float()
                    nid    = int(logits.argmax())
                    tokens.append(self.sp.id_to_piece(nid))
                    nt = torch.tensor([[nid]], dtype=torch.long,
                                      device=self.device).clamp(0, vocab-1)
                    x  = torch.cat([x, base.emb[0](nt).to(torch.bfloat16)], dim=1)
            self.moshi_model.train()
            return " ".join(tokens)
        except Exception as e:
            self.moshi_model.train()
            return f"[sample failed: {e}]"


    @modal.method()
    def extract_embeddings(self) -> None:
        """
        Pre-extract MERT embeddings for all train + eval pairs.
        Run ONCE before training. Caches to EMB_DIR on trubai-checkpoints volume.
        Safe to re-run — skips already-cached files.

        modal run trubai_lora_v4.py::extract_embeddings_run
        """
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        train_data = self._load_or_extract(self.train_raw, "train")
        eval_data  = self._load_or_extract(self.eval_raw,  "eval")
        vol.commit()
        print()
        print("=" * 60)
        print("EMBEDDING EXTRACTION COMPLETE")
        print(f"  Train pairs: {len(train_data)}")
        print(f"  Eval pairs:  {len(eval_data)}")
        print(f"  Cache:       {EMB_DIR}")
        print("  Ready for: modal run trubai_lora_v4.py::main")
        print("=" * 60)

    def _ensure_embeddings(self) -> None:
        """Load embeddings from cache into self.train_data / self.eval_data."""
        if self.train_data is not None:
            return
        print("Loading embeddings from cache...")
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        self.train_data = self._load_or_extract(self.train_raw, "train")
        self.eval_data  = self._load_or_extract(self.eval_raw,  "eval")


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoints
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def extract_embeddings_run() -> None:
    """
    Pre-extract MERT embeddings before training.
    Run once: modal run trubai_lora_v4.py::extract_embeddings_run
    """
    LoRATrainV4().extract_embeddings.remote()

@app.local_entrypoint()
def main(resume_epoch: int = 0) -> None:
    """
    Production run. Stops at each gate epoch for Muse confirmation.
    Fresh run:  modal run trubai_lora_v4.py::main
    Resume:     modal run trubai_lora_v4.py::main --resume-epoch 20
    """
    LoRATrainV4().train.remote(resume_epoch=resume_epoch)