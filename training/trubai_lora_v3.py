"""
trubai_lora_v3.py
LoRA retraining — SPEC-LORA-V2-v1 (Muse-approved)

Trains LoRA adapters on top of Moshi LM conditioned by TensorConditioner v13.
Key differences from Phase 5 LoRA:
  - Conditioner: v13 ConditionerV12 (frozen, direction * output_scale injection)
  - Margin loss: hard negatives from observed junk vocabulary (Gmina, pgfplots etc)
  - Best checkpoint: held-out pair 22 CE loss, not epoch 80
  - Reporting gates: epoch 20, 40, 60, 80 with 3-token free-gen sample

Resume bug fix (confirmed):
  - Bug: optimizer state not saved → fresh AdamW on resume → zero moment estimates
    → full LR applied to trained weights → CE spike (observed: 3.8 → 33)
  - Fix: save optimizer_{epoch:03d}.pt at every gate alongside adapter checkpoint
    Load after optimizer construction on resume — restores momentum/velocity/steps

Smoke test:
    modal run trubai_lora_v3.py --smoke-test 1
    Runs 3 epochs, saves, resumes from epoch 3, runs to epoch 5.
    Reports epoch 3 and 5 metrics side by side to confirm continuity.

80-epoch run:
    modal run trubai_lora_v3.py
    modal run trubai_lora_v3.py --resume-epoch 20
"""


import modal
import json
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
        "peft",
    ])
)

app      = modal.App("trubai-lora-v3", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

CKPT_DIR       = Path("/checkpoints")
HF_DIR         = Path("/hf-cache")
ANCHOR_JSONL   = CKPT_DIR / "trubai_training_pairs.jsonl"
PHASE5_EMB_DIR = CKPT_DIR / "embeddings"
V13_CKPT       = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_V3_DIR    = CKPT_DIR / "lora_v3"

# Training hyperparameters — SPEC-LORA-V2-v1
LORA_RANK        = 8
LORA_ALPHA       = 16
LORA_LR          = 5e-5
LORA_WD          = 0.1
EPOCHS           = 80
KL_LAMBDA        = 0.05
MARGIN           = 2.0
HELD_OUT_IDX     = 21
GATE_EPOCHS      = {20, 40, 60, 80}
FREE_GEN_TOKENS  = 3

HARD_NEGATIVES = [
    " Gmina",
    " pgfplots",
    " martingale",
]


def build_conditioner_v12(device: str):
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
            self.register_buffer("output_scale", torch.tensor(36.5))

        def forward(self, emb):
            mask  = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            cond  = TensorCondition(tensor=emb, mask=mask)
            proj  = self.tc(cond)[0]
            raw   = proj.squeeze(1)
            direction = F.normalize(raw, dim=-1)
            return direction, self.output_scale

        def condition_sum_vector(self, emb):
            direction, scale = self.forward(emb)
            return direction * scale.detach()

    return ConditionerV12().to(device)


@app.cls(
    gpu="H100",
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=21600,
)
class LoRATrainV3:

    @modal.enter()
    def setup(self):
        import torch._dynamo
        torch._dynamo.config.disable = True

        import os
        os.environ["HF_HOME"] = str(HF_DIR)

        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        from peft import LoraConfig, get_peft_model

        self.device = torch.device("cuda")

        print("Loading Moshi LM...")
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16,
        )
        self._moshi_weight = moshi_weight  # retained for KL reference load

        print("Loading v13 conditioner...")
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for param in self.conditioner.parameters():
            param.requires_grad = False
        frozen = sum(p.numel() for p in self.conditioner.parameters() if not p.requires_grad)
        total  = sum(p.numel() for p in self.conditioner.parameters())
        assert frozen == total, f"Conditioner not fully frozen: {frozen}/{total}"
        print(f"  ConditionerV12 frozen ✓ ({frozen:,} parameters)")

        from moshi.conditioners import ConditionProvider, ConditionFuser
        self.condition_provider = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)
        moshi_lm.condition_provider = self.condition_provider
        moshi_lm.fuser              = self.fuser

        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        self.moshi_model = get_peft_model(moshi_lm, lora_config)
        self.moshi_model.print_trainable_parameters()

        base = self.moshi_model.get_base_model()
        base.condition_provider = self.condition_provider
        base.fuser               = self.fuser

        lora_param_ids = {id(p) for p in self.moshi_model.parameters() if p.requires_grad}
        tc_param_ids   = {id(p) for p in self.conditioner.parameters()}
        assert len(lora_param_ids & tc_param_ids) == 0, "Conditioner params leaked into LoRA"
        print("  Conditioner parameters absent from LoRA trainable set ✓")

        tok_path = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp  = sentencepiece.SentencePieceProcessor(tok_path)

        self.hard_neg_ids = [self.sp.encode(s) for s in HARD_NEGATIVES]

        print("Loading reference model for KL...")
        ref_lm = loaders.get_moshi_lm(moshi_weight, device="cpu", dtype=torch.bfloat16)
        self.ref_text_linear = ref_lm.text_linear.weight.detach().float().to(self.device)
        del ref_lm

        with open(ANCHOR_JSONL) as f:
            records = [json.loads(l) for l in f]
        assert len(records) == 22, f"Expected 22 pairs, got {len(records)}"

        self.train_records = [r for i, r in enumerate(records) if i != HELD_OUT_IDX]
        self.eval_record   = records[HELD_OUT_IDX]
        print(f"  Training pairs: {len(self.train_records)} | held-out: pair {HELD_OUT_IDX+1}")

        self.train_data = []
        for rec in self.train_records:
            emb  = self._load_emb(rec)
            ids  = self.sp.encode(rec["inner_monologue"])
            self.train_data.append({"emb": emb, "ids": ids, "text": rec["inner_monologue"]})

        eval_emb  = self._load_emb(self.eval_record)
        eval_ids  = self.sp.encode(self.eval_record["inner_monologue"])
        self.eval_data = {"emb": eval_emb, "ids": eval_ids, "text": self.eval_record["inner_monologue"]}

        print("Setup complete.")

    @modal.method()
    def train(
            self,
            resume_epoch:  int = 0,
            epoch_override: int = 0,    # 0 = use module EPOCHS constant
            gate_override:  str = "",   # "" = use module GATE_EPOCHS; else comma-separated ints
    ) -> dict:
        """
        Main training loop.
        epoch_override / gate_override allow smoke-test runs without modifying
        module constants. The 80-epoch production run passes neither — defaults
        to EPOCHS=80, GATE_EPOCHS={20,40,60,80}.
        """
        import torch
        import torch.nn.functional as F
        import random

        n_epochs   = epoch_override if epoch_override > 0 else EPOCHS
        gate_set   = (
            {int(x) for x in gate_override.split(",") if x.strip()}
            if gate_override else GATE_EPOCHS
        )
        # For smoke test: only gate at epochs within the run range
        gate_set = {e for e in gate_set if e <= n_epochs}

        LORA_V3_DIR.mkdir(parents=True, exist_ok=True)

        optimizer = torch.optim.AdamW(
            [p for p in self.moshi_model.parameters() if p.requires_grad],
            lr=LORA_LR, weight_decay=LORA_WD,
        )

        # ── Resume bug fix: restore optimizer state after construction ─────────
        # Bug: on resume, fresh AdamW had zero moment estimates → full LR applied
        # to trained weights → CE spiked from ~3.8 to ~33.
        # Fix: save optimizer_{epoch:03d}.pt at each gate; load here after
        # construction so momentum/velocity/step counts are fully restored.
        if resume_epoch > 0:
            ckpt     = LORA_V3_DIR / f"epoch_{resume_epoch:03d}"
            opt_path = LORA_V3_DIR / f"optimizer_{resume_epoch:03d}.pt"
            self.moshi_model.load_adapter(str(ckpt), adapter_name="default")
            print(f"Resumed adapter from epoch {resume_epoch} ✓")
            if opt_path.exists():
                optimizer.load_state_dict(
                    torch.load(str(opt_path), map_location=self.device, weights_only=True)
                )
                print(f"Optimizer state restored from {opt_path} ✓")
            else:
                print(f"WARNING: optimizer state not found at {opt_path}")
                print(f"  Moment estimates lost — expect CE spike on first epoch.")

        start_epoch    = resume_epoch + 1 if resume_epoch > 0 else 1
        best_eval_loss = float("inf")
        gate_metrics   = {}   # epoch → {ce, kl, mg, eval_ce} — returned for smoke test

        print(f"Training epochs {start_epoch}–{n_epochs}  |  gates: {sorted(gate_set)}")
        print("=" * 70)

        for epoch in range(start_epoch, n_epochs + 1):
            self.moshi_model.train()
            random.shuffle(self.train_data)
            epoch_ce = 0.0
            epoch_kl = 0.0
            epoch_mg = 0.0

            for item in self.train_data:
                optimizer.zero_grad()
                emb      = item["emb"].to(self.device)
                cond_vec = self.conditioner.condition_sum_vector(
                    emb.to(torch.bfloat16)
                ).detach()

                L_ce = self._compute_ce(item["ids"], cond_vec)
                L_kl = self._compute_kl(item["ids"])
                L_mg = torch.tensor(0.0, device=self.device)
                for neg_ids in self.hard_neg_ids:
                    L_ce_neg = self._compute_ce(neg_ids, cond_vec)
                    L_mg = L_mg + torch.clamp(MARGIN - (L_ce_neg - L_ce), min=0.0)
                L_mg = L_mg / max(len(self.hard_neg_ids), 1)

                loss = L_ce + KL_LAMBDA * L_kl + L_mg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.moshi_model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()

                epoch_ce += L_ce.item()
                epoch_kl += L_kl.item()
                epoch_mg += L_mg.item()

            n = len(self.train_data)
            print(
                f"Epoch {epoch:3d} | "
                f"CE={epoch_ce/n:.4f} | KL={epoch_kl/n:.4f} | Mg={epoch_mg/n:.4f}"
            )

            if epoch in gate_set:
                self._report_gate(epoch, cond_vec)

                gate_path = LORA_V3_DIR / f"epoch_{epoch:03d}"
                self.moshi_model.save_pretrained(str(gate_path))
                # Resume bug fix: save optimizer state alongside adapter
                torch.save(
                    optimizer.state_dict(),
                    LORA_V3_DIR / f"optimizer_{epoch:03d}.pt",
                    )
                vol.commit()
                print(f"  Saved adapter + optimizer state → epoch_{epoch:03d}/ ✓")

                eval_loss = self._eval_ce()
                print(f"  Held-out pair 22 CE = {eval_loss:.4f}")

                gate_metrics[epoch] = {
                    "train_ce": round(epoch_ce / n, 4),
                    "train_kl": round(epoch_kl / n, 4),
                    "train_mg": round(epoch_mg / n, 4),
                    "eval_ce":  round(eval_loss, 4),
                }

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_path = LORA_V3_DIR / "best"
                    self.moshi_model.save_pretrained(str(best_path))
                    vol.commit()
                    print(f"  New best → {best_path}  (eval_ce={best_eval_loss:.4f})")

        print()
        print("=" * 70)
        print(f"Training complete. Best eval CE = {best_eval_loss:.4f}")
        return gate_metrics

    @modal.method()
    def smoke_test(self) -> None:
        """
        Resume bug smoke test — Muse requirement before 80-epoch rerun.

        Phase A: train epochs 1-3, gate at 3, save adapter + optimizer.
        Phase B: resume from epoch 3, train epochs 4-5, gate at 5.
        Report epoch 3 and epoch 5 metrics side by side.
        Continuous training = no CE spike, no LR discontinuity.

        Uses the same train() method and code path as the 80-epoch run.
        Not a simplified version — identical loss, optimizer, and save logic.
        """
        print()
        print("=" * 70)
        print("RESUME BUG SMOKE TEST")
        print("Phase A: epochs 1-3, save at gate 3")
        print("Phase B: resume from epoch 3, epochs 4-5, gate at 5")
        print("=" * 70)

        # Phase A — train from scratch to epoch 3
        metrics_a = self.train.local(
            resume_epoch=0,
            epoch_override=3,
            gate_override="3",
        )
        print()
        print(f"Phase A complete. Epoch 3 metrics: {metrics_a.get(3, 'NOT RECORDED')}")

        # Phase B — resume from epoch 3, train to epoch 5
        metrics_b = self.train.local(
            resume_epoch=3,
            epoch_override=5,
            gate_override="5",
        )
        print()
        print(f"Phase B complete. Epoch 5 metrics: {metrics_b.get(5, 'NOT RECORDED')}")

        # Side-by-side comparison
        e3 = metrics_a.get(3, {})
        e5 = metrics_b.get(5, {})

        print()
        print("=" * 70)
        print("SMOKE TEST RESULT — EPOCH 3 vs EPOCH 5")
        print("=" * 70)
        print(f"  {'Metric':<15}  {'Epoch 3':>10}  {'Epoch 5':>10}  {'Delta':>10}  Assessment")
        print("  " + "-" * 65)

        all_continuous = True
        for key in ["train_ce", "eval_ce", "train_mg"]:
            v3  = e3.get(key, float("nan"))
            v5  = e5.get(key, float("nan"))
            delta = v5 - v3
            # CE should continue decreasing or be stable (not spike up)
            # Spike threshold: >0.5 absolute increase is a failure
            if key in ("train_ce", "eval_ce"):
                ok = delta < 0.5
                if not ok:
                    all_continuous = False
                verdict = "✓ continuous" if ok else "✗ SPIKE — resume bug present"
            else:
                ok = abs(delta) < 1.0
                verdict = "✓" if ok else "⚠ check"
            print(f"  {key:<15}  {v3:>10.4f}  {v5:>10.4f}  {delta:>+10.4f}  {verdict}")

        print()
        if all_continuous:
            print("  RESULT: PASS — no CE spike, optimizer state restored correctly")
            print("  80-epoch rerun is cleared pending Muse confirmation.")
        else:
            print("  RESULT: FAIL — spike detected, resume bug still present")
            print("  Do NOT proceed to 80-epoch rerun.")
        print("=" * 70)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _load_emb(self, record: dict):
        import torch
        stem = Path(record["embedding_path"]).stem
        path = PHASE5_EMB_DIR / f"{stem}.pt"
        emb  = torch.load(str(path), map_location="cpu", weights_only=False)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        return emb.float()

    def _compute_ce(self, token_ids: list, cond_vec=None) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

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

    def _compute_kl(self, token_ids: list) -> "torch.Tensor":
        import torch
        import torch.nn.functional as F

        if len(token_ids) < 2:
            return torch.tensor(0.0, device=self.device)

        base       = self.moshi_model.get_base_model()
        vocab_size = base.emb[0].weight.shape[0]
        ids_t      = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        ids_t      = ids_t.clamp(0, vocab_size - 1)

        x         = base.emb[0](ids_t[:-1]).unsqueeze(0).to(torch.bfloat16)
        h_ft      = base.transformer(x)
        logits_ft = base.text_linear(h_ft).squeeze(0).float()

        with torch.no_grad():
            logits_ref = (h_ft.detach().float().squeeze(0) @ self.ref_text_linear.T)

        p_ft  = F.log_softmax(logits_ft,  dim=-1)
        p_ref = F.softmax(logits_ref, dim=-1)
        return F.kl_div(p_ft, p_ref, reduction="batchmean")

    def _eval_ce(self) -> float:
        import torch
        self.moshi_model.eval()
        with torch.no_grad():
            emb      = self.eval_data["emb"].to(self.device)
            cond_vec = self.conditioner.condition_sum_vector(emb.to(torch.bfloat16))
            loss     = self._compute_ce(self.eval_data["ids"], cond_vec)
        self.moshi_model.train()
        return loss.item()

    def _report_gate(self, epoch: int, cond_vec: "torch.Tensor") -> None:
        import torch
        print()
        print(f"{'='*60}")
        print(f"  GATE — Epoch {epoch}")
        print(f"{'='*60}")
        self.moshi_model.eval()
        try:
            sample_tokens = []
            with torch.no_grad():
                emb    = self.eval_data["emb"].to(self.device)
                cv     = self.conditioner.condition_sum_vector(emb.to(torch.bfloat16))
                base_m = self.moshi_model.get_base_model()
                vocab_sz = base_m.emb[0].weight.shape[0]
                ids_seed = self.eval_data["ids"][:4]
                ids_t    = torch.tensor(ids_seed, dtype=torch.long, device=self.device)
                ids_t    = ids_t.clamp(0, vocab_sz - 1)
                x        = base_m.emb[0](ids_t).unsqueeze(0).to(torch.bfloat16)
                x        = x + cv.unsqueeze(1).to(x.dtype)
                h        = base_m.transformer(x)
                logits   = base_m.text_linear(h[:, -1:]).squeeze().float()
                for _ in range(FREE_GEN_TOKENS):
                    next_id = int(logits.argmax())
                    piece   = self.sp.id_to_piece(next_id)
                    sample_tokens.append(piece)
                    next_t = torch.tensor([[next_id]], dtype=torch.long, device=self.device)
                    next_t = next_t.clamp(0, vocab_sz - 1)
                    # emb[0] returns [1, 1, D] for a [1,1] input — squeeze to [1, D]
                    # then cat with x [1, T, D] along dim=1
                    x_new  = base_m.emb[0](next_t).to(torch.bfloat16)  # [1, 1, D]
                    x      = torch.cat([x, x_new], dim=1)               # [1, T+1, D]
                    h      = base_m.transformer(x)
                    logits = base_m.text_linear(h[:, -1:]).squeeze().float()
            print(f"  Free-gen sample ({FREE_GEN_TOKENS} tokens): {' '.join(sample_tokens)}")
        except Exception as e:
            print(f"  Free-gen sample failed: {e}")
        print(f"{'='*60}")
        self.moshi_model.train()


# ──────────────────────────────────────────────────────────────────────────────
# Local entrypoints
# ──────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(resume_epoch: int = 0) -> None:
    """80-epoch production run. Uses module constants EPOCHS=80, GATE_EPOCHS={20,40,60,80}."""
    LoRATrainV3().train.remote(resume_epoch=resume_epoch)


@app.local_entrypoint()
def smoke_test_run() -> None:
    """
    Resume bug smoke test. Run before the 80-epoch rerun.
    modal run trubai_lora_v3.py::smoke_test_run
    """
    LoRATrainV3().smoke_test.remote()