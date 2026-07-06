"""
trubai_derive_logit_bias.py
SPEC-5BA-v1 §3 — Derive LogitBiasVector

Computes bias_vector = log(p_pedagogical) - log(p_failure)
from 22 predicate completion triples and failure_tokens.json.

Prerequisites:
  - /checkpoints/data/failure_tokens.json (§1 output)
  - /checkpoints/trubai_training_pairs.jsonl (22 approved triples)
  - /checkpoints/embeddings/ (MERT embeddings per pair)
  - /checkpoints/retrain_v13/best/tensor_conditioner.pt (v13 conditioner)
  - /checkpoints/lora_v3/best (LoRA adapter)

Outputs:
  - /checkpoints/logit_bias/bias_v1.pt

§3.4 summary reported to Muse before §4 alpha calibration.

Usage:
    modal run trubai_derive_logit_bias.py
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
        "sentencepiece",
        "huggingface_hub",
        "safetensors",
        "peft",
    ])
)

app      = modal.App("trubai-derive-logit-bias", image=image)
vol      = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

CKPT_DIR      = Path("/checkpoints")
HF_DIR        = Path("/hf-cache")
FAILURE_JSON  = CKPT_DIR / "data" / "failure_tokens.json"
PAIRS_JSONL   = CKPT_DIR / "trubai_training_pairs.jsonl"
EMB_DIR       = CKPT_DIR / "embeddings"
V13_CKPT      = CKPT_DIR / "retrain_v13" / "best" / "tensor_conditioner.pt"
LORA_CKPT     = CKPT_DIR / "lora_v3" / "best"
BIAS_DIR      = CKPT_DIR / "logit_bias"
VOCAB_SIZE    = 32000
EPS           = 1e-8


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
            mask      = torch.ones(emb.shape[:2], dtype=torch.bool, device=emb.device)
            cond      = TensorCondition(tensor=emb, mask=mask)
            proj      = self.tc(cond)[0]
            raw       = proj.squeeze(1)
            direction = F.normalize(raw, dim=-1)
            return direction, self.output_scale

        def condition_sum_vector(self, emb):
            direction, scale = self.forward(emb)
            return direction * scale.detach()

    c = ConditionerV12().to(device)
    return c


@app.cls(
    gpu="H100",
    volumes={"/checkpoints": vol, "/hf-cache": hf_cache},
    timeout=3600,
)
class DeriveBias:

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
        from moshi.conditioners import ConditionAttributes, ConditionProvider, ConditionFuser
        from moshi.conditioners import TensorCondition
        from peft import LoraConfig, get_peft_model

        self.device = torch.device("cuda")

        # Tokenizer
        tok_path  = hf_hub_download(loaders.DEFAULT_REPO, loaders.TEXT_TOKENIZER_NAME)
        self.sp   = sentencepiece.SentencePieceProcessor(tok_path)

        # Moshi LM
        moshi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MOSHI_NAME)
        moshi_lm     = loaders.get_moshi_lm(
            moshi_weight, device=self.device, dtype=torch.bfloat16
        )

        # v13 ConditionerV12 — frozen
        self.conditioner = build_conditioner_v12(str(self.device))
        state = torch.load(str(V13_CKPT), map_location=self.device, weights_only=True)
        self.conditioner.load_state_dict(state)
        self.conditioner.eval()
        for p in self.conditioner.parameters():
            p.requires_grad = False

        # ConditionProvider / Fuser
        self.cp = ConditionProvider(
            conditioners={"mert": self.conditioner.tc}, device=self.device,
        ).to(torch.bfloat16).to(self.device)
        self.fuser = ConditionFuser(
            fuse2cond={"sum": ["mert"], "cross": []},
        ).to(torch.bfloat16).to(self.device)

        moshi_lm.condition_provider = self.cp
        moshi_lm.fuser              = self.fuser

        # LoRA adapter
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["in_projs.0", "out_projs.0"],
            lora_dropout=0.0, bias="none",
            layers_to_transform=list(range(28, 32)),
        )
        self.moshi_model = get_peft_model(moshi_lm, lora_config)
        self.moshi_model.load_adapter(str(LORA_CKPT), adapter_name="default")
        self.moshi_model.eval()

        base = self.moshi_model.get_base_model()
        base.condition_provider = self.cp
        base.fuser              = self.fuser

        # Load training pairs
        with open(PAIRS_JSONL) as f:
            self.pairs = [json.loads(l) for l in f]
        assert len(self.pairs) == 22, f"Expected 22 pairs, got {len(self.pairs)}"
        print(f"Loaded {len(self.pairs)} training pairs")
        print("Setup complete.")

    @modal.method()
    def derive(self) -> dict:
        import torch
        import torch.nn.functional as F
        from moshi.models import LMGen
        from moshi.conditioners import ConditionAttributes, TensorCondition

        # ── §3.1 Pedagogical distribution ─────────────────────────────────────
        print("\n§3.1 Collecting pedagogical logit distribution...")
        logits_accum = []   # each entry: [vocab_size] float32 softmax

        base   = self.moshi_model.get_base_model()
        merged = self.moshi_model.merge_and_unload()
        merged.condition_provider = self.cp
        merged.fuser              = self.fuser

        for i, pair in enumerate(self.pairs):
            # Load embedding
            stem  = Path(pair["embedding_path"]).stem
            emb   = torch.load(str(EMB_DIR / f"{stem}.pt"),
                               map_location=self.device, weights_only=False)
            if emb.dim() == 2:
                emb = emb.unsqueeze(1)
            emb = emb.float()

            cond_vec = self.conditioner.condition_sum_vector(
                emb.to(torch.bfloat16)
            )

            # Encode completion text
            completion_ids = self.sp.encode(pair["inner_monologue"])
            completion_ids = [min(max(0, t), VOCAB_SIZE - 1) for t in completion_ids]

            if len(completion_ids) < 2:
                print(f"  pair {i+1}: too short, skipping")
                continue

            # Clamp against actual embedding table size (may differ from VOCAB_SIZE)
            base2 = self.moshi_model.get_base_model()
            actual_vocab = base2.emb[0].weight.shape[0]
            completion_ids = [min(max(0, t), actual_vocab - 1) for t in completion_ids]

            # Teacher-forced forward through transformer at each completion position
            ids_t = torch.tensor(completion_ids, dtype=torch.long, device=self.device)
            base2 = merged

            with torch.no_grad():
                x = base2.emb[0](ids_t).unsqueeze(0).to(torch.bfloat16)
                x = x + cond_vec.unsqueeze(1).to(x.dtype)
                h = base2.transformer(x)                        # [1, T, D]
                # Collect logits at completion positions (all but last input)
                logits_all = base2.text_linear(h).float()       # [1, T, vocab]
                logits_all = logits_all.squeeze(0)               # [T, vocab]
                probs_all  = F.softmax(logits_all, dim=-1)       # [T, vocab]

            for t in range(probs_all.shape[0]):
                logits_accum.append(probs_all[t].cpu())

            print(f"  pair {i+1:2d}: {len(completion_ids)} tokens → {probs_all.shape[0]} positions")

        print(f"Total pedagogical positions collected: {len(logits_accum)}")

        # Mean pedagogical distribution
        logits_stack   = torch.stack(logits_accum, dim=0)   # [N, vocab]
        p_pedagogical  = logits_stack.mean(dim=0)            # [vocab]
        print(f"p_pedagogical sum: {p_pedagogical.sum().item():.4f} (should be ~1.0)")

        # ── §3.2 Failure distribution ──────────────────────────────────────────
        print("\n§3.2 Loading failure distribution...")
        with open(FAILURE_JSON) as f:
            failure_data = json.load(f)

        p_failure = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        for entry in failure_data["token_entries"]:
            tid = entry["id"]
            if 0 <= tid < VOCAB_SIZE:
                p_failure[tid] = entry["count"]

        total_fail = p_failure.sum()
        p_failure  = p_failure / total_fail
        print(f"Failure distribution: {int(total_fail)} total occurrences across "
              f"{(p_failure > 0).sum().item()} tokens")

        # ── §3.3 Bias vector computation — SPLIT APPROACH ──────────────────────
        # Muse correction: positive/negative sides require different treatment.
        # Negative side: formula is valid — failure tokens are specific.
        # Positive side: formula is INVALID when p_fail=0 due to sample sparsity —
        #   ANY token absent from 44-token failure sample gets +18 bias.
        #   Restrict positive bias to W_positive vocabulary only, cap at +8.0.
        print("\n§3.3 Computing bias vector (split approach)...")

        # Confirmed W_POSITIVE_IDS from Muse (SPEC-5BA-v1, confirmed pre-run)
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

        import math
        log_p_ped  = torch.log(p_pedagogical + EPS)
        log_p_fail = torch.log(p_failure      + EPS)

        # Negative side: formula applied only to failure token IDs
        # log(p_ped) - log(p_fail) — negative where p_ped << p_fail
        failure_ids = [e["id"] for e in failure_data["token_entries"]
                       if 0 <= e["id"] < VOCAB_SIZE]
        negative_bias = torch.zeros(VOCAB_SIZE)
        for tid in failure_ids:
            val = log_p_ped[tid].item() - log_p_fail[tid].item()
            negative_bias[tid] = val   # negative when p_ped << p_fail

        # Positive side: W_positive tokens only, capped at +8.0
        # Use log(p_ped) - log(eps) as attraction signal (p_fail=0 for these)
        POSITIVE_CAP = 8.0
        positive_bias = torch.zeros(VOCAB_SIZE)
        for tid in W_POSITIVE_IDS:
            if p_pedagogical[tid].item() > EPS:
                val = log_p_ped[tid].item() - math.log(EPS)
                positive_bias[tid] = min(val, POSITIVE_CAP)

        bias_vector = negative_bias + positive_bias

        n_positive = (positive_bias > 0).sum().item()
        n_negative = (negative_bias < 0).sum().item()
        print(f"Tokens with positive bias: {n_positive} (expect ≤17)")
        print(f"Tokens with negative bias: {n_negative} (expect 44)")
        print(f"Tokens with zero bias:     {(bias_vector == 0).sum().item()}")

        # ── §3.4 Summary report ────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("§3.4 BIAS VECTOR SUMMARY — REPORT TO MUSE")
        print("=" * 60)

        above_1   = (bias_vector.abs() > 1.0).sum().item()
        below_001 = (bias_vector.abs() < 0.01).sum().item()
        n_positive_report = (positive_bias > 0).sum().item()
        n_negative_report = (negative_bias < 0).sum().item()
        print(f"Tokens with positive bias: {n_positive_report}  (expect ≤17 — W_positive only)")
        print(f"Tokens with negative bias: {n_negative_report}  (expect 44 — failure tokens)")
        print(f"Tokens with |bias| > 1.0:  {above_1}")
        print(f"Tokens with |bias| < 0.01: {below_001}  (zero/neutral)")
        print()

        # Top 10 positively biased (pedagogical vocabulary attracted)
        top_pos_vals, top_pos_idx = bias_vector.topk(10)
        print("Top 10 POSITIVE bias (attracted toward pedagogical register):")
        for val, idx in zip(top_pos_vals.tolist(), top_pos_idx.tolist()):
            piece = self.sp.id_to_piece(idx)
            ped   = p_pedagogical[idx].item()
            fail  = p_failure[idx].item()
            print(f"  id={idx:6d}  {piece!r:25}  bias={val:+.4f}  "
                  f"p_ped={ped:.6f}  p_fail={fail:.6f}")
        print()

        # Top 10 negatively biased (failure register suppressed)
        top_neg_vals, top_neg_idx = (-bias_vector).topk(10)
        print("Top 10 NEGATIVE bias (suppressed failure register):")
        for val, idx in zip(top_neg_vals.tolist(), top_neg_idx.tolist()):
            piece = self.sp.id_to_piece(idx)
            ped   = p_pedagogical[idx].item()
            fail  = p_failure[idx].item()
            print(f"  id={idx:6d}  {piece!r:25}  bias={-val:+.4f}  "
                  f"p_ped={ped:.6f}  p_fail={fail:.6f}")
        print()

        # Specific values Muse requested
        mr_id   = self.sp.piece_to_id("▁Mr")
        sig_id  = self.sp.piece_to_id("▁significantly")
        print(f"Specific values requested by Muse:")
        mr_bias  = bias_vector[mr_id].item()
        sig_bias = bias_vector[sig_id].item()
        print(f"  ▁Mr           (id={mr_id}):  bias={mr_bias:+.4f}  "
              f"p_ped={p_pedagogical[mr_id].item():.8f}  "
              f"p_fail={p_failure[mr_id].item():.6f}")
        print(f"  ▁significantly (id={sig_id}): bias={sig_bias:+.4f}  "
              f"p_ped={p_pedagogical[sig_id].item():.8f}  "
              f"p_fail={p_failure[sig_id].item():.6f}")
        print()
        print(f"Core W_positive bias values:")
        core_wp = [
            ("▁air",    1142),
            ("▁breath", 8735),
            ("▁tone",   9064),
            ("▁center", 1611),
        ]
        for piece, tid in core_wp:
            b   = bias_vector[tid].item()
            ped = p_pedagogical[tid].item()
            print(f"  {piece:15} (id={tid:5d}):  bias={b:+.4f}  p_ped={ped:.6f}")
        print()

        # ── §3.4 Save ──────────────────────────────────────────────────────────
        BIAS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = BIAS_DIR / "bias_v1.pt"
        torch.save({"bias_vector": bias_vector, "alpha": 1.0}, str(out_path))
        vol.commit()
        print(f"Saved: {out_path}")
        print("=" * 60)

        return {
            "n_positive":  n_positive_report,
            "n_negative":  n_negative_report,
            "above_1":     above_1,
            "mr_bias":     mr_bias,
            "sig_bias":    sig_bias,
            "n_positions": len(logits_accum),
        }


@app.local_entrypoint()
def main():
    result = DeriveBias().derive.remote()
    print()
    print("Derivation complete.")
    print(f"  n_positions:  {result['n_positions']}")
    print(f"  n_positive:   {result['n_positive']}")
    print(f"  n_negative:   {result['n_negative']}")
    print(f"  above_1:      {result['above_1']}")
    print(f"  ▁Mr bias:     {result['mr_bias']:+.4f}")
    print(f"  ▁sig bias:    {result['sig_bias']:+.4f}")
    print()
    print("Return full §3.4 output to Muse before §4 alpha calibration.")