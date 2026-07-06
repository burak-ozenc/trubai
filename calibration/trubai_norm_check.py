# Norm check — original vs retrained TensorConditioner
# Prints four numbers and exits. No training, no session.
# ─────────────────────────────────────────────────────────────────────────────

import modal
import numpy as np
from pathlib import Path

app = modal.App("trubai-norm-check")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(["ffmpeg"])
    .pip_install([
        "torch==2.6.0",
        "moshi",
        "huggingface_hub",
    ])
)

volume   = modal.Volume.from_name("trubai-checkpoints", create_if_missing=True)
hf_cache = modal.Volume.from_name("trubai-hf-cache",    create_if_missing=True)

ORIGINAL_CKPT = "/checkpoints/epoch_047/tensor_conditioner.pt"
RETRAINED_CKPT = "/checkpoints/retrain_v2/best/tensor_conditioner.pt"
SAMPLE_EMB    = "/checkpoints/embeddings"  # pick first .pt file found
MERT_DIM      = 768


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/checkpoints": volume,
        "/hf-cache":    hf_cache,
    },
    timeout=300,
)
def check_norms():
    import os
    os.environ["HF_HOME"] = "/hf-cache"

    import torch
    from huggingface_hub import hf_hub_download
    from moshi.models import loaders
    from moshi.conditioners.tensors import TensorConditioner
    from moshi.conditioners import TensorCondition

    device = "cuda"

    def make_conditioner():
        return TensorConditioner(
            dim=768, output_dim=4096, device=device,
            force_linear=True, output_bias=False, learn_padding=True,
        ).to(device).float()

    def load_conditioner(path):
        tc = make_conditioner()
        state = torch.load(path, map_location=device, weights_only=False)
        tc.load_state_dict(state)
        tc.eval()
        return tc

    def project(tc, emb):
        mask = torch.ones(1, 1, dtype=torch.bool, device=device)
        tc_input = TensorCondition(tensor=emb, mask=mask)
        out = tc(tc_input)
        return out.condition  # [1, 1, 4096]

    # Load both checkpoints
    tc_original  = load_conditioner(ORIGINAL_CKPT)
    tc_retrained = load_conditioner(RETRAINED_CKPT)

    # Discover parameter names before accessing
    print("TensorConditioner parameters:")
    for name, param in tc_original.named_parameters():
        print(f"  {name}: {param.shape}")

    # Weight norms — find the linear projection weight
    # Collect all weight tensors and pick the largest (the 768→4096 projection)
    all_params = {n: p for n, p in tc_original.named_parameters()}
    proj_param_name = max(all_params, key=lambda n: all_params[n].numel())
    print(f"\nUsing '{proj_param_name}' as projection weight")

    orig_weight_norm      = all_params[proj_param_name].norm().item()
    retrained_all_params  = {n: p for n, p in tc_retrained.named_parameters()}
    retrained_weight_norm = retrained_all_params[proj_param_name].norm().item()

    # Load one sample embedding
    emb_files = sorted(Path(SAMPLE_EMB).glob("*.pt"))
    assert emb_files, f"No .pt files found in {SAMPLE_EMB}"
    sample_emb = torch.load(str(emb_files[0]), map_location=device, weights_only=False).float()
    print(f"Sample embedding: {emb_files[0].name} | shape: {sample_emb.shape}")

    # Projection output norms
    with torch.no_grad():
        proj_original  = project(tc_original,  sample_emb)   # [1, 1, 4096]
        proj_retrained = project(tc_retrained, sample_emb)   # [1, 1, 4096]

    orig_proj_norm     = proj_original.norm().item()
    retrained_proj_norm = proj_retrained.norm().item()

    ratio_weight = retrained_weight_norm / orig_weight_norm
    ratio_proj   = retrained_proj_norm   / orig_proj_norm

    print("\n" + "="*50)
    print("NORM CHECK RESULTS")
    print("="*50)
    print(f"Weight matrix norm:")
    print(f"  original:  {orig_weight_norm:.3f}")
    print(f"  retrained: {retrained_weight_norm:.3f}")
    print(f"  ratio:     {ratio_weight:.2f}x")
    print(f"\nProjection output norm (on {emb_files[0].name}):")
    print(f"  original:  {orig_proj_norm:.3f}")
    print(f"  retrained: {retrained_proj_norm:.3f}")
    print(f"  ratio:     {ratio_proj:.2f}x")
    print(f"\nDiagnosis:")
    if ratio_proj > 3.0:
        print(f"  MAGNITUDE — retrained projection is {ratio_proj:.1f}x larger.")
        print(f"  Fix: add L2 norm penalty, target_norm = {orig_proj_norm:.1f}")
    else:
        print(f"  DIRECTION — norms similar ({ratio_proj:.2f}x), likely direction overshoot.")
        print(f"  Fix: expand W_positive to include neutral tokens.")

    return {
        "orig_weight_norm":      round(orig_weight_norm, 3),
        "retrained_weight_norm": round(retrained_weight_norm, 3),
        "weight_ratio":          round(ratio_weight, 3),
        "orig_proj_norm":        round(orig_proj_norm, 3),
        "retrained_proj_norm":   round(retrained_proj_norm, 3),
        "proj_ratio":            round(ratio_proj, 3),
    }


@app.local_entrypoint()
def main():
    result = check_norms.remote()
    print("\n── Returned dict ──")
    import json
    print(json.dumps(result, indent=2))