#!/usr/bin/env python3
"""BF16 sidecar: dequantize the EXL3 tensors Tempo cannot run as EXL3 linears.

Tempo reads these as plain weights, so they must be BF16:
  wo_a                  grouped torch.bmm in deep_gemm_fp8_o_proj when not fp8;
                        slices 0..7 (each [out=1024, in=4096]) stacked to [8192, 4096]
  compressor.*          raw torch.mm on fused_wkv_wgate.weight
  indexer.wk            ReplicatedLinear(quant_config=None)
  ffn.shared_experts.*  the Tempo exl3_config refuses EXL3 shared experts (P5)

Everything else stays EXL3 and is served by cuda-exl3's linear method.
Dense weight = exl3_linear(I): y = x @ W_eff, so the nn.Linear weight is W_eff.T.

Usage (Path A′ image, one GPU): dequant-nonrouted.py /model /out
Writes /out/nonrouted-bf16-<block>.safetensors and /out/nonrouted-bf16.index.json.
"""
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cuda_exl3 import ops

CB_MUL1 = 2
TILE = 16
MODEL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
IDX = json.load(open(MODEL / "model.safetensors.index.json"))["weight_map"]
_files = {}


def get(name):
    f = IDX[name]
    if f not in _files:
        _files[f] = safe_open(str(MODEL / f), framework="pt", device="cuda:0")
    return _files[f].get_tensor(name)


def dense(mod):
    """nn.Linear weight [out, in] in bf16."""
    t, suh, svh = get(f"{mod}.trellis"), get(f"{mod}.suh").half(), get(f"{mod}.svh").half()
    k, n = t.shape[0] * TILE, t.shape[1] * TILE
    eye = torch.eye(k, dtype=torch.half, device="cuda")
    w = ops.exl3_linear(eye, t, suh.view(1, -1), svh, [n], CB_MUL1)
    if not bool(torch.isfinite(w).all()):
        raise SystemExit(f"{mod}: non-finite values after dequant")
    return w.t().contiguous().to(torch.bfloat16)


mods = sorted({k[: -len(".trellis")] for k in IDX if k.endswith(".trellis") and ".experts." not in k})
want = [m for m in mods
        if re.search(r"\.attn\.wo_a\.slice\.\d+$", m)
        or ".attn.compressor." in m
        or m.endswith(".attn.indexer.wk")
        or ".ffn.shared_experts." in m]
if "--present-only" in sys.argv:  # smoke test on a partial tree
    want = [m for m in want if (MODEL / IDX[f"{m}.trellis"]).exists()]
blocks = sorted({re.match(r"^(layers\.\d+|mtp\.\d+)\.", m).group(1) for m in want},
                key=lambda b: (b.split(".")[0] != "layers", int(b.split(".")[1])))

torch.set_grad_enabled(False)
weight_map, total = {}, 0
for b in blocks:
    out = {}
    slices = sorted((m for m in want if m.startswith(f"{b}.attn.wo_a.slice.")),
                    key=lambda m: int(m.rsplit(".", 1)[1]))
    if slices:
        out[f"{b}.attn.wo_a.weight"] = torch.cat([dense(m) for m in slices], dim=0)
    for m in want:
        if m.startswith(f"{b}.") and ".wo_a.slice." not in m:
            out[f"{m}.weight"] = dense(m)
    fname = f"nonrouted-bf16-{b.replace('.', '-')}.safetensors"
    save_file({k: v.cpu() for k, v in out.items()}, str(OUT / fname),
              metadata={"format": "pt", "source": "exl3_linear(I) of LibertAIDAI 2.9bpw c178a012"})
    for k, v in out.items():
        weight_map[k] = fname
        total += v.numel() * 2
    print(f"{b}: {len(out)} tensors, {sum(v.numel() for v in out.values()) * 2 / 2**20:.1f} MiB",
          flush=True)
    _files.clear()
    torch.cuda.empty_cache()

json.dump({"metadata": {"total_size": total,
                        "replaces_exl3_modules": want},
           "weight_map": weight_map},
          open(OUT / "nonrouted-bf16.index.json", "w"), indent=1)
print(f"done: {len(weight_map)} tensors, {total / 2**30:.2f} GiB, {len(want)} EXL3 modules replaced")
