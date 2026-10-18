#!/usr/bin/env python3
"""Single-GPU check: can Tempo's cuda-exl3 serve LibertAIDAI 2.9 bpw non-routed tensors at TP3?

Runs inside the Tempo TP3 image (cuda_exl3 present, exllamav3 absent) against real
checkpoint shards. No vLLM engine, no distributed group: every TP3 rank is simulated
on one GPU and compared with the unsharded result.

Checks per module:
  dequant   W = exl3_linear(I) gives the dense weight; x @ W must match exl3_linear(x)
  column    virtual-head pad by repeating whole 128-blocks, split into 3 ranks, concat == full
  row       zero-suh pad on the input dim, split into 3 ranks, sum of partials == full
  fused     wq_a+wkv and w1+w3 as one trellis with one suh row per shard == separate calls
  bmm       wo_a as one exl3_linear per local group (9 virtual groups, slice.8 = slice.7)

Usage: python3 exl3-tp3-kernel-check.py /model [layer ... | all]
  all: every layer, every MTP block, and the TP3-padded head
"""
import json
import sys

import torch
from safetensors import safe_open

from cuda_exl3 import ops

CB_MUL1 = 2
TILE = 16
HAD = 128
TP = 3
TOL = 5e-3

MODEL = sys.argv[1]
LAYERS = [int(x) for x in sys.argv[2:] if x.isdigit()] or [0, 20]
IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
_files = {}
results = []


def get(name):
    f = IDX[name]
    if f not in _files:
        _files[f] = safe_open(f"{MODEL}/{f}", framework="pt", device="cuda:0")
    return _files[f].get_tensor(name)


def load(mod):
    t = get(f"{mod}.trellis")
    return t, get(f"{mod}.suh").half(), get(f"{mod}.svh").half()


def lin(x, t, suh, svh, group_n=None):
    n = t.shape[1] * TILE
    return ops.exl3_linear(x, t.contiguous(), suh.reshape(-1, x.shape[-1]).contiguous(),
                           svh.contiguous(), group_n or [n], CB_MUL1)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-12)).item()


def record(mod, check, err, extra=""):
    ok = err < TOL
    results.append((mod, check, err, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {mod:42s} {check:8s} rel={err:.2e} {extra}", flush=True)


def xin(k, m=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, k), generator=g, device="cuda", dtype=torch.half) / k ** 0.5


def dequant(t, suh, svh):
    k = t.shape[0] * TILE
    eye = torch.eye(k, dtype=torch.half, device="cuda")
    return lin(eye, t, suh, svh)


def check_dequant(mod):
    t, suh, svh = load(mod)
    k, n, bits = t.shape[0] * TILE, t.shape[1] * TILE, t.shape[2] // TILE
    w = dequant(t, suh, svh)
    finite = bool(torch.isfinite(w).all())
    x = xin(k)
    err = rel(x.float() @ w.float(), lin(x, t, suh, svh)) if finite else float("inf")
    record(mod, "dequant", err, f"k={k} n={n} bits={bits} |W|={w.float().abs().mean():.3e}")
    return t, suh, svh


def check_column(mod, pad_n=0):
    """Column-parallel; pad_n output columns repeat the last pad_n real columns."""
    t, suh, svh = load(mod)
    k, n = t.shape[0] * TILE, t.shape[1] * TILE
    assert pad_n % HAD == 0
    if pad_n:
        t = torch.cat([t, t[:, -pad_n // TILE:]], dim=1)
        svh = torch.cat([svh, svh[-pad_n:]])
    full_n = n + pad_n
    shard = full_n // TP
    assert full_n % TP == 0 and shard % HAD == 0, (full_n, shard)
    x = xin(k)
    ref = lin(x, *load(mod))
    parts = [lin(x, t[:, r * shard // TILE:(r + 1) * shard // TILE], suh,
                 svh[r * shard:(r + 1) * shard]) for r in range(TP)]
    out = torch.cat(parts, dim=1)
    err = rel(out[:, :n], ref)
    if pad_n:
        err = max(err, rel(out[:, n:], ref[:, n - pad_n:]))
    record(mod, "column", err, f"n={n}+{pad_n} shard={shard}")


def check_row(mod, pad_k=0):
    """Row-parallel; pad_k input rows carry junk activations that zero suh must cancel."""
    t, suh, svh = load(mod)
    k, n = t.shape[0] * TILE, t.shape[1] * TILE
    assert pad_k % HAD == 0
    full_k = k + pad_k
    shard = full_k // TP
    assert full_k % TP == 0 and shard % HAD == 0, (full_k, shard)
    tp = torch.cat([t, torch.zeros_like(t[: pad_k // TILE])], dim=0) if pad_k else t
    sp = torch.cat([suh, torch.zeros(pad_k, dtype=suh.dtype, device=suh.device)]) if pad_k else suh
    x = xin(k)
    xp = torch.cat([x, xin(pad_k, seed=1) * 10], dim=1) if pad_k else x
    ref = lin(x, t, suh, svh)
    out = sum(lin(xp[:, r * shard:(r + 1) * shard].contiguous(),
                  tp[r * shard // TILE:(r + 1) * shard // TILE], sp[r * shard:(r + 1) * shard], svh)
              for r in range(TP))
    record(mod, "row", rel(out, ref), f"k={k}+{pad_k} shard={shard}")


def check_fused(mods, column_tp=False):
    ts, suhs, svhs = zip(*(load(m) for m in mods))
    bits = {t.shape[2] for t in ts}
    name = "+".join(m.rsplit(".", 1)[-1] for m in mods)
    label = f"{mods[0].rsplit('.', 1)[0]}.{name}"
    if len(bits) != 1:
        record(label, "fused", float("inf"), f"mixed bits {sorted(b // TILE for b in bits)}")
        return
    k = ts[0].shape[0] * TILE
    x = xin(k)
    ref = torch.cat([lin(x, t, s, v) for t, s, v in zip(ts, suhs, svhs)], dim=1)
    if not column_tp:
        ns = [t.shape[1] * TILE for t in ts]
        out = lin(x, torch.cat(ts, 1), torch.stack(suhs), torch.cat(svhs), ns)
        record(label, "fused", rel(out, ref), f"group_n={ns}")
        return
    parts = []
    for r in range(TP):
        rt, rv, ns = [], [], []
        for t, v in zip(ts, svhs):
            sn = t.shape[1] * TILE // TP
            assert sn % HAD == 0
            rt.append(t[:, r * sn // TILE:(r + 1) * sn // TILE])
            rv.append(v[r * sn:(r + 1) * sn])
            ns.append(sn)
        parts.append(lin(x, torch.cat(rt, 1), torch.stack(suhs), torch.cat(rv), ns).split(ns, 1))
    out = torch.cat([parts[r][i] for i in range(len(ts)) for r in range(TP)], dim=1)
    record(label, "fused", rel(out, ref), f"TP3 group_n={ns}")


def check_wo_a(block, groups_real=8, groups_virtual=9):
    base = f"{block}.attn.wo_a.slice"
    slices = [load(f"{base}.{g}") for g in range(groups_real)]
    slices += [slices[-1]] * (groups_virtual - groups_real)
    k = slices[0][0].shape[0] * TILE
    per_rank = groups_virtual // TP
    x = torch.stack([xin(k, seed=g) for g in range(groups_virtual)], dim=1)  # [T, G, k]
    ref = torch.stack([lin(x[:, g].contiguous(), *slices[g]) for g in range(groups_virtual)], 1)
    errs = []
    for r in range(TP):
        for j in range(per_rank):
            g = r * per_rank + j
            w = dequant(*slices[g])
            errs.append(rel(x[:, g].float() @ w.float(), ref[:, g]))
    record(f"{base}.*", "bmm", max(errs),
           f"k={k} n={slices[0][0].shape[1] * TILE} groups={groups_virtual} ({per_rank}/rank) via BF16 dequant")


def check_head(vocab_pad_to=192):
    """ParallelLMHead at TP3: vLLM pads the vocab to lcm(64, 3) = 192 and gives each
    rank a contiguous slice; the last rank's pad is zeroed through svh."""
    t, suh, svh = load("head")
    n = t.shape[1] * TILE
    padded = -(-n // vocab_pad_to) * vocab_pad_to
    shard = padded // TP
    assert shard % HAD == 0 and (padded - n) % HAD == 0, (n, padded, shard)
    x = xin(t.shape[0] * TILE)
    ref = lin(x, t, suh, svh)
    parts = []
    for r in range(TP):
        lo, hi = r * shard, min((r + 1) * shard, n)
        tr = torch.zeros((t.shape[0], shard // TILE, t.shape[2]), dtype=t.dtype, device=t.device)
        sv = torch.zeros(shard, dtype=svh.dtype, device=svh.device)
        tr[:, :(hi - lo) // TILE] = t[:, lo // TILE:hi // TILE]
        sv[:hi - lo] = svh[lo:hi]
        parts.append(lin(x, tr, suh, sv))
    out = torch.cat(parts, dim=1)
    err = max(rel(out[:, :n], ref), out[:, n:].abs().max().item())
    record("head", "vocab", err, f"n={n} padded={padded} shard={shard}")


def check_block(p):
    a, f = f"{p}.attn", f"{p}.ffn.shared_experts"
    for m in [f"{a}.wq_a", f"{a}.wkv", f"{a}.wq_b", f"{a}.wo_b", f"{f}.w1", f"{f}.w2", f"{f}.w3",
              f"{a}.indexer.wq_b", f"{a}.indexer.wk", f"{a}.compressor.wkv", f"{p}.engram.wkv",
              f"{p}.main_proj"]:
        if f"{m}.trellis" in IDX:
            check_dequant(m)
    check_fused([f"{a}.wq_a", f"{a}.wkv"])
    check_column(f"{a}.wq_b", pad_n=8 * 512)
    if f"{a}.wo_a.slice.0.trellis" in IDX:
        check_wo_a(p)
    check_row(f"{a}.wo_b", pad_k=1024)
    check_fused([f"{f}.w1", f"{f}.w3"], column_tp=True)
    check_row(f"{f}.w2")


torch.set_grad_enabled(False)
print(f"cuda_exl3 backend={ops.backend()} gpu={torch.cuda.get_device_name(0)}")
if sys.argv[2:] == ["all"]:
    n_layers = 1 + max(int(k.split(".")[1]) for k in IDX if k.startswith("layers."))
    n_mtp = len({k.split(".")[1] for k in IDX if k.startswith("mtp.") and ".attn.wq_a." in k})
    BLOCKS = [f"layers.{i}" for i in range(n_layers)] + [f"mtp.{i}" for i in range(n_mtp)]
    check_head()
else:
    BLOCKS = [f"layers.{i}" for i in LAYERS]
for p in BLOCKS:
    try:
        check_block(p)
    except Exception as e:  # keep going: one bad block must not hide the rest
        record(p, "error", float("inf"), repr(e)[:200])
    _files.clear()
    torch.cuda.empty_cache()

bad = [r for r in results if not r[3]]
print(f"\n{len(results) - len(bad)}/{len(results)} passed; max rel={max(r[2] for r in results):.2e}")
sys.exit(1 if bad else 0)
