# DeepSeek-V4.1-Flash-1017 at TP3: 2.9 bpw EXL3 port (in progress)

DeepSeek published a new revision of V4.1-Flash, `1017`. I want to know
whether a smaller EXL3 quant of it fixes what held the 3.51 bpw Pollard
route back on this box: no room for long context with DSpark on. The quant is
[LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
at revision `c178a012`. The runtime is the same Jake Tempo cuda-exl3 TP3
image as [../exl3/README.md](../exl3/README.md), with the router-bias fix.

Nothing here serves yet. So far I have checked that the checkpoint's
tensors can run on that runtime at TP3, and I have written the conversion
for the parts that cannot. No throughput or quality numbers.

## Checkpoints

| | Revision | Shards | Size | Checked |
|---|---|---|---|---|
| Official `deepseek-ai/DeepSeek-V4.1-Flash-1017` | `1bae0e10` | 48 | 476 GiB | sha256 of all 48 shards against the HF LFS oids |
| LibertAIDAI 2.9 bpw EXL3 | `c178a012` | 43 | 196.3 GiB | sha256 of all 43 shards against the HF LFS oids |

The quant carries no Engram tables (0.2 GiB of Engram projections only).
The tables come from the official 1017 shards 47 and 48 (189.1 GiB),
loaded into pinned host RAM as on the Pollard route.

The official 1017 tree is a drop-in for `fb2764a5` on the UVA route.
`config.json`, `tokenizer.json`, `tokenizer_config.json` and the tensor
index (same weight map) are byte-identical. Every one of the 48 weight
shards changed, Engram included, so no shard from `fb2764a5` can be
reused, and Engram tables from one revision must not be served with the
other.

## Why it might fit where Pollard did not

Pollard is 428.5 GiB, and 189.1 GiB of that is Engram, so its trunk is
about 239 GiB. The 2.9 bpw trunk is 196.3 GiB: about 43 GiB less, roughly
14 GiB per GPU at TP3. The Pollard 131k attempt with DSpark on missed by
tens of MiB (see [../exl3/README.md](../exl3/README.md), P6). This is a
file-size estimate. The real margin will be in the boot log.

## TP3 shape

The quant was made for a 2-way split, and some dimensions do not divide
by 3. The Tempo runtime already pads attention to 72 virtual heads and 9
`o_groups` for Pollard. The question is whether that padding works on
EXL3 tensors, where everything moves in 128-wide Hadamard blocks.

| Tensor | TP3 split | Virtual-head pad on EXL3 |
|---|---|---|
| Routed and shared experts | 2,304 = 18 × 128, 768 per rank | none needed |
| `wq_b` (64 heads × 512) | uneven | repeat the last group: 4,096 output columns = 32 whole blocks, copy those trellis tiles and `svh` entries |
| `wo_a` (8 slices) | uneven | one tensor per group in this quant, so slice 8 is a copy of slice 7 |
| `wo_b` (8 × 1,024 inputs) | uneven | 1,024 zero input columns = 8 whole blocks; zero `suh` cancels them |
| `head` (vocab 129,280) | uneven | vocab padded to lcm(64, 3) = 192: 129,408, 43,136 per rank (337 blocks) |
| Indexer, compressor, `wq_a`, `wkv`, Engram `wkv` | replicated | none |

Do not pad by writing zero trellis codes: they do not decode to zero.

## Kernel check: 579/579

[`tools/exl3-tp3-kernel-check.py`](tools/exl3-tp3-kernel-check.py) runs
inside the Tempo image, on one GPU, against real shards. It uses only
the image's `cuda_exl3` extension, with no vLLM engine and no
exllamav3. Every TP3 rank is simulated and compared with the unsharded
result. Per module, it checks:

- the dense weight from `exl3_linear(I)` against the kernel on random input;
- column splits with the virtual-head pad;
- row splits with the zero-`suh` pad, with junk activations ×10 on the pad
  so a leak would show;
- fused projections (`wq_a`+`wkv`, shared `w1`+`w3`) as one trellis with
  one `suh` row per shard;
- `wo_a` as 9 groups, 3 per rank, through BF16;
- the head with the 192-padded vocab.

Over all 40 layers, the three MTP blocks and the head: 579 of 579
passed, max relative error 6.32e-4 (fp16 accumulation, tolerance 5e-3).
The `wq_b` virtual-head pad is bit-exact. Not covered: the routed experts
through the fused MoE kernel (including the 2-bit layers 18–22), and
anything inside the vLLM engine.

## What has to be BF16

The Tempo runtime reads some modules as plain weights rather than through a
linear method. Those need dense weights. I read the code paths in the
image:

| Module | Code path in the Tempo image | Serve as |
|---|---|---|
| `wq_a`+`wkv` | fused, replicated linear | EXL3 |
| `wq_b` | column-parallel linear; with EXL3 the FP8 query-quant fusion is off and it is called plainly | EXL3 + pad |
| `wo_a` | `torch.bmm` over `weight.view(groups, o_lora, -1)` when the weight is not FP8 | BF16 |
| `wo_b` | row-parallel linear | EXL3 + pad |
| `indexer.wq_b` | replicated linear | EXL3 |
| `indexer.wk` | replicated linear built with `quant_config=None` | BF16 |
| `compressor.*` | raw `torch.mm` on the fused weight | BF16 |
| `engram.wkv` | replicated linear | EXL3 |
| shared experts | the EXL3 config refuses EXL3 shared experts | BF16 |
| `head` | vocab-parallel head | EXL3 |
| routed experts | EXL3 fused MoE | EXL3 |

[`tools/dequant-nonrouted.py`](tools/dequant-nonrouted.py) writes those
as a BF16 sidecar. BF16 holds the 2.9 bpw values exactly, so this adds no
error beyond the quant itself. For the full tree it replaced 484 EXL3
modules with 183 tensors, 5.56 GiB. The BF16 modules then go in
`ignored_layers` so they load unquantized.

Still to write before a boot: EXL3 padding in `virtual_heads.py`, a
generated `quantization_config.json` (BF16 modules out of
`tensor_storage`, the `mtp.*` modules in; the published one lists none of
the 1,198 MTP EXL3 modules), a merged index (quant, sidecar, official
Engram shards), and the 72-head / 9-group `config.json`.

## Plan for the first boot

From the Pollard work, not re-tested here:

- Start from the P4 knobs: custom all-reduce with P2P, Engram tables in
  pinned DDR, PIECEWISE CUDA graphs, FlashInfer autotune off.
- Try 131k with DSpark on first, since that is what the smaller trunk
  might unlock.
- Same `llm-inference-bench` cells as the rest of the repo.
- Look for the router-bias warning in the boot log; it should be absent.

## The other path: exllamav3 PR #449

[turboderp-org/exllamav3#449](https://github.com/turboderp-org/exllamav3/pull/449)
(open) adds DeepSeek-V4.1 to exllamav3. It splits layers across cards
instead of tensor parallelism, so none of the padding above applies. The
author reports 46.4 tok/s decode on three cards, 68.5 with the DSpark head
through TabbyAPI, and about 1.4k tok/s prefill. Those are lower than UVA
on this box (~75 decode, 4.5–5.3k prefill), but it serves this quant with
no porting.

Building it for SM120 on the same vLLM base image took four tries, in case
you hit the same:

1. The base image has no `git`. I fetch pinned commits as GitHub
   tarballs instead.
2. `cusparse.h` is missing from `/usr/local/cuda/include`. In this base
   image the cuSPARSE/cuSOLVER headers ship only in torch's pip CUDA
   wheels (`nvidia/cu13/include`).
3. Putting that directory on `CPATH` breaks every `.cu` file: it shadows
   nvcc's own `crt/host_runtime.h` with a different version
   (`macro "__cudaLaunch" passed 2 arguments, but takes just 1`).
4. Symlinking only the headers missing from `/usr/local/cuda/include`
   builds cleanly with `TORCH_CUDA_ARCH_LIST=12.0a`.

Built, not yet run.

---

*Recorded 2026-10-18. Same box as the rest of the repo: 3x RTX PRO 6000
96 GB SM120 on PCIe 4.0 x16, NODE topology.*
