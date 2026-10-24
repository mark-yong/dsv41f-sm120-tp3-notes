# DeepSeek-V4.1-Flash at TP3 on 3x RTX PRO 6000 (SM120)

I wanted to see which TP3 route made more sense on my 3x RTX PRO 6000 box:
the 3.51 bpw EXL3 checkpoint, or the official DeepSeek checkpoint with
decoder-half UVA expert offload. I got both running and benchmarked them on
the same machine.

The short version: EXL3 is faster at short-context prefill. UVA is much
faster for single-user decode and is the only one of the two that I got
working at 128k-1M context. The rest of the repo is the configs,
measurements, and failed attempts behind that result.

Update, 2026-10-25: BoxTops4Coochie's Kraken recipe for the 1017
revision beats both on this box. Decode at one user is 106–119 tok/s,
prefill on real code 7.7–8.7k tok/s, and a 994,842-token needle passes.
It serves the official weights with DSpark on. On this host it needed
both Engram tables pinned in RAM. See
[Kraken 1017](#kraken-1017-2026-10-25) below.

- **Candidate A, EXL3 3.51 bpw** ([candidates/exl3/README.md](candidates/exl3/README.md)):
  the Pollard checkpoint on Jake Tempo's cuda-exl3 TP3 port.
- **Candidate B, official checkpoint + decoder-half UVA** ([candidates/uva/NOTES.md](candidates/uva/NOTES.md)):
  recipe by peterkilfeather; the UVA expert offload re-targeted to the
  decoder half.
- **Head-to-head** ([candidates/uva/benchmarks/COMPARE.md](candidates/uva/benchmarks/COMPARE.md)):
  same-box comparison of both final configs plus Pete's Gen5 numbers.
- **1017 revision, in progress** ([candidates/exl3-1017/README.md](candidates/exl3-1017/README.md)):
  porting a 2.9 bpw EXL3 quant of DeepSeek-V4.1-Flash-1017 to the same
  TP3 runtime. Kernel-level checks so far, no serving numbers yet.
- **Candidate C, Kraken 1017** ([candidates/kraken-1017/README.md](candidates/kraken-1017/README.md)):
  [BoxTops4Coochie](https://github.com/BoxTops4Coochie)'s recipe for the
  official 1017 checkpoint on the karmic-kraken image, run on this box
  and compared with his Gen5 numbers.

## Result

| | EXL3 3.51 bpw (P4) | official checkpoint + UVA | Kraken 1017 (BoxTops4Coochie), Engram in RAM |
|---|---|---|---|
| Serve | `candidates/exl3/reproduce/compose/` | `candidates/uva/compose/` | his compose at `bcf72fe` plus the deltas in `candidates/kraken-1017/` |
| Port | 8014 | 8015 | 8019 |
| Checkpoint | Pollard 3.51 bpw, 48 shards, 428.5 GiB, sha-pinned | official `fb2764a5` | official 1017 `1bae0e10` |
| Image | GHCR `dsv41f-tempo-sm120-tp3@sha256:ddd31bc7…` (build recipe in repo) | LIL r38 `localinferencelab/vllm@sha256:f41ca8bb…` | `azallaza/ds41-kraken-tp3:20261019.3` (`sha256:c4c2de52…`) |
| Max context | 32k (131k boots, OOMs under bench) | 1M window (1M decode cell passed at 1,034,830-token context) | 1M window (994,842-token needle passed) |
| KV pool | 4 GiB fp8 at 32k; a 7 GiB (reported as 3,479,680 tokens) attempt booted but OOMed under bench | 2.72M tokens realized on this box (2.60x the 1M ceiling); Pete's Gen5 host: 2.85 GiB ≈ 2.6M tokens | 2,185,581 tokens (2.08x 1M), fixed 2.25 GiB per GPU |
| Prefill (8k / 16k / 32k) | 6,140 / 5,933 / skipped | 4,005 / 4,508 / 4,425 | — / 6,160 / 8,969; real code 7,697–8,665 |
| Decode C=1 (0 / 32k / 1M) | 55.5 / skipped / — | 75.8 / 74.8 / 71.7 | 111.8 / 111.1 / — (128k: 105.6) |
| Bring-up to READY | ~25 min | ~14–17 min on both timed boots (InstantTensor still ~4.5 min); graph capture only 47 s → 13 s warm; page-cache-hot restart including weight load **not** timed | 12–18 min to `expert promotion armed and warmed` (two boots, warm kernel cache) |
| Speculation | DSpark on (accept 2.3-2.4 tok/step) | DSpark off (it regressed under offload when Pete tried it) | DSpark on, 5 drafts (accept 2.4–2.6 tok/step) |

On my 3x RTX PRO 6000 Gen4 box, the Kraken 1017 recipe is now the one I
would serve: fastest decode and prefill in the repo, at 1M context.
Between the two original candidates, UVA is the better choice for decode
and long context; EXL3 wins short-context prefill. Pete's Gen5 host (same
image and flags) prefills 6.8-7.4k, above both of my routes, which is a
good reminder that PCIe generation changes this picture. I would not
choose EXL3 over UVA on this hardware unless I specifically needed EXL3:
if I did, the working config and image build are documented in
candidates/exl3/, so there is no need to repeat the failed bring-up
attempts.

Numbers are from `llm-inference-bench` sustained-decode runs with no other
GPU workloads running. Community figures (Pete's Gen5 gist) are cited with
their provenance, never mixed with my own measurements.

One shared bottleneck worth knowing, on the September r38 image: B12X's
PCIe all-reduce rejects world size 3, so TP3 falls back to PYNCCL. Pete
measured that collective at roughly 40% of decode kernel time on his
box. Writeup:
[b12x-410-pcie-ar-world3.md](candidates/uva/b12x-410-pcie-ar-world3.md).
I re-ran the same UVA recipe once a newer image actually accepted world
size 3. Decode did not budge.

## Kraken 1017 (2026-10-25)

[BoxTops4Coochie](https://github.com/BoxTops4Coochie) published a TP3
recipe for the 1017 revision
([RTX-PRO-6000-TP3-Inference-Recipes](https://github.com/BoxTops4Coochie/RTX-PRO-6000-TP3-Inference-Recipes),
`DeepSeek-V4.1-Flash Kraken-tp3-20261019`). It serves the official
weights on Local Inference Lab's karmic-kraken image. Expert layers
20–25 start in host memory, and the most-routed experts move back to the
GPU once the server is healthy. DSpark runs at 5 drafts. He measured it
on a Gen5 host. I ran his image and his benchmark scripts, with the
compose changed only where this host needed it.

The first run, with Engram layer 14 read from the checkpoint as upstream
ships it, prefilled real source code at only 2.9–3.8k tok/s. The
checkpoint was on nvme0, which manages 14k random 4 KiB reads per second
where nvme1 manages 747k, and layer 14 is read one 4 KiB block per row.
This host has 500 GB of RAM, so I pinned both Engram tables
(`ENGRAM_RAM_LAYERS=1,14`):

| | Layer 14 from disk | Layers 1 and 14 in RAM | Upstream, Gen5 |
|---|---|---|---|
| Real-code prefill (105k / 259k / 102k tokens) | 2,917 / 3,789 / 3,842 | 8,574 / 7,697 / 8,665 | 10,646 / 10,228 / 11,354 |
| 994,842-token needle | pass, 4,675 tok/s | pass, 6,747 tok/s | pass, 7,193 tok/s |
| Decode 32k, 1 / 4 users | 94.9 / 170.3 | 113.9 / 200.1 | 186.7 / 340.7 |
| Decode 128k, 1 / 4 users | 92.4 / 179.0 | 118.8 / 193.2 | 179.2 / 307.1 |
| Engine steps/s at 32k, 1 user | 39.8 | 45.7 | 67.5 |

With the tables in RAM, this box runs at about two thirds of his engine
step rate for decode and three quarters of his real-code prefill. His
host has PCIe Gen5, and the parked experts are read over that link every
step. Against the other routes here, on the repo's usual cells, decode
at one user is 106–119 tok/s, against UVA's ~75 and EXL3's ~55. More
detail, the compose deltas, and the caveats:
[candidates/kraken-1017/README.md](candidates/kraken-1017/README.md).

## Correction: EXL3 router bias (2026-10-18)

The EXL3 image behind every EXL3 number in this repo never loads the MoE
router's correction bias. The cuda-exl3 MoE method takes over the
loader of every parameter on the expert layer, the router bias included,
and then declines the bias quietly. Experts were selected without it.
The throughput numbers are what the engine did; output quality was never
measured, and with this bug it cannot be assumed. The UVA route does not
use cuda-exl3 and is not affected. The fix is open upstream as
[Zeuss5/cuda-exl3#8](https://github.com/Zeuss5/cuda-exl3/pull/8). Walkthrough:
[candidates/exl3/README.md](candidates/exl3/README.md#router-bias-not-loaded-found-2026-10-18).

## Rebench after #410 (2026-10-08)

I kept everything else the same: official `fb2764a5`, 8.13 GiB of
decoder-half experts offloaded (ordinals 20–23), DSpark off. Image:
`ghcr.io/local-inference-lab/vllm@sha256:edc0998c63df59eada70438b998dec60858a04d95c95541257c094e547ae591c`.

This time `tp:0` came up on the B12X PCIe oneshot path. PYNCCL was still
listed behind it. `ep:0` never left PYNCCL.

The first try, at util 0.98, built a 3,115,460-token KV pool and then
died while b12x was preparing the MLA kernels, about 128 MiB short. I
dropped util to 0.97 so it would finish. That boot's KV pool was
2,101,050 tokens (1.94 GiB left for cache). Same bench order as
September: context 0, then 16k, 32k, 128k, one request at a time, 30
seconds, 2,048 max tokens.

| | This rebench | September UVA (PYNCCL, util 0.98, KV 2.72M) |
|---|---|---|
| Image | `sha256:edc0998c63df…` | r38 `sha256:f41ca8bb…` |
| `tp:0` all-reduce | B12X PCIe oneshot | PYNCCL |
| KV pool | 2,101,050 tokens | 2.72M tokens |
| Decode C=1 (0 / 16k / 32k / 128k) | 76.3 / 75.9 / 74.6 / 74.5 | 75.8 / 75.4 / 74.8 / 73.8 |
| Prefill (16k / 32k / 128k) | 4,650 / 5,274 / 4,629 | 4,508 / 4,425 / 4,319 |

Decode is the same ~75 tok/s as September. The oneshot path was not what
was holding this box back. I left DSpark off for this run. A two-token
try is the next section. Longer table:
[candidates/uva/benchmarks/COMPARE.md](candidates/uva/benchmarks/COMPARE.md).

## Two-token DSpark (2026-10-08)

Same image and checkpoint as the rebench. I turned DSpark on at 2
speculative tokens. Pete's K7 measurement on this kind of offload was
42–58 tok/s, so the shorter window was the thing I wanted to check.

The 8.13 GiB offload did not leave room for the drafter. At util 0.96
the KV line was −2.0 GiB after the draft model loaded. util 0.99 never
started: each GPU had 93.16 GiB free, and 0.99 asks for 94.11.

I raised the offload and kept util at 0.97. `--cpu-offload-gb 10`
parked 10.37 GiB and left 1.2 GiB of KV (1,290,975 tokens).
`--cpu-offload-gb 11` parked 11.16 GiB and left 2.02 GiB (2,147,680
tokens). Bench was context 0, then 16k, then 32k.

| | 10.37 GiB off | 11.16 GiB off | Rebench, DSpark off |
|---|---|---|---|
| Decode C=1 (0 / 16k / 32k) | 60.2 / 61.5 / 60.0 | 56.0 / 57.5 / 56.3 | 76.3 / 75.9 / 74.6 |
| Accept length | 2.01 / 2.00 / 2.02 | 1.99 / 2.01 / 1.97 | — |
| Steps/s | 29.9 / 30.7 / 29.8 | 28.2 / 28.6 / 28.6 | — |
| Prefill (16k / 32k) | 5,146 / 4,310 | 4,603 / 5,137 | 4,650 / 5,274 |

The draft was taken. Accept length sat on 2. The step rate was about
30 per second, against the single-token run at about 75, so decode
landed around 60. Another gigabyte of experts in RAM made it a little
slower, about 56, which is the same band as Pete's K7 note. The
DSpark-off rebench is still the faster decode.

## EXL3 follow-up (2026-10-08)

Same P4 config as September, then with DSpark removed. On EXL3 the
draft is in VRAM with the experts, so this is a different question from
the UVA offload above.

| | P4, DSpark on | P4, DSpark off |
|---|---|---|
| Prefill (8k / 16k) | 6,140 / 5,933 | 6,514 / 6,568 |
| Decode C=1 (0 / 16k) | 55.5 / 56.0 | 45.4 / 44.1 |

The draft is worth about 10 tok/s of decode here. Prefill is a bit
faster without it. Accept length on the DSpark-on run was 2.3–2.4
tokens per step.

The 131k window (7 GiB KV, seqs 2, batched 2048) booted with DSpark
off and prefilled at 5,888 tok/s (32k) and 5,852 (64k). The harness
did not run 128k: that prompt is 131,071 tokens, and the filter keeps
only contexts at or under the model length minus 64. Two boots that
left DSpark on, one with a smaller CUDA graph and one with a 32k-only
prefill, reached READY and died on the first prefill request. I did
not build a 3.25 bpw quant. Tempo's current recipe still serves this
3.51 bpw checkpoint, and the 131k failure with DSpark on is activation
memory, not weight size.

I then ran that same 131k boot again and measured decode as well.
Prefill was 6,276 / 6,001 / 5,585 tok/s at 32k / 64k / 96k. Decode at
concurrency 1 was 42.6 tok/s at 32k and 40.2 at 64k, with no errors.
That is the no-draft rate, a little under the 45 measured at short
context with DSpark off, and well under the 55 from P4 with DSpark on.

The 96k prefill stayed near the 5.8k from the first 64k cell, so EXL3
has a long-prefill number to set next to UVA (about 4.5k–5.3k at
16k–32k). Decode did not fall apart. The EXL3 setup I would keep for
serving is still the 32k P4 config with DSpark on. The 131k window
with DSpark off is the long-prefill configuration.

## Test setup

- Hardware: 3x RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16, NODE topology;
  NVIDIA driver 615.71.09, CUDA 13.4 user mode. Pete's gist numbers come
  from a PCIe Gen5 x16/x16/x8 host; the two are labeled separately
  everywhere in this repo.
- Bench: [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  (Martin Vit) @ `d115fee` (2026-09-01); 30 s sustained decode per cell,
  2,048 max output tokens, engine-default sampling. The Kraken 1017 runs
  use `llm_decode_bench` 0.7.8 (`a0bff1a`), the version behind
  BoxTops4Coochie's published numbers, with the cell settings above
  matched for the repo's own cells.
- The 1M cell on the UVA engine was decode-only over a 1,034,830-token
  context (TTFT 6.0 s): a context-capacity result, not a standalone 1M
  prefill or a retrieval test. Standalone 256k/512k/1M prefill was not
  collected on this engine.
- Output quality (EXL3 vs official weights) is not tested yet; what I ran
  was smoke tests and the bench tables. The EXL3 runs also had the
  router-bias bug above.

## What I'd test next

Standalone UVA prefill at 64k and above. EXL3's 96k cell landed at
5,585 tok/s, which is close enough to compare. The 256k/512k/1M UVA
cells so far are decode-only.

EXL3 P4 again on an image with the router-bias fix, plus a top-k expert
agreement check against the official checkpoint, so the EXL3 numbers
have a quality reading next to them.

For Kraken 1017: layer 14 read from nvme1's direct mount instead of RAM,
to see whether a fast drive matches the RAM-pinned numbers. Also
BoxTops4Coochie's quality suite on this box (lavd, hotel-lights, long retrieval, long
generation).

## Credits

A warm thanks to the many members of the Local Inference Lab Discord;
this repo is mostly assembled from their work.

- **Jake Tempo** ([jakejharris/jspark3-deepseek](https://github.com/jakejharris/jspark3-deepseek)):
  the cuda-exl3 TP3 serving runtime the EXL3 candidate is built on.
- **tonyd2wild** ([DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)):
  the upstream TP3 patch set the rebuild applies.
- **bot-lab-21**: the EXL3 3.51 bpw checkpoint (WestWaters' Pollard method,
  turboderp's EXL3 format).
- **[peterkilfeather](https://github.com/peterkilfeather)** (Pete): the
  decoder-half UVA offload overlay, compose, and the Gen5 measurements
  ([gist](https://gist.github.com/peterkilfeather/7af387df07ff0df2327b8fd7f77596ed))
  that candidate B reproduces.
- **[BoxTops4Coochie](https://github.com/BoxTops4Coochie)**: the Kraken
  TP3 recipe for DeepSeek-V4.1-Flash-1017
  ([RTX-PRO-6000-TP3-Inference-Recipes](https://github.com/BoxTops4Coochie/RTX-PRO-6000-TP3-Inference-Recipes)):
  the `azallaza/ds41-kraken-tp3` image and its TP3 port, in-place
  hot-first expert promotion, the move to FlashInfer's CSF scale
  compression, the Engram row cache, the benchmark and quality scripts, and the Gen5 measurements
  candidate C is compared with. Thank you for publishing all of it,
  including the configurations that did not work.
- **[infernix](https://github.com/infernix)**: the `ds41f-tp3`
  expert-parking design on karmic-kraken (host-backed VMM regions for
  the decoder-half expert layers, promotion behind an arm file) that the
  Kraken recipe builds on.
- **[voipmonitor](https://github.com/voipmonitor)**: the docker containers
  ([blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker))
  and recipes ([rtx6kpro](https://github.com/local-inference-lab/rtx6kpro))
  most LIL projects run on, the InstantTensor loader and
  `voipmonitor/vllm` image line behind the UVA route, and the
  [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
  harness every number in this repo comes from.
- **[lukealonso](https://github.com/lukealonso)** (Luke Alonso): the
  [b12x](https://github.com/local-inference-lab/b12x) kernel backend (this
  node's vLLM stacks launch with `BACKEND=b12x`), the PCIe oneshot and
  even-width all-reduce work, and the quants in
  [quant-toolkit](https://github.com/local-inference-lab/quant-toolkit).
- **[DeepSeek](https://github.com/deepseek-ai)**:
  [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
  DSpark, CED, and Engram.
- **[NNNtrance](https://github.com/NNNtrance/cuda-exl3)**: the cuda-exl3
  router-bias and fused-combine split-k fixes
  ([#8](https://github.com/Zeuss5/cuda-exl3/pull/8),
  [#7](https://github.com/Zeuss5/cuda-exl3/pull/7)).
- **LibertAIDAI**: the
  [DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  quant the 1017 notes use.
- **anoane**: DeepSeek-V4.1 support for exllamav3
  ([turboderp-org/exllamav3#449](https://github.com/turboderp-org/exllamav3/pull/449)).
- **vLLM**, **FlashInfer**, **TileLang**, **NVIDIA CUDA / SM120**.

## Licence and notices

Apache-2.0. Third-party notices preserved at the root ([NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)); per-candidate provenance
in each candidate directory. The UVA overlay is peterkilfeather's, pinned
by sha256 in candidates/uva/NOTES.md. BoxTops4Coochie's recipe repo has
no licence file, so candidates/kraken-1017/ links his files at commit
`bcf72fe` instead of copying them.
