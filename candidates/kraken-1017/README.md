# DeepSeek-V4.1-Flash-1017 at TP3: BoxTops4Coochie's Kraken recipe

This is someone else's recipe, run on my box. The recipe, image and TP3
port are
[BoxTops4Coochie](https://github.com/BoxTops4Coochie)'s
([RTX-PRO-6000-TP3-Inference-Recipes](https://github.com/BoxTops4Coochie/RTX-PRO-6000-TP3-Inference-Recipes),
folder `DeepSeek-V4.1-Flash Kraken-tp3-20261019`, commit `bcf72fe`). It
serves the **original** `deepseek-ai/DeepSeek-V4.1-Flash-1017` weights
(FP8 dense, MXFP4 routed experts, nothing requantized) at TP3 with the
full 1M window, on the Local Inference Lab karmic-kraken vLLM image. Please
read his README for how it works. Below is what I changed to run it here
and what I measured.

The short version: on this Gen4 box it is the fastest route in the repo,
for both decode and prefill, and it serves 1M context. Before it got
there, one change was needed. With Engram layer 14 read from disk, as
upstream ships it, real-text prefill was about 3k tok/s here. With both
Engram tables pinned in host RAM it was 7.7–8.7k.

## What runs

| | |
|---|---|
| Image | `azallaza/ds41-kraken-tp3:20261019.3` (`sha256:c4c2de52…`), his build |
| Weights | official `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, the revision the [1017 EXL3 notes](../exl3-1017/README.md) checked shard by shard |
| Serve | Dockge stack `deepseek-v41-flash-1017-kraken-tp3` on the inference host, port 8019 |
| Context / KV | 1,048,576-token window, 2,185,581-token KV pool (2.08x 1M), 4 sequences |
| Speculation | DSpark, 5 drafts, NVFP4 drafter head; server default temperature 1.0, top-p 1.0 |
| Expert offload | layers 20–25 parked in host memory, hottest 12–14 of 24 chunks per region promoted to GPU after `/health` |
| `tp:0` all-reduce | `['B12X_PCIE', 'PYNCCL']`, B12X oneshot first (`ep:0` PYNCCL) |

His compose is used as is, apart from these changes for this host:

| Change | Why |
|---|---|
| Host port 8019 bound to the host's LAN IP instead of `:15015` on all interfaces | the inference host's firewall only opens LAN-IP ports |
| `nofile` ulimit removed | this unprivileged container cannot raise it to 1,048,576 under runc |
| Second read-only mount of `/models/safetensors/DeepSeek-V4.1-Flash-1017-1bae0e10` | the HF cache entry is a symlink farm onto that tree, so the links have to resolve inside the container |
| `XDG_CACHE_HOME` and `TMPDIR` on the `/cache` volume | keeps JIT output off the 8 GiB root filesystem |
| `ENGRAM_RAM_LAYERS=1,14` (upstream default `1`) | see below; this host has 500 GB of RAM, his has 256 GB |

The upstream repo has no licence file, so I have not copied his compose,
scripts or patches here. Clone his repo at `bcf72fe` and apply the table
above.

## Engram layer 14 belongs in RAM on this host

Upstream keeps Engram layer 1 pinned in RAM and reads layer 14 from the
checkpoint with io_uring, with a 2 GiB-per-GPU row cache in front. His
host has 256 GB, and both tables do not fit next to the parked experts.
On this host the checkpoint was on nvme0, behind a FUSE `mergerfs` union
(`nvme0:sas1:sas2:rpool`), and that read path was slow.

The synthetic filler in `llm_decode_bench` repeats, so the row cache
covers most of it, and its prefill looked fine (7.3–8.0k tok/s). Real
source code, and his random-word needle, missed the cache and fell to
about 3k:

| | Layer 14 on disk (mergerfs) | Layers 1 and 14 in RAM | Upstream (Gen5, layer 14 on disk) |
|---|---|---|---|
| Real-code prefill, 105k / 259k / 102k tokens | 2,917 / 3,789 / 3,842 | **8,574 / 7,697 / 8,665** | 10,646 / 10,228 / 11,354 |
| Needle, 28,553 tokens (TTFT) | 9.79 s | **3.07 s** | 2.29 s |
| Needle, 994,842 tokens | pass, 4,675 tok/s | **pass, 6,747 tok/s** | pass, 7,193 tok/s |
| Decode 32k, 1 / 4 users | 94.9 / 170.3 | **113.9 / 200.1** | 186.7 / 340.7 |
| Host RAM in use while serving | 162 GB | 250 GB | ~150 GiB |

Decode moved as well. The quick-test decode went from about 50 to 59
engine steps/s, so layer 14's row lookups were on the decode path too.

The reader turns every 256-byte row miss into one 4 KiB `O_DIRECT`
io_uring read (queue depth 64 per reader), so what matters is random-read
IOPS, not bandwidth. It turned out to be the drive. 4 KiB random reads
with `fio` (io_uring, direct, 3 jobs, read-only):

| Path | QD 1 per job | QD 64 per job |
|---|---|---|
| nvme0 raw device (generic "PCIe SSD", fw `ELFMC1.0`, Gen4 x4) | 6.1k IOPS, 0.48 ms | 14.2k IOPS, 13.5 ms |
| nvme0 through `mergerfs`, shard 48 (layer 14's table) | | 40.8k IOPS, 4.5 ms |
| nvme1 raw device (Crucial T710) | 94.5k IOPS, 0.03 ms | 747k IOPS, 0.25 ms |
| nvme1 direct xfs mount, from the container | | 624k IOPS, 0.30 ms |

The `mergerfs` row beats nvme0's raw device because `mergerfs` runs
with `cache.files=partial`. The reader's `O_DIRECT` does not get through
to the drive, so part of shard 48 was still in the hypervisor's page
cache from the boot-time load. nvme0 has no SMART warnings, sits at
46 °C, and was otherwise idle. It is just slow at small random reads.

I have not served layer 14 from nvme1. With the RAM to spare, pinning it
is simpler. The stack now runs with `ENGRAM_RAM_LAYERS=1,14` in its
`.env`. If that RAM is needed elsewhere, the fallback is the checkpoint
on nvme1, mounted directly rather than through `mergerfs`.

## Against upstream

Same scripts and the same `llm_decode_bench` version (0.7.8) as his
`bench-20261019.3/` run, same cells, from a fresh restart. These are the
RAM-pinned numbers.

| | This box (Gen4) | Upstream (Gen5) | Ratio |
|---|---|---|---|
| `llm_decode_bench` prefill 32k / 128k | 8,347 / 6,851 | 13,486 / 12,245 | 0.62 / 0.56 |
| Real-code prefill (3 prompts) | 8,574 / 7,697 / 8,665 | 10,646 / 10,228 / 11,354 | 0.75–0.81 |
| 983k needle | 6,747 tok/s, peak 97,339 MiB | 7,193 tok/s, peak 97,319 MiB | 0.94 |
| Decode 32k, C1 / C4 | 113.9 / 200.1 | 186.7 / 340.7 | 0.61 / 0.59 |
| Decode 128k, C1 / C4 | 118.8 / 193.2 | 179.2 / 307.1 | 0.66 / 0.63 |
| Engine steps/s 32k, C1 / C4 | 45.7 / 81.2 | 67.5 / 139.9 | 0.68 / 0.58 |
| Coding peak (3 runs, mean) | 181.2 | 271.8 | 0.67 |
| Quick decode (short prompt) | 170.1 / 174.0 (59.5 / 58.9 steps/s) | 228.5 / 223.3 (79.1 / 79.4) | 0.75 |
| 4 concurrent short | 289.7 | 474.3 | 0.61 |
| Known answers | 5/5 | 5/5 | |

Accept length at 1 user was 2.4–2.6 tokens per step on the bench filler
here, against 2.7–2.8 in his run, so part of the tok/s gap is acceptance
at temperature 1. Steps/s is the cleaner comparison: about two thirds of
his rate at 1 user.

The rest is most likely the PCIe link. The non-promoted experts in
layers 20–25 are read from host memory on every step. Average PCIe
receive traffic over his decode run was about 43 GB/s; over mine it was
about 23–25 GB/s, on a Gen4 x16 link. His cards also run at 350 W with a
+6000 memory offset. Here GPUs 0 and 1 are Max-Q cards capped at 300 W,
GPU 2 runs at 350 W, and I did not set his memory offset.

## Against the other routes on this box

The repo's usual cells, one request at a time: contexts 0 / 16k / 32k /
128k, 30 seconds, 2,048 max tokens, no decode warm-up. The harness is
newer than the one behind the UVA and EXL3 rows (0.7.8 against 0.4.32),
with the cell settings matched.

| | Kraken 1017, Engram in RAM | UVA rebench (2026-10-08) | EXL3 P4 |
|---|---|---|---|
| Weights | official 1017 `1bae0e10` | official `fb2764a5` | Pollard 3.51 bpw |
| Decode C=1 (0 / 16k / 32k / 128k) | **111.8 / 119.2 / 111.1 / 105.6** | 76.3 / 75.9 / 74.6 / 74.5 | 55.5 / 56.0 / skipped / — |
| Prefill, bench filler (16k / 32k / 128k) | **6,160 / 8,969 / 8,140** | 4,650 / 5,274 / 4,629 | 5,933 / skipped / failed |
| Prefill, real code | 7,697–8,665 | not measured | not measured |
| 983k needle | pass, 6,747 tok/s | 1M decode-only cell, no prefill | — |
| Max context | 1M | 1M | 32k |
| Speculation | DSpark, 5 drafts | off | DSpark |

The 16k prefill cell is a single sample and came in under the 32k one
(6,160 against 8,969). The disk-backed run had 7,618 at 16k, so I would
read it as noise. Every prefill cell in this repo is one sample.

Decode is 42–57% ahead of UVA at every context. On the filler,
prefill is 1.3–1.8x UVA's. The UVA route has not been measured on real
code, so I am not putting those two side by side.

## Bring-up

| | Disk-backed boot | RAM-pinned boot |
|---|---|---|
| Container start to `expert promotion armed and warmed` | 17.7 min | 12.0 min |
| `Loading weights took` (rank 0) | 337 s | 424 s, includes pinning layer 14 |

Graph capture on the RAM-pinned boot took 7 s. Both boots reused the
same `runtime-cache` volume. I did not time a
fresh-volume boot; upstream says it adds about 10 minutes of kernel
compilation.

## What I have not run

His quality suite (lavd, hotel-lights, 128k/900k retrieval, long
generation, vision) and the coding task. Here the known-answer checks
and both needles passed. Nothing past 4 concurrent requests, which is
the configured limit.

## Files

- `bench/baseline-mergerfs/`: the first run, layer 14 read from the
  checkpoint through mergerfs.
- `bench/engram-ram/`: the run with both Engram tables in RAM.

Each holds `llm_decode_bench_prefill.json`, `llm_decode_bench.json`
(his cells plus the coding peak), `repo_cells.json` (this repo's cells),
and the outputs of his `bench.py`, `prefill_real.py` and `bench_long.py`
(`quick.json`, `prefill_real_code.json`, `needle_983k.json`). The
real-code corpus is `local-inference-lab/flashinfer` at `5b065abe`, the
same as his.

---

*Measured 2026-10-25. 3x RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16,
NODE topology, driver 615.71.09. GPUs exclusive to this stack during
every run.*
