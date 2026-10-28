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

Update, 2026-10-28: most of that was the drive. With the checkpoint
moved to nvme1 and both Engram tables read from it, nothing pinned in
RAM, real-code prefill is 7.2–7.4k, decode is unchanged or slightly
faster, and the server is ready in 4.9 minutes instead of 12. Pinning
both tables still adds up to about 20% on ~100k-token prompts; pinning
only one of them adds nothing. The stack now runs from nvme1 with both
pinned. See [Rebench from nvme1](#rebench-from-nvme1-2026-10-28).

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
| Second read-only mount of `/models/safetensors/DeepSeek-V4.1-Flash-1017-1bae0e10` | the HF cache entry is a symlink farm onto that tree, so the links have to resolve inside the container. Since 2026-10-28 the source is nvme1's direct xfs mount, not the `mergerfs` pool |
| `XDG_CACHE_HOME` and `TMPDIR` on the `/cache` volume | keeps JIT output off the 8 GiB root filesystem |
| `ENGRAM_RAM_LAYERS=1,14` (upstream default `1`) | both tables pinned; this host has 500 GB of RAM, his has 256 GB. The 2026-10-28 runs also cover empty, `1` and `14`; see below |

The upstream repo has no licence file, so I have not copied his compose,
scripts or patches here. Clone his repo at `bcf72fe` and apply the table
above.

## Engram layer 14 on a slow drive

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

At the time I pinned both tables in RAM rather than move the checkpoint.
The next section is the move.

## Rebench from nvme1 (2026-10-28)

The checkpoint now lives on nvme1 (sha256-verified after the copy) and
the container mounts nvme1's xfs filesystem directly, without
`mergerfs`. `ENGRAM_RAM_LAYERS` is empty, so both Engram tables are read
from the drive through the 2 GiB-per-GPU row caches. That is less RAM
than upstream, which keeps layer 1 pinned. nvme1 sits on a Gen4 x4 link
here. Image, compose and KV pool (2,185,581 tokens) are unchanged. I
recreated the container, then ran the same `run_all.sh` sequence once
`expert promotion armed and warmed` was logged. All 60 requests in the
run window came from the bench host.

| | Layer 14 on nvme0 (mergerfs) | Layers 1 and 14 in RAM | **Both on nvme1, direct** | Upstream (Gen5) |
|---|---|---|---|---|
| Real-code prefill, 105k / 259k / 102k tokens | 2,917 / 3,789 / 3,842 | 8,574 / 7,697 / 8,665 | **7,171 / 7,272 / 7,419** | 10,646 / 10,228 / 11,354 |
| Needle, 28,553 tokens (TTFT) | 9.79 s | 3.07 s | **3.89 s** | 2.29 s |
| Needle, 994,842 tokens | pass, 4,675 tok/s | pass, 6,747 tok/s | **pass, 6,456 tok/s** | pass, 7,193 tok/s |
| `llm_decode_bench` prefill 32k / 128k (synthetic filler) | 7,765 / 8,035 | 8,347 / 6,851 | **8,486 / 8,107** | 13,486 / 12,245 |
| Decode 32k, 1 / 4 users | 94.9 / 170.3 | 113.9 / 200.1 | **117.3 / 208.7** | 186.7 / 340.7 |
| Decode 128k, 1 / 4 users | 92.4 / 179.0 | 118.8 / 193.2 | **122.5 / 194.4** | 179.2 / 307.1 |
| Engine steps/s 32k, 1 / 4 users | 39.8 / 69.4 | 45.7 / 81.2 | **47.1 / 83.1** | 67.5 / 139.9 |
| Coding peak (3 runs, mean) | 165.8 | 181.2 | **178.2** | 271.8 |
| 4 concurrent short | 240.4 | 289.7 | **286.3** | 474.3 |
| Known answers | 5/5 | 5/5 | **5/5** | 5/5 |

Real-code prefill from nvme1 is 84–94% of the RAM-pinned run, and the
983k needle 96%. Decode is the same or a few percent faster, within what
I would call run-to-run noise. The 128k `llm_decode_bench` prefill cell
came in higher (8,107 against 6,851), but every prefill cell is one
sample. Against nvme0, the same disk path is 1.9–2.5x faster on real
code, which matches the fio gap: it is random 4 KiB IOPS that matter.

So on nvme1 the RAM pinning is no longer what makes this route usable,
though it still adds 6–20% to real-code prefill. The next two
subsections work out which table that comes from.

### Which table to pin, with the checkpoint on nvme1

Same day, same boot procedure. I ran two more configurations from nvme1
to see which table the prefill gap comes from: layer 1 pinned with
layer 14 read from the drive (upstream's default), and both pinned.
This server also backs my Hermes agent's LiteLLM lanes, and Hermes was
in use during the sweep. Where requests from outside the bench landed inside
a bench phase (the layer-1 needle and the both-pinned `llm_decode_bench`
prefill), I reran that phase once the server had seen no outside traffic
for 90 seconds. The numbers below are the clean reruns.

| | Neither pinned | Layer 1 pinned, layer 14 from nvme1 | Both pinned |
|---|---|---|---|
| Real-code prefill, 105k / 259k / 102k tokens | 7,171 / 7,272 / 7,419 | 7,117 / 7,567 / 7,530 | **8,663 / 7,538 / 8,591** |
| Needle, 28,553 tokens (TTFT) | 3.89 s | 3.63 s | **3.02 s** |
| Needle, 994,842 tokens | pass, 6,456 tok/s | pass, 6,399 tok/s | **pass, 6,654 tok/s** |
| `llm_decode_bench` prefill 32k / 128k | 8,486 / 8,107 | 8,524 / 7,434 | 9,355 / 8,330 |
| Decode 32k, 1 / 4 users | 117.3 / 208.7 | 111.1 / 193.3 | 127.7 / 207.5 |
| Decode 128k, 1 / 4 users | 122.5 / 194.4 | 118.9 / 204.4 | 121.3 / 196.6 |
| Engine steps/s 32k, 1 / 4 users | 47.1 / 83.1 | 47.1 / 81.3 | 48.3 / 86.7 |
| Decode C=1, repo cells (0 / 16k / 32k / 128k) | 120.9 / 118.3 / 121.5 / 120.0 | 114.6 / 119.8 / 112.4 / 112.0 | 117.8 / 124.1 / 115.3 / 114.7 |
| Coding peak (3 runs, mean) | 178.2 | 182.1 | 177.9 |
| 4 concurrent short | 286.3 | 281.4 | 304.6 |
| Known answers | 5/5 | 5/5 | 5/5 |
| Container start to promotion armed | 4.9 min, 4.4 min | 5.0 min, 5.3 min | 5.6 min |
| `Loading weights took` / model load (rank 0) | 77 s / 108 s, 65 s / 93 s | 79 s / 139 s, 81 s / 147 s | 95 s / 182 s |
| Hypervisor RAM in use while serving, less ZFS ARC | 117 GiB | 209 GiB | 295 GiB |

Pinning layer 1 on its own buys nothing measurable: real-code prefill
and both needles land where the unpinned run did, and it costs about
92 GiB of host RAM and 30–50 s more model load. Pinning layer 14 as
well brings the two ~100k-token code prompts up 14–22% (8.6–8.7k
against 7.1–7.5k) and the short needle's TTFT down from 3.6–3.9 s to
3.0 s. The 259k-token prompt and the 994k needle barely move, so the
reads matter most on prompts that are mostly new text at moderate
length. Both-pinned matches the 2026-10-25 RAM-pinned run from nvme0
(8,574 / 7,697 / 8,665), so with the tables in RAM the checkpoint's
drive no longer shows in prefill.

Decode does not depend on placement. Engine steps/s is 47–48 in all
three. The tok/s spread at 1 user follows the accept length, which at
temperature 1 ran 2.36 to 2.64 tokens per step between runs.

Boot times include a warm page cache to some degree: the checkpoint
stays cached on nvme1's xfs mount between restarts, which `mergerfs`
did not allow. The two unpinned boots, 4.9 and 4.4 minutes, give an
idea of the spread. The RAM column is the hypervisor's used memory
minus its ZFS ARC, taken while each configuration was serving; it
includes every other guest on the host, so only the differences mean
anything (about 92 GiB for layer 1 and 86 GiB for layer 14).

### Layer 14 alone

That left layer 14 on its own (`ENGRAM_RAM_LAYERS=14`), which would
have been the cheaper way to get the gain. It is not. I ran a short
check, the three real-code prompts and then his `bench.py`, straight
after boot, and repeated the same two phases straight after a
both-pinned boot so the order matched. Neither run saw outside traffic.

| Straight after boot | Layer 14 pinned, layer 1 from nvme1 | Both pinned |
|---|---|---|
| Real-code prefill, 105k / 259k / 102k tokens | 6,446 / 7,233 / 7,449 | **8,304 / 8,054 / 8,104** |
| Needle, 28,553 tokens (TTFT) | 3.63 s | **3.06 s** |
| Quick decode (short prompt) | 170.6 / 166.8 (58.2 / 58.6 steps/s) | 167.4 / 179.8 (58.7 / 60.7 steps/s) |
| 4 concurrent short | 307.2 | 281.1 |
| Known answers | 5/5 | 5/5 |
| Container start to promotion armed | 5.0 min | 5.9 min |
| Hypervisor RAM in use while serving, less ZFS ARC | 209 GiB | 295 GiB |

Pinned on its own, layer 14 lands where the unpinned and layer-1 runs
did, for about the same 92 GiB. The gain only shows once neither
table is read from the drive. My guess is that any table left on disk
puts the io_uring reader on the prefill path, and that wait costs about
the same whether it serves one table or two. I have not profiled that.
The both-pinned short check lands within the sweep's both-pinned
range, so the order of the bench phases is not what made the
difference.

So on this host it is both tables or neither. The stack now runs with
`ENGRAM_RAM_LAYERS=1,14` from nvme1: about 178 GiB of host RAM for
14–22% more prefill on ~100k-token prompts and roughly a minute more
boot. If that RAM is wanted elsewhere, unpinned from nvme1 is the
fallback, and decode is the same either way.

## Against upstream

Same scripts and the same `llm_decode_bench` version (0.7.8) as his
`bench-20261019.3/` run, same cells, from a fresh restart. The ratio
column is for the RAM-pinned run; the nvme1 rebench is next to it.

| | This box, RAM-pinned (2026-10-25) | This box, nvme1 (2026-10-28) | Upstream (Gen5) | Ratio (RAM-pinned) |
|---|---|---|---|---|
| `llm_decode_bench` prefill 32k / 128k | 8,347 / 6,851 | 8,486 / 8,107 | 13,486 / 12,245 | 0.62 / 0.56 |
| Real-code prefill (3 prompts) | 8,574 / 7,697 / 8,665 | 7,171 / 7,272 / 7,419 | 10,646 / 10,228 / 11,354 | 0.75–0.81 |
| 983k needle | 6,747 tok/s, peak 97,339 MiB | 6,456 tok/s, peak 97,333 MiB | 7,193 tok/s, peak 97,319 MiB | 0.94 |
| Decode 32k, C1 / C4 | 113.9 / 200.1 | 117.3 / 208.7 | 186.7 / 340.7 | 0.61 / 0.59 |
| Decode 128k, C1 / C4 | 118.8 / 193.2 | 122.5 / 194.4 | 179.2 / 307.1 | 0.66 / 0.63 |
| Engine steps/s 32k, C1 / C4 | 45.7 / 81.2 | 47.1 / 83.1 | 67.5 / 139.9 | 0.68 / 0.58 |
| Coding peak (3 runs, mean) | 181.2 | 178.2 | 271.8 | 0.67 |
| Quick decode (short prompt) | 170.1 / 174.0 (59.5 / 58.9 steps/s) | 166.6 / 167.2 (57.7 / 55.9 steps/s) | 228.5 / 223.3 (79.1 / 79.4) | 0.75 |
| 4 concurrent short | 289.7 | 286.3 | 474.3 | 0.61 |
| Known answers | 5/5 | 5/5 | 5/5 | |

Accept length at 1 user was 2.4–2.7 tokens per step on the bench filler
here, across both runs, against 2.7–2.8 in his run, so part of the
tok/s gap is acceptance at temperature 1. Steps/s is the cleaner comparison: about two thirds of
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

| | Kraken 1017, nvme1 (2026-10-28) | Kraken 1017, Engram in RAM (2026-10-25) | UVA rebench (2026-10-08) | EXL3 P4 |
|---|---|---|---|---|
| Weights | official 1017 `1bae0e10` | official 1017 `1bae0e10` | official `fb2764a5` | Pollard 3.51 bpw |
| Decode C=1 (0 / 16k / 32k / 128k) | **120.9 / 118.3 / 121.5 / 120.0** | 111.8 / 119.2 / 111.1 / 105.6 | 76.3 / 75.9 / 74.6 / 74.5 | 55.5 / 56.0 / skipped / — |
| Prefill, bench filler (16k / 32k / 128k) | **8,282 / 8,397 / 7,778** | 6,160 / 8,969 / 8,140 | 4,650 / 5,274 / 4,629 | 5,933 / skipped / failed |
| Prefill, real code | 7,171–7,419 | 7,697–8,665 | not measured | not measured |
| 983k needle | pass, 6,456 tok/s | pass, 6,747 tok/s | 1M decode-only cell, no prefill | — |
| Max context | 1M | 1M | 1M | 32k |
| Speculation | DSpark, 5 drafts | DSpark, 5 drafts | off | DSpark |

The RAM-pinned run's 16k prefill cell is a single sample and came in
under its 32k one (6,160 against 8,969). The disk-backed run had 7,618
at 16k and the nvme1 run 8,282, so I would read it as noise. Every
prefill cell in this repo is one sample.

On the nvme1 run, decode is 56–63% ahead of UVA at every context (the
RAM-pinned run was 42–57%). On the filler, prefill is 1.6–1.8x UVA's.
The UVA route has not been measured on real code, so I am not putting
those two side by side.

## Bring-up

| | Disk-backed boot (nvme0) | RAM-pinned boot (nvme0) | nvme1 boot (2026-10-28) |
|---|---|---|---|
| Container start to `expert promotion armed and warmed` | 17.7 min | 12.0 min | **4.9 min** |
| `Loading weights took` (rank 0) | 337 s | 424 s, includes pinning layer 14 | **77 s** |

On the nvme1 boot, the API server was up 4 min 47 s after container
start and expert promotion finished 6 s later. Model load took 108 s in
total, the 77 s weight read plus the DSpark drafter, and engine init (profile, KV
cache, warm-up) took 94 s, including 8 s of graph capture. Graph capture
on the RAM-pinned boot took 7 s. All three boots reused the same
`runtime-cache` volume. I did not drop the page cache before the nvme1
boot. The checkpoint is 475 GiB against about 140 GiB of page cache on
the hypervisor, so most of it must have come off the drive, but this is
not a controlled cold boot. I did not time a fresh-volume boot; upstream
says it adds about 10 minutes of kernel compilation.

## What I have not run

His quality suite (lavd, hotel-lights, 128k/900k retrieval, long
generation, vision) and the coding task. Here the known-answer checks
and both needles passed. Nothing past 4 concurrent requests, which is
the configured limit.

## Files

- `bench/baseline-mergerfs/`: the first run, layer 14 read from the
  checkpoint through mergerfs.
- `bench/engram-ram/`: the run with both Engram tables in RAM.
- `bench/nvme1-disk/`: the 2026-10-28 rebench, checkpoint on nvme1's
  direct mount and both Engram tables read from it.
- `bench/nvme1-l1ram/` and `bench/nvme1-bothram/`: the same day, from
  nvme1, with layer 1 pinned and with both tables pinned. Phases that
  had outside traffic hold the clean rerun.
- `bench/nvme1-l14ram/` and `bench/nvme1-bothram-short/`: the short
  checks (`prefill_real_code.json` and `quick.json` only) with layer 14
  pinned on its own and with both pinned, each run straight after boot.

Each holds `llm_decode_bench_prefill.json`, `llm_decode_bench.json`
(his cells plus the coding peak), `repo_cells.json` (this repo's cells),
and the outputs of his `bench.py`, `prefill_real.py` and `bench_long.py`
(`quick.json`, `prefill_real_code.json`, `needle_983k.json`). The
real-code corpus is `local-inference-lab/flashinfer` at `5b065abe`, the
same as his.

---

*Measured 2026-10-25, nvme1 rebench 2026-10-28. 3x RTX PRO 6000 96 GB (SM120) on PCIe 4.0 x16,
NODE topology, driver 615.71.09. GPUs exclusive to this stack during
every run.*
