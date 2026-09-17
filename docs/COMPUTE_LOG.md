# Compute log

## Completed RTX 5090 revision

The separate revision matrix used one 32 GB RTX 5090 for 15 adapter-training
runs and 64 inference conditions (6,400 scene generations). All completion
markers and outputs are included under `results/revision/gpu/`.
Generation-batch timers sum to 5,362.286 seconds; reported trainer runtimes sum
to 2,133.1 seconds. The combined timed execution is **2.082 hours** on one GPU,
excluding model loading and setup outside those timers. This is not measured
device-active utilization or an energy estimate. See `REVISION_RESULTS.md` for
the counting method and `results/revision/environment.json` for the exact
software and model revision. Original-platform timing below remains separate.

## Reported expansion run

- Hardware: 2 × NVIDIA GeForce RTX 4090 D.
- Arithmetic used by training: 4-bit NF4 base weights with BF16 computation.
- First recorded stage start: 2026-09-16 23:09:25 UTC+08:00.
- Final recorded completion: 2026-09-17 03:07:11 UTC+08:00.
- End-to-end wall time: 3 h 57 min 46 s.
- Qwen3-8B cache repair interval: approximately 19 min between the first failed 8B start and the resumed stage. This interval was not model execution.

The run did not collect per-process power or utilization traces, so an exact sum of device-active seconds is unavailable. A transparent upper bound is about **7.3 GPU-hours**: two devices multiplied by the recorded wall-clock interval after subtracting the 8B cache-repair gap. The true device-active total is lower because orchestration, scoring, file I/O, and uneven completion leave one or both devices idle. This upper bound is provided instead of presenting wall time as measured utilization.

The nine newly trained adapters report a combined trainer runtime of 1,980 seconds (0.55 GPU-hours): three learning-curve adapters, five template-holdout adapters, and one 8B adapter. The remainder of the GPU-active budget is inference across prompt, source, fold, and scale conditions.

## Stage timestamps

| Stage | Recorded start (UTC+08:00) |
|---|---:|
| Prompt robustness | 2026-09-16 23:09:25 |
| Learning curves | 2026-09-16 23:42:18 |
| Template holdout | 2026-09-17 00:58:58 |
| Initial 8B attempt | 2026-09-17 02:14:41 |
| 8B resumed after cache repair | 2026-09-17 02:33:43 |
| Complete | 2026-09-17 03:07:11 |

Trainer runtimes and raw stage markers are retained in `results/expansion/logs/`, `results/expansion/orchestrator.log`, and `results/expansion/orchestrator_resume.log`.
