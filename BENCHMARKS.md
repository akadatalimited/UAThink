# UAThink reference benchmarks

Measured on a reference x86-64-v3 machine (FP32 build, single thread).
Figures are indicative; rerun `make bench` and `tests/test.sh` locally.

| Workload | Result |
|---|---|
| Forward pass, short synthetic UA (`./uathink bench`) | ~3,900 cycles/row |
| Forward pass, browser-length UA | ~11,000–15,000 cycles/row |
| `eval`, 7,902 rows | ~0.03 s (~250k rows/s) |
| `train`, 1 epoch over ~47k rows | ~0.6 s (~78k samples/s) |
| Executable size | ~7 KB |
| Model parameter file | ~261 KB (66,661 FP32 weights + 64 B header) |

No network, GPU, or external services are used in any path
(verified by syscall audit: only `read/write/open/close/fsync/rename/exit`
plus CPUID/RDTSC).
