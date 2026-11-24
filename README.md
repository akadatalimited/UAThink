# UAThink

Native x86-64 User-Agent identity classifier: a compact MLP that maps a
`User-Agent` string to one of five identity classes. Pure FASM assembly,
Linux syscalls only — no libc, no interpreter, no Python, no PyTorch, no
network access at runtime.

Classes: `crawler`, `human_like`, `known_bot`, `scanner`, `script_tool`.

UAThink classifies **identity claims** ("what kind of client says this string").
It does not decide hostile behaviour: identical strings can come from
legitimate browsers and from automation, so behavioural detection is a
separate layer that consumes — but never replaces — this signal.

## Requirements

- Linux x86-64 with the x86-64-v3 baseline (AVX/AVX2/FMA/BMI1/BMI2/LZCNT).
  The binary checks this at startup and exits with a clear error otherwise.
- [FASM](https://flatassembler.net/) 1.73+ to build.

## Build

```sh
make            # assembles uathink.asm -> ./uathink (needs ~128 MB FASM heap)
```

## Usage

All input is plain ASCII/UTF-8 text; no training or evaluation data is
required at runtime beyond the model file you pass in.

```sh
./uathink selftest                                   # built-in smoke test
./uathink train  <train.tsv> <model.bin> [epochs]    # train (default 12)
./uathink predict <model.bin> [ua...]                # classify (args or stdin)
./uathink eval   <model.bin> <eval.tsv>              # accuracy + per-class P/R/F1
./uathink hash   <ua>                                # hashed feature indices (debug)
./uathink bench                                       # ~2000-forward timing loop
```

TSV format: `<label>\t<user-agent>` per line, label `0..4`
(`0=crawler 1=human_like 2=known_bot 3=scanner 4=script_tool`),
one ASCII line per row, max 512 chars of UA. Malformed rows abort
training with an error; over-long or empty lines are skipped on prediction.

`predict` prints one line per input: the predicted label index followed by
the five class probabilities, e.g. `4 0.0001 0.0002 0.0003 0.0004 0.9990`.

Model checkpoints use the `UATH v1` container (64-byte header with magic,
dimensions, step count and FNV-64 integrity hashes, then raw FP32 weights).
Checkpoints are written via temp-file + fsync + atomic rename; corrupt or
truncated files are rejected and never touch the live model.

## Repository contents

- `uathink.asm` — the complete trainer + classifier (FASM, x86-64-v3).
- `Makefile` — build, test and bench targets.
- `models/uathink_d1024h16_v2.bin` — reference FP32 weights (see MODEL.md).
- `examples/synthetic.tsv` — small synthetic training/eval demo (all strings
  invented for documentation; no real traffic).
- `tests/test.sh` — offline validation: selftest, determinism, synthetic
  predictions, checkpoint integrity, malformed-input handling.
- `MODEL.md` — architecture, training methodology and measured metrics.
- `BENCHMARKS.md` — reference performance figures.

## Testing

```sh
make test
```

Runs entirely offline with no external services.

## License

MIT — see LICENSE.
