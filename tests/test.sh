#!/bin/sh
# UAThink offline validation. No network, no external services.
# Usage: sh tests/test.sh  (run from the repository root after `make`)
set -u
BIN=./uathink
MODEL=models/uathink_d1024h16_v2.bin
TMPD=/tmp/uathink-release-test-$$
mkdir -p "$TMPD"
fail() { echo "FAIL: $1"; rm -rf "$TMPD"; exit 1; }
pass() { echo "ok: $1"; }

[ -x "$BIN" ] || fail "binary missing (run make first)"
[ -f "$MODEL" ] || fail "model missing"

# 1. built-in selftest
"$BIN" selftest > "$TMPD/selftest.out" 2>&1 || fail "selftest exit code"
grep -q "SELFTEST PASS" "$TMPD/selftest.out" || fail "selftest output"
pass "selftest"

# 2. synthetic predictions match committed expectations (determinism)
"$BIN" predict "$MODEL" < examples/synthetic_predict.txt > "$TMPD/pred.out" 2>&1 || fail "predict exit code"
diff -u examples/synthetic.expected "$TMPD/pred.out" > /dev/null || fail "predict output differs from examples/synthetic.expected"
pass "synthetic predictions deterministic"

# 3. small train->eval roundtrip on synthetic data only
"$BIN" train examples/synthetic.tsv "$TMPD/tiny.bin" 2 > /dev/null || fail "tiny train exit code"
[ -f "$TMPD/tiny.bin" ] || fail "tiny model not written"
"$BIN" eval "$TMPD/tiny.bin" examples/synthetic.tsv > "$TMPD/eval.out" 2>&1 || fail "tiny eval exit code"
pass "synthetic train+eval roundtrip"

# 4. checkpoint integrity: corrupt a byte -> must refuse; good file must load
cp "$MODEL" "$TMPD/good.bin"
cp "$MODEL" "$TMPD/bad.bin"
printf '\001' | dd of="$TMPD/bad.bin" bs=1 seek=100 conv=notrunc status=none
"$BIN" predict "$TMPD/bad.bin" "x" > /dev/null 2>&1 && fail "corrupt model accepted"
pass "corrupt model rejected"
"$BIN" predict "$TMPD/good.bin" "x" > /dev/null 2>&1 || fail "good model rejected"
pass "good model loads"

# 5. malformed TSV rejected, no partial artifact
printf '9\tNot A Label\n' > "$TMPD/bad.tsv"
"$BIN" train "$TMPD/bad.tsv" "$TMPD/should-not-exist.bin" 2 > /dev/null || true
[ -f "$TMPD/should-not-exist.bin" ] && fail "model written from bad TSV"
pass "malformed TSV rejected cleanly"

rm -rf "$TMPD"
echo "ALL TESTS PASSED"
