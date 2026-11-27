#!/bin/bash
# Gate: every dependency-light unit test passes (and the torch ones, where torch is present).
set -u
cd "$(dirname "$0")/../.." || exit 1
fail=0
for t in tests/test_*.py; do
    if python "$t" >/tmp/gate_$$.log 2>&1; then echo "ok   $t"
    else echo "FAIL $t"; tail -5 /tmp/gate_$$.log; fail=1; fi
done
rm -f /tmp/gate_$$.log
exit $fail
