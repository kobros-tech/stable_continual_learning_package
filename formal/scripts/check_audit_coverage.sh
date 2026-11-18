#!/usr/bin/env bash
# Fails if a theorem/lemma in any formal/StableCL/**/*.lean file is not listed in Audit.lean.
# This only checks that declarations are *listed*; the authoritative check is the axiom audit
# that `lake build` runs on Audit.lean.
set -euo pipefail
cd "$(dirname "$0")/.."
status=0
while IFS= read -r f; do
  [ "$f" = "StableCL/Audit.lean" ] && continue
  for name in $(grep -hoE '^(theorem|lemma)[[:space:]]+[A-Za-z0-9_.]+' "$f" | awk '{print $2}'); do
    if ! grep -qE "StableCL\.${name}([[:space:]]|$)" StableCL/Audit.lean; then
      echo "not audited: ${name} (${f})" >&2
      status=1
    fi
  done
done < <(find StableCL -name '*.lean' | sort)
exit $status
