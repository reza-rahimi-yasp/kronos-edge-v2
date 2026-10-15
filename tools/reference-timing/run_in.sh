#!/usr/bin/env bash
# run_in.sh <image> <result-name> [docker run extra args...] -- <command inside the container...>
# Runs one foreground disposable container with MODEL (model.py), NPZ (input.npz), and ARTIFACT (the artifact
# directory holding model.pt2 and kernel.py) mounted read-only, this directory at /work/scripts, and OUT at
# /work/out. Samples the GPU before, during (nvidia-smi every 500 ms), and after, and exits with the container's
# exit code. A diagnostic for docs/reference-timing.md, not part of the check.
set -u
: "${MODEL:?path to model.py}" "${NPZ:?path to input.npz}" "${ARTIFACT:?artifact directory}" "${OUT:?output directory}"
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=$1; NAME=$2; shift 2
EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do EXTRA+=("$1"); shift; done
shift || true
Q=name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,memory.used,clocks_throttle_reasons.active
mkdir -p "$OUT/gpu"
echo "== before" > "$OUT/gpu/$NAME.txt"; nvidia-smi --query-gpu=$Q --format=csv,noheader >> "$OUT/gpu/$NAME.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$OUT/gpu/$NAME.txt"
nvidia-smi --query-gpu=timestamp,$Q --format=csv,noheader -lms 500 > "$OUT/gpu/$NAME.samples.csv" &
SMI=$!
docker run --rm --name "reftiming-$NAME" --gpus all --shm-size 8g \
  -v "$MODEL":/work/model.py:ro -v "$NPZ":/work/input.npz:ro -v "$ARTIFACT":/work/artifact:ro \
  -v "$HERE":/work/scripts:ro -v "$OUT":/work/out \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_ext -e PYTHONDONTWRITEBYTECODE=1 \
  "${EXTRA[@]}" "$IMAGE" "$@"
RC=$?
kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
echo "== after (rc=$RC)" >> "$OUT/gpu/$NAME.txt"; nvidia-smi --query-gpu=$Q --format=csv,noheader >> "$OUT/gpu/$NAME.txt"
python3 - "$OUT/gpu/$NAME.samples.csv" >> "$OUT/gpu/$NAME.txt" <<'PY'
import sys, csv
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 8]
def col(i):
    vals = []
    for r in rows:
        try: vals.append(float(r[i].strip().split()[0]))
        except Exception: pass
    return vals
if rows:
    sm, temp, pw, ut = col(3), col(5), col(6), col(7)
    print(f"== during: {len(rows)} samples; sm MHz min/med/max {min(sm):.0f}/{sorted(sm)[len(sm)//2]:.0f}/{max(sm):.0f}; "
          f"temp max {max(temp):.0f}C; power max {max(pw):.0f}W; util max {max(ut):.0f}%; throttle reasons {sorted(set(r[9].strip() for r in rows))}")
PY
cat "$OUT/gpu/$NAME.txt"
exit $RC
