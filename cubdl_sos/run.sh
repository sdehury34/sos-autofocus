#!/usr/bin/env bash
# CUBDL speed-of-sound comparison: one command for each step (Linux, macOS, Windows Git Bash).
#   bash run.sh build     build the Docker image (GPU image if an NVIDIA GPU is available)
#   bash run.sh test      installation test on a synthetic file (true speed of sound 1500 m/s)
#   bash run.sh check     beamformer check on one file of each institution in data/
#   bash run.sh run       comparison on all files in data/ (resumes after a stop)
#   bash run.sh summary   summary table and figure in results/
#   bash run.sh all       build, test, check, run and summary
# Options (environment variables): CUBDL_DATA=/other/folder  CUBDL_CPU=1 (do not use the GPU)
#   Extra arguments after "run" go to run_sos_comparison.py, e.g.  bash run.sh run --files OSL002 MYO001
set -euo pipefail
cd "$(dirname "$0")"
export MSYS_NO_PATHCONV=1                       # Git Bash: do not convert the container paths
HERE="$PWD"; DATA="${CUBDL_DATA:-$HERE/data}"
if [ -z "${CUBDL_CPU:-}" ] && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  IMG=cubdl-sos-gpu; DOCKERFILE=Dockerfile.gpu; GPU=(--gpus all)
else
  IMG=cubdl-sos; DOCKERFILE=Dockerfile; GPU=()
fi

build() { echo "== build $IMG ($DOCKERFILE)"; docker build -f "$DOCKERFILE" -t "$IMG" .; }
need_image() { docker image inspect "$IMG" >/dev/null 2>&1 || build; }

test_install() {
  need_image; mkdir -p test
  docker run --rm "${GPU[@]}" -v "$HERE/test":/work --entrypoint python "$IMG" /tool/make_synthetic.py \
    --out /work/synthetic/SYN001.hdf5 --c 1500 --angles 31 --max-angle 15
  rm -f test/results/results.csv
  docker run --rm "${GPU[@]}" -v "$HERE/test":/work "$IMG" --data /work/synthetic --out /work/results/results.csv
  echo "== expected: grid_cf and gd_cf within about 5 m/s of 1500"
}

check() {
  need_image; mkdir -p test
  local files; files=$(ls "$DATA" | grep -i -E '\.(hdf5|h5)$' | awk '{p=substr($0,1,3); if(!(p in s)){s[p]=1; print}}')
  [ -n "$files" ] || { echo "no .hdf5 / .h5 files in $DATA"; exit 1; }
  for f in $files; do
    docker run --rm "${GPU[@]}" -v "$DATA":/data:ro -v "$HERE/test":/work "$IMG" --check "/data/$f"
  done
  echo "== the images are in test/check_<name>.png; the best NCC of each file must be 0.6 or more"
}

run_all() {
  need_image; mkdir -p results
  docker run --rm "${GPU[@]}" -v "$DATA":/data:ro -v "$HERE/results":/work/results "$IMG" \
    --data /data --out /work/results/results.csv "$@"
}

summary() {
  need_image
  docker run --rm -v "$HERE/results":/work/results --entrypoint python "$IMG" /tool/summarize.py /work/results/results.csv
}

cmd="${1:-help}"; shift || true
case "$cmd" in
  build) build ;;
  test) test_install ;;
  check) check ;;
  run) run_all "$@" ;;
  summary) summary ;;
  all) build; test_install; check; run_all "$@"; summary ;;
  *) sed -n '2,10p' "$0" ;;
esac
