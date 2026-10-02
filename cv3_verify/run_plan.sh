#!/usr/bin/env bash
# Run the CosyVoice3 voice-leak measurement matrix.
#
#   run_plan.sh [--dry-run] [--phase control|rest|all]
#
# One git worktree per commit (editable install, --no-deps), one server per
# (commit, mode[, talker overlay]). Every failure is recorded and the cell is
# skipped; nothing is retried. Outputs go to $OUT (default /home/ubuntu/cv3_out),
# outside the repository.
#
# Matrix (N=120 requests per run, 44 runs):
#   c85f1a4, bbee488 : {async, no-async} x K{2,8} x (C=1 seed 1, C=8 seeds 1-3)
#   1ea0ee3          : {async, no-async} x K=2 x C=8 seeds 1-3
#   bbee488 extra    : async, K=8, C=8 seeds 1-3, with talker overlay
#                      max_num_seqs=4, and separately enable_prefix_caching=true
# The c85f1a4 positive control runs first; if no scorer finds a wrong voice in
# its C=8 runs, FINDINGS.md is written and the plan stops.

set -uo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$KIT/.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
VLLM="$VENV/bin/vllm"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
WT_ROOT="${WT_ROOT:-/home/ubuntu/cv3_wt}"
OUT="${OUT:-/home/ubuntu/cv3_out}"
MODEL="${MODEL:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
PORT="${PORT:-8091}"
N="${N:-120}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1200}"
export HF_HOME="${HF_HOME:-/home/ubuntu/hf_home}"
# FlashInfer's top-k/top-p sampler JIT-compiles with nvcc, which this VM lacks.
# The PyTorch sampler is used for every commit instead; neither bug under test
# (conditioning routing, embed_input_ids batch order) goes through the sampler.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

C_CTRL=c85f1a4
C_FIX=1ea0ee3
C_HEAD=bbee488

DRY=0
PHASE=all
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --phase) PHASE="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
  shift
done
case "$PHASE" in control|rest|all) ;; *) echo "bad --phase $PHASE" >&2; exit 64 ;; esac

SERVERS_CONTROL=("$C_CTRL async -" "$C_CTRL no-async -")
SERVERS_REST=("$C_FIX async -" "$C_FIX no-async -"
              "$C_HEAD async -" "$C_HEAD no-async -"
              "$C_HEAD async mns4" "$C_HEAD async prefix")

# "K C seed" triples for one server.
cells_for() {
  local commit="$1" overlay="$2"
  if [[ "$overlay" != "-" ]]; then
    printf '%s\n' "8 8 1" "8 8 2" "8 8 3"
  elif [[ "$commit" == "$C_FIX" ]]; then
    printf '%s\n' "2 8 1" "2 8 2" "2 8 3"
  else
    printf '%s\n' "2 1 1" "2 8 1" "2 8 2" "2 8 3" "8 1 1" "8 8 1" "8 8 2" "8 8 3"
  fi
}

log() { echo "[$(date -u +%FT%TZ)] $*"; }

server_tag() { local t="$1_$2"; [[ "$3" != "-" ]] && t="${t}_$3"; echo "$t"; }

model_dir() {
  HF_HUB_OFFLINE=1 "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL', local_files_only=True))"
}

prepare_worktree() {
  local commit="$1" wt="$WT_ROOT/$1" envdir="$OUT/env/$1"
  mkdir -p "$envdir"
  if [[ ! -d "$wt" ]]; then
    git -C "$REPO" worktree add --detach "$wt" "$commit" || return 1
  fi
  "$UV" pip install --python "$PY" --no-deps -e "$wt" > "$envdir/install.log" 2>&1 || return 1
  (cd / && "$PY" - "$wt" > "$envdir/env.txt" 2>&1 <<'EOF'
import os, sys, subprocess, torch, vllm, vllm_omni
wt = os.path.realpath(sys.argv[1])
print("worktree", wt)
print("worktree_head", subprocess.check_output(["git", "-C", wt, "rev-parse", "HEAD"], text=True).strip())
print("vllm.__version__", vllm.__version__)
print("vllm_omni.__file__", vllm_omni.__file__)
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("python", sys.version.split()[0])
print("VLLM_USE_FLASHINFER_SAMPLER", os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"))
ok = os.path.realpath(vllm_omni.__file__).startswith(wt + os.sep)
print("vllm_omni_from_worktree", ok)
sys.exit(0 if ok else 3)
EOF
  )
  local rc=$?
  "$UV" pip freeze --python "$PY" > "$envdir/pip_freeze.txt" 2>/dev/null
  nvidia-smi > "$envdir/nvidia-smi.txt" 2>&1
  cat "$envdir/env.txt"
  return $rc
}

save_failure() {
  local tag="$1" stage="$2" logf="$3" f="$OUT/findings/${1}_${2}.txt"
  mkdir -p "$OUT/findings"
  {
    echo "server: $tag   stage: $stage   time: $(date -u +%FT%TZ)"
    echo "log: $logf"
    echo; echo "=== tracebacks ==="
    grep -n -A60 'Traceback (most recent call last)' "$logf" 2>/dev/null | tail -400
    echo; echo "=== last 150 log lines ==="
    tail -150 "$logf" 2>/dev/null
  } > "$f"
  log "failure recorded: $f"
}

SERVER_PID=""
start_server() {
  local commit="$1" mode="$2" overlay="$3" logf="$4" wt="$WT_ROOT/$1"
  local deploy="vllm_omni/deploy/cosyvoice3.yaml" extra=()
  if [[ "$overlay" != "-" ]]; then
    deploy="$OUT/overlays/${commit}_${overlay}.yaml"
    mkdir -p "$OUT/overlays"
    {
      echo "base_config: $wt/vllm_omni/deploy/cosyvoice3.yaml"
      echo "stages:"
      echo "  - stage_id: 0"
      case "$overlay" in
        mns4) echo "    max_num_seqs: 4" ;;
        prefix) echo "    enable_prefix_caching: true" ;;
      esac
    } > "$deploy"
  fi
  [[ "$mode" == "no-async" ]] && extra+=(--no-async-chunk)
  if ss -ltn | grep -q ":$PORT "; then
    log "port $PORT already in use; refusing to start"
    return 1
  fi
  # setsid forks when its caller leads a process group, so $! would be a
  # short-lived wrapper. The child shell writes its own PID and execs the
  # server, so SERVER_PID is the leader of the server's own process group.
  rm -f "$logf.pid"
  # shellcheck disable=SC2016  # $$ and $@ must expand in the child shell
  (cd "$wt" && HF_HUB_OFFLINE=1 setsid sh -c 'echo $$ > "$0"; exec "$@"' "$logf.pid" \
      "$VLLM" serve "$MODEL" --omni --trust-remote-code --port "$PORT" \
      --deploy-config "$deploy" "${extra[@]}" > "$logf" 2>&1 &)
  for _ in $(seq 50); do [[ -s "$logf.pid" ]] && break; sleep 0.1; done
  SERVER_PID="$(cat "$logf.pid")"
  log "server pid $SERVER_PID (VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER): vllm serve $MODEL --omni --trust-remote-code --port $PORT --deploy-config $deploy ${extra[*]}"
  local t0=$SECONDS
  while (( SECONDS - t0 < STARTUP_TIMEOUT )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then log "server exited during startup"; return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then
      log "healthy after $((SECONDS - t0))s"; return 0
    fi
    sleep 5
  done
  log "startup timeout ${STARTUP_TIMEOUT}s"
  return 1
}

stop_server() {
  [[ -z "$SERVER_PID" ]] && return 0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SERVER_PID" 2>/dev/null
  for _ in $(seq 120); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( used < 2000 )) && ! ss -ltn | grep -q ":$PORT " && break
    sleep 1
  done
  if kill -0 "$SERVER_PID" 2>/dev/null || ss -ltn | grep -q ":$PORT "; then
    log "server $SERVER_PID did not stop (GPU used ${used:-?} MiB); aborting"
    exit 7
  fi
  log "server $SERVER_PID stopped (GPU used ${used:-?} MiB)"
  SERVER_PID=""
}
trap stop_server EXIT

TOTAL_RUNS=0
run_server() {
  local commit="$1" mode="$2" overlay="$3"
  local tag; tag="$(server_tag "$commit" "$mode" "$overlay")"
  local sdir="$OUT/runs/$tag" logf="$OUT/logs/server_$tag.log"
  local cells; mapfile -t cells < <(cells_for "$commit" "$overlay")
  TOTAL_RUNS=$((TOTAL_RUNS + ${#cells[@]}))
  if (( DRY )); then
    echo "server $tag: worktree $WT_ROOT/$commit; vllm serve $MODEL --omni --trust-remote-code --port $PORT" \
         "--deploy-config $([[ $overlay == - ]] && echo vllm_omni/deploy/cosyvoice3.yaml || echo "$OUT/overlays/${commit}_${overlay}.yaml")" \
         "$([[ $mode == no-async ]] && echo --no-async-chunk)"
    for c in "${cells[@]}"; do
      read -r k cc seed <<< "$c"
      echo "  run K=$k C=$cc seed=$seed N=$N -> $sdir/K${k}_C${cc}_s${seed}"
    done
    return 0
  fi
  mkdir -p "$sdir" "$OUT/logs"
  log "=== server $tag ==="
  if ! prepare_worktree "$commit"; then
    save_failure "$tag" env "$OUT/env/$commit/env.txt"; log "worktree/env check failed for $commit; aborting"; exit 5
  fi
  if ! start_server "$commit" "$mode" "$overlay" "$logf"; then
    save_failure "$tag" startup "$logf"; stop_server; return 0
  fi
  if ! "$PY" "$KIT/measure_voice_leak.py" run --voices "$KIT/voices_k2.json" --n 1 -c 1 --seed 999 \
        --commit "$commit" --mode "$mode" --overlay "${overlay/-/}" --label smoke --timeout 300 \
        --out "$sdir/smoke" || ! grep -q '"wav"' "$sdir/smoke/requests.jsonl" 2>/dev/null; then
    cp "$sdir/smoke/requests.jsonl" "$OUT/findings/${tag}_smoke_requests.jsonl" 2>/dev/null
    save_failure "$tag" smoke "$logf"; stop_server; return 0
  fi
  local done_dirs=()
  for c in "${cells[@]}"; do
    read -r k cc seed <<< "$c"
    local rdir="$sdir/K${k}_C${cc}_s${seed}"
    log "run $tag K=$k C=$cc seed=$seed"
    "$PY" "$KIT/measure_voice_leak.py" run --voices "$KIT/voices_k$k.json" --n "$N" -c "$cc" --seed "$seed" \
        --commit "$commit" --mode "$mode" --overlay "${overlay/-/}" --label "$tag" --out "$rdir"
    done_dirs+=("$rdir")
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      save_failure "$tag" "died_K${k}_C${cc}_s${seed}" "$logf"
      log "server died; remaining cells of $tag skipped"
      break
    fi
  done
  stop_server
  log "scoring ${#done_dirs[@]} runs of $tag"
  "$PY" "$KIT/measure_voice_leak.py" score "${done_dirs[@]}" --model-dir "$MODEL_DIR" 2>&1 | tail -n 20
}

main() {
  local servers=()
  [[ "$PHASE" != rest ]] && servers+=("${SERVERS_CONTROL[@]}")
  if (( ! DRY )); then
    mkdir -p "$OUT"
    MODEL_DIR="$(model_dir)" || { log "model $MODEL not in HF cache ($HF_HOME)"; exit 6; }
    log "model dir $MODEL_DIR; out $OUT; phase $PHASE"
    for c in "$C_CTRL" "$C_FIX" "$C_HEAD"; do git -C "$REPO" rev-parse "$c^{commit}"; done > "$OUT/commits.txt"
  fi
  for s in "${servers[@]}"; do read -r a b c <<< "$s"; run_server "$a" "$b" "$c"; done
  if [[ "$PHASE" != rest ]] && (( ! DRY )); then
    "$PY" "$KIT/measure_voice_leak.py" gate --runs-root "$OUT/runs" --commit "$C_CTRL" --findings "$OUT/FINDINGS.md"
    local g=$?
    if (( g == 3 )); then log "positive control negative: see $OUT/FINDINGS.md; stopping"; exit 3; fi
    if (( g != 0 )); then log "gate could not evaluate the positive control (rc=$g); stopping"; exit $g; fi
  fi
  if [[ "$PHASE" != control ]]; then
    for s in "${SERVERS_REST[@]}"; do read -r a b c <<< "$s"; run_server "$a" "$b" "$c"; done
  fi
  if (( DRY )); then
    echo "dry-run: phase $PHASE, $TOTAL_RUNS runs planned"
    return 0
  fi
  "$PY" "$KIT/measure_voice_leak.py" report --runs-root "$OUT/runs" --out-dir "$OUT/report" > /dev/null
  log "report: $OUT/report/report.md"
}

main
