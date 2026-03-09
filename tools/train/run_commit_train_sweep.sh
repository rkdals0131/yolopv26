#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/train/run_commit_train_sweep.sh \
    --repo /abs/path/to/YOLOPv26 \
    --dataset-root /abs/path/to/dataset_root \
    --out-root /abs/path/to/output_root \
    COMMIT [COMMIT ...] [-- EXTRA_TRAIN_ARGS...]

What it does:
  - keeps the current checkout untouched
  - creates one detached git worktree per commit
  - runs train_pv26.py with a fixed absolute dataset root and run name
  - verifies completion by checking:
      1) process exit code == 0
      2) log contains "[pv26] finished."
      3) checkpoints/latest.pt exists
      4) checkpoints/best.pt exists
  - writes a TSV summary under --out-root

Defaults:
  --repo           current working directory
  --dataset-root   <repo>/datasets/pv26_v1_bdd_full
  --out-root       <repo>/runs/pv26_commit_sweep
  --worktree-root  /tmp/pv26_commit_sweep_worktrees
  --python         <repo>/.venv/bin/python if present, else python3/python
  --epochs         5
  --keep-going     on
  --skip-existing  on

Examples:
  bash tools/train/run_commit_train_sweep.sh \
    --repo /home/user1/Python_Workspace/YOLOPv26 \
    9e5ab34 d3bdd19 5948e0d 6884550

  bash tools/train/run_commit_train_sweep.sh \
    --repo /home/user1/Python_Workspace/YOLOPv26 \
    --out-root /tmp/pv26_sweep \
    --stop-on-failure \
    9e5ab34 d3bdd19 -- --device cuda:0 --batch-size 16
EOF
}

die() {
  echo "[runner] error: $*" >&2
  exit 1
}

find_python() {
  local repo_root="$1"
  if [[ -x "$repo_root/.venv/bin/python" ]]; then
    echo "$repo_root/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

read_commits_file() {
  local file_path="$1"
  [[ -f "$file_path" ]] || die "commits file not found: $file_path"
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(printf '%s' "$line" | xargs)"
    [[ -n "$line" ]] || continue
    COMMITS+=("$line")
  done <"$file_path"
}

REPO_ROOT=""
DATASET_ROOT=""
OUT_ROOT=""
WORKTREE_ROOT="/tmp/pv26_commit_sweep_worktrees"
PYTHON_BIN=""
EPOCHS="5"
KEEP_GOING="1"
SKIP_EXISTING="1"
KEEP_WORKTREES_ON_SUCCESS="0"
KEEP_WORKTREES_ON_FAILURE="1"
COMMITS_FILE=""
declare -a COMMITS=()
declare -a EXTRA_TRAIN_ARGS=()

while (($# > 0)); do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || die "--repo requires a value"
      REPO_ROOT="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --dataset-root)
      [[ $# -ge 2 ]] || die "--dataset-root requires a value"
      DATASET_ROOT="$2"
      shift 2
      ;;
    --worktree-root)
      [[ $# -ge 2 ]] || die "--worktree-root requires a value"
      WORKTREE_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --epochs)
      [[ $# -ge 2 ]] || die "--epochs requires a value"
      EPOCHS="$2"
      shift 2
      ;;
    --commits-file)
      [[ $# -ge 2 ]] || die "--commits-file requires a value"
      COMMITS_FILE="$2"
      shift 2
      ;;
    --stop-on-failure)
      KEEP_GOING="0"
      shift
      ;;
    --keep-going)
      KEEP_GOING="1"
      shift
      ;;
    --keep-worktrees)
      KEEP_WORKTREES_ON_SUCCESS="1"
      KEEP_WORKTREES_ON_FAILURE="1"
      shift
      ;;
    --remove-failed-worktrees)
      KEEP_WORKTREES_ON_FAILURE="0"
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING="1"
      shift
      ;;
    --rerun-existing)
      SKIP_EXISTING="0"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_TRAIN_ARGS=("$@")
      break
      ;;
    *)
      COMMITS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(pwd)"
fi
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
[[ -d "$REPO_ROOT/.git" || -f "$REPO_ROOT/.git" ]] || git -C "$REPO_ROOT" rev-parse --git-dir >/dev/null 2>&1 || die "not a git repo: $REPO_ROOT"

if [[ -n "$COMMITS_FILE" ]]; then
  read_commits_file "$COMMITS_FILE"
fi
[[ ${#COMMITS[@]} -gt 0 ]] || die "no commits provided"

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$REPO_ROOT/runs/pv26_commit_sweep"
fi
mkdir -p "$OUT_ROOT" "$WORKTREE_ROOT"

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(find_python "$REPO_ROOT")" || die "python interpreter not found"
fi
[[ -x "$PYTHON_BIN" ]] || die "python not executable: $PYTHON_BIN"

if [[ -z "$DATASET_ROOT" ]]; then
  DATASET_ROOT="$REPO_ROOT/datasets/pv26_v1_bdd_full"
fi
[[ -d "$DATASET_ROOT" ]] || die "dataset root not found: $DATASET_ROOT"
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"

for arg in "${EXTRA_TRAIN_ARGS[@]}"; do
  case "$arg" in
    --out-dir|--run-name)
      die "do not override $arg via EXTRA_TRAIN_ARGS; this runner manages run paths"
      ;;
  esac
done

SUMMARY_PATH="$OUT_ROOT/summary_$(date +%Y%m%d_%H%M%S).tsv"
printf 'index\tcommit_input\tcommit_resolved\tcommit_short\tsubject\tstatus\texit_code\tstarted_at\tfinished_at\trun_dir\tlog_path\tlatest_pt\tbest_pt\tworktree\n' >"$SUMMARY_PATH"

echo "[runner] repo=$REPO_ROOT"
echo "[runner] out_root=$OUT_ROOT"
echo "[runner] worktree_root=$WORKTREE_ROOT"
echo "[runner] python=$PYTHON_BIN"
echo "[runner] dataset_root=$DATASET_ROOT"
echo "[runner] epochs=$EPOCHS"
echo "[runner] commits=${#COMMITS[@]}"
echo "[runner] summary=$SUMMARY_PATH"

for idx in "${!COMMITS[@]}"; do
  commit_input="${COMMITS[$idx]}"
  started_at="$(date -Iseconds)"
  finished_at=""
  status="not_started"
  exit_code="NA"
  resolved_commit=""
  short_commit=""
  subject=""
  worktree_path=""
  run_dir=""
  log_path=""
  latest_pt=""
  best_pt=""

  if ! resolved_commit="$(git -C "$REPO_ROOT" rev-parse --verify "${commit_input}^{commit}" 2>/dev/null)"; then
    finished_at="$(date -Iseconds)"
    status="bad_commit"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$((idx + 1))" "$commit_input" "" "" "" "$status" "$exit_code" "$started_at" "$finished_at" "" "" "" "" "" >>"$SUMMARY_PATH"
    echo "[runner] commit not found: $commit_input"
    [[ "$KEEP_GOING" == "1" ]] && continue
    exit 1
  fi

  short_commit="$(git -C "$REPO_ROOT" rev-parse --short "$resolved_commit")"
  subject="$(git -C "$REPO_ROOT" show -s --format=%s "$resolved_commit" | tr '\t' ' ')"
  run_dir="$OUT_ROOT/$((idx + 1))_${short_commit}"
  log_path="$run_dir/train.log"
  latest_pt="$run_dir/checkpoints/latest.pt"
  best_pt="$run_dir/checkpoints/best.pt"

  echo
  echo "[runner] ===== [$((idx + 1))/${#COMMITS[@]}] $short_commit $subject ====="

  if [[ "$SKIP_EXISTING" == "1" && -f "$latest_pt" && -f "$best_pt" ]]; then
    finished_at="$(date -Iseconds)"
    status="skipped_existing"
    exit_code="0"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$((idx + 1))" "$commit_input" "$resolved_commit" "$short_commit" "$subject" \
      "$status" "$exit_code" "$started_at" "$finished_at" "$run_dir" "$log_path" "$latest_pt" "$best_pt" "" >>"$SUMMARY_PATH"
    echo "[runner] skipping existing run: $run_dir"
    continue
  fi

  mkdir -p "$run_dir"
  printf 'commit_input=%s\ncommit_resolved=%s\ncommit_short=%s\nsubject=%s\nstarted_at=%s\n' \
    "$commit_input" "$resolved_commit" "$short_commit" "$subject" "$started_at" >"$run_dir/meta.txt"

  worktree_path="$(mktemp -d "$WORKTREE_ROOT/${short_commit}_XXXXXX")"
  rmdir "$worktree_path"
  git -C "$REPO_ROOT" worktree add --force --detach "$worktree_path" "$resolved_commit" >/dev/null

  declare -a train_cmd=(
    "$PYTHON_BIN"
    "tools/train/train_pv26.py"
    "--dataset-root" "$DATASET_ROOT"
    "--out-dir" "$OUT_ROOT"
    "--run-name" "$((idx + 1))_${short_commit}"
    "--epochs" "$EPOCHS"
    "--no-progress"
  )
  if [[ ${#EXTRA_TRAIN_ARGS[@]} -gt 0 ]]; then
    train_cmd+=("${EXTRA_TRAIN_ARGS[@]}")
  fi

  set +e
  (
    cd "$worktree_path"
    echo "[runner] commit=$short_commit"
    echo "[runner] resolved_commit=$resolved_commit"
    echo "[runner] worktree=$worktree_path"
    printf '[runner] cmd='
    printf '%q ' "${train_cmd[@]}"
    printf '\n'
    "${train_cmd[@]}"
  ) 2>&1 | tee "$log_path"
  exit_code="${PIPESTATUS[0]}"
  set -e

  finished_at="$(date -Iseconds)"
  status="failed"

  if [[ "$exit_code" == "0" && -f "$latest_pt" && -f "$best_pt" ]] && grep -Fq "[pv26] finished." "$log_path"; then
    status="ok"
  fi

  printf 'finished_at=%s\nstatus=%s\nexit_code=%s\nlog_path=%s\nlatest_pt=%s\nbest_pt=%s\nworktree=%s\n' \
    "$finished_at" "$status" "$exit_code" "$log_path" "$latest_pt" "$best_pt" "$worktree_path" >>"$run_dir/meta.txt"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$((idx + 1))" "$commit_input" "$resolved_commit" "$short_commit" "$subject" \
    "$status" "$exit_code" "$started_at" "$finished_at" "$run_dir" "$log_path" "$latest_pt" "$best_pt" "$worktree_path" >>"$SUMMARY_PATH"

  if [[ "$status" == "ok" ]]; then
    echo "[runner] success: $short_commit"
    if [[ "$KEEP_WORKTREES_ON_SUCCESS" != "1" ]]; then
      git -C "$REPO_ROOT" worktree remove --force "$worktree_path" >/dev/null 2>&1 || true
    fi
  else
    echo "[runner] failed: $short_commit exit_code=$exit_code"
    if [[ "$KEEP_WORKTREES_ON_FAILURE" != "1" ]]; then
      git -C "$REPO_ROOT" worktree remove --force "$worktree_path" >/dev/null 2>&1 || true
    fi
    if [[ "$KEEP_GOING" != "1" ]]; then
      echo "[runner] stopping on failure"
      break
    fi
  fi
done

git -C "$REPO_ROOT" worktree prune >/dev/null 2>&1 || true
echo
echo "[runner] done"
echo "[runner] summary=$SUMMARY_PATH"
