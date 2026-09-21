#!/usr/bin/env bash
set -u

REPO_ROOT="/home/kai/yolopv26"
LOG="$REPO_ROOT/runs/aihub_lane_download.log"

mkdir -p "$REPO_ROOT/runs"

while true; do
    printf '\n[%s] AIHub 차선 데이터 준비를 시도합니다.\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
    if "$REPO_ROOT/tools/download_lane_aihub_250k.sh" >> "$LOG" 2>&1; then
        printf '[%s] 다운로드와 압축 해제가 완료되었습니다.\n' "$(date --iso-8601=seconds)" | tee -a "$LOG"
        exit 0
    fi
    printf '[%s] 준비에 실패했습니다. 자세한 원인은 %s를 확인하세요. 10분 후 재시도합니다.\n' \
        "$(date --iso-8601=seconds)" "$LOG" | tee -a "$LOG"
    sleep 600
done
