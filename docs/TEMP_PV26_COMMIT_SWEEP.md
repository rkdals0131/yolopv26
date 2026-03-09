# TEMP_PV26 Commit Sweep

기준 날짜: `2026-03-07`

## 목적

여러 후보 커밋을 unattended 방식으로 순차 학습하기 위한 실행 메모다.

- 현재 checkout은 그대로 둔다.
- 각 커밋은 별도 `git worktree`에서 detached 상태로 돈다.
- 각 run이 끝나면 성공 조건을 확인한 뒤 다음 커밋으로 넘어간다.
- 결과는 `summary_*.tsv` 한 장으로 모인다.

## 스크립트

- 경로: `tools/train/run_commit_train_sweep.sh`

이 스크립트는 아래를 자동 수행한다.

1. 절대 `--repo` 경로를 기준으로 commit resolve
2. commit마다 임시 worktree 생성
3. `tools/train/train_pv26.py` 실행
4. 종료 판정
5. success/failure/경로를 TSV summary에 기록
6. 다음 commit으로 진행

종료 판정 조건:

1. 학습 프로세스 exit code가 `0`
2. log에 `[pv26] finished.` 존재
3. `checkpoints/latest.pt` 존재
4. `checkpoints/best.pt` 존재

## 기본 사용법

BDD-only 기준 기본 데이터셋 루트는 아래다.

- `/home/user1/Python_Workspace/YOLOPv26/datasets/pv26_v1_bdd_full`

기본 epoch는 `5`다.

```bash
bash /home/user1/Python_Workspace/YOLOPv26/tools/train/run_commit_train_sweep.sh \
  --repo /home/user1/Python_Workspace/YOLOPv26 \
  9e5ab34 d3bdd19 5948e0d 6884550 2887148 66ae3bd
```

## commit 파일로 실행

```text
# one commit per line
9e5ab34
d3bdd19
5948e0d
6884550
2887148
66ae3bd
```

```bash
cat >/tmp/pv26_commit_candidates.txt <<'EOF'
9e5ab34
d3bdd19
5948e0d
6884550
2887148
66ae3bd
EOF

bash /home/user1/Python_Workspace/YOLOPv26/tools/train/run_commit_train_sweep.sh \
  --repo /home/user1/Python_Workspace/YOLOPv26 \
  --commits-file /tmp/pv26_commit_candidates.txt
```

## merged dataset 등 다른 루트로 돌릴 때

```bash
bash /home/user1/Python_Workspace/YOLOPv26/tools/train/run_commit_train_sweep.sh \
  --repo /home/user1/Python_Workspace/YOLOPv26 \
  --dataset-root /absolute/path/to/pv26_dataset_root \
  9e5ab34 d3bdd19
```

## 추가 학습 옵션 전달

`--` 뒤에 붙는 인자는 `train_pv26.py`로 그대로 전달된다.

예시:

```bash
bash /home/user1/Python_Workspace/YOLOPv26/tools/train/run_commit_train_sweep.sh \
  --repo /home/user1/Python_Workspace/YOLOPv26 \
  9e5ab34 d3bdd19 \
  -- --device cuda:0 --batch-size 16
```

주의:

- `--out-dir`
- `--run-name`

이 둘은 runner가 관리하므로 extra arg로 덮어쓰면 안 된다.

## 산출물

기본 출력 루트:

- `/home/user1/Python_Workspace/YOLOPv26/runs/pv26_commit_sweep`

각 commit별 산출물:

- `<out-root>/<index>_<short_commit>/train.log`
- `<out-root>/<index>_<short_commit>/meta.txt`
- `<out-root>/<index>_<short_commit>/checkpoints/latest.pt`
- `<out-root>/<index>_<short_commit>/checkpoints/best.pt`

요약:

- `<out-root>/summary_YYYYmmdd_HHMMSS.tsv`

`summary_*.tsv` 컬럼:

- `index`
- `commit_input`
- `commit_resolved`
- `commit_short`
- `subject`
- `status`
- `exit_code`
- `started_at`
- `finished_at`
- `run_dir`
- `log_path`
- `latest_pt`
- `best_pt`
- `worktree`

## 실패 정책

기본값은 실패해도 다음 commit으로 계속 간다.

- 기본: `--keep-going`
- 실패 즉시 중단: `--stop-on-failure`

## worktree 정책

기본값:

- 성공한 worktree는 지운다.
- 실패한 worktree는 남긴다.

옵션:

- `--keep-worktrees`
- `--remove-failed-worktrees`

## 현재 회귀 확인용 후보

현재 good 기준점은 `b322371`이다.

우선 순서:

1. `9e5ab34`
2. `d3bdd19`
3. `5948e0d`
4. `6884550`
5. `2887148`
6. `66ae3bd`

## 실제 스윕 결과

기준: BDD-only, 기본 epoch `5`, commit별 train 후 `render_weights_example.py` 확인

- `2887148`: 정상
- `66ae3bd`: 정상
- `5948e0d`: 정상
- `d3bdd19`: 학습 회귀로 보이지 않음. 다만 렌더 시 checkpoint key layout mismatch 발생
- `6884550`: 비정상. 세그멘테이션/레인 subclass 결과 붕괴 재현

정리:

1. 기능 회귀는 `5948e0d -> 6884550` 사이에서 들어왔다.
2. `6884550` diff는 학습 기본값 변경이 핵심이다.
3. 그중 가장 강한 후보는 `seg_output_stride` 기본값 `1 -> 2` 변경이다.
4. `d3bdd19`는 detection adapter shell 이전 세대 checkpoint라서 `det_model.*` prefix 호환이 별도로 필요했다.
