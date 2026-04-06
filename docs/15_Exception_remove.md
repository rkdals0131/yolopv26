# PV26 코드베이스 방어코드/폴백 감사 보고서

## 조사 범위와 방법

- 대상 리포지토리: `yolopv26-develop`
- 전수 스캔 대상: Python 소스 전체 169개 파일
- 이 중 운영 코드 중심 분석: 120개 파일, 약 29,875 LOC
- 방법:
  - 전체 파일에 대해 `try/except`, `fallback`, `return None`, `return []`, `return {}` 패턴 전수 스캔
  - broad exception(`except Exception`, naked `except`) AST 추출
  - 핫스팟 파일은 line-level 수동 확인
- 제외 기준:
  - 입력 shape/type 계약 검증, 명시적 `ValueError`/`TypeError`는 **좋은 방어코드**로 보고 제외
  - 트랜잭션 롤백처럼 **상태 일관성을 지키는 복구 코드**는 유지 권장

---

## 한 줄 총평

이 리포지토리는 전체적으로 문서/계약 지향 성향이 강하고, 코어 수학/인코딩 쪽은 생각보다 깔끔합니다. 다만 **학습 런타임**, **resume/seed checkpoint 탐색**, **bootstrap 파이프라인**, **환경/미리보기 도구**에 "일단 굴러가게" 성격의 예외 처리와 fallback이 꽤 몰려 있습니다.

특히 문제인 건 아래 4종입니다.

1. **학습 배치를 조용히 skip** 하는 코드
2. **원래 쓰려던 checkpoint/config가 실패하면 다른 걸 집어드는 코드**
3. **실패 원인을 숨기고 `None`/빈 자료구조로 계속 가는 코드**
4. **환경 깨짐을 로컬 fallback으로 덮는 코드**

사용자 의도대로라면, 이 중 상당수는 **strict fail-fast**로 바꾸는 게 맞습니다.

---

## 우선순위별 정리

## P0 — 바로 줄이거나 제거할 것

### 1) 학습 step에서 loss/assigner/OOM을 먹고 배치를 skip함

**파일**
- `model/engine/_trainer_step.py:187-194`
- `model/engine/_trainer_step.py:221-248`
- `model/engine/trainer.py:293-295, 322-323`

**현재 동작**
- total loss가 non-finite면 기본값으로는 예외를 안 터뜨리고 step을 skip합니다.
- `PV26DetAssignmentUnavailable`가 나와도 배치를 skip합니다.
- OOM도 기본값으로는 복구 가능한 것으로 보고 skip합니다.
- 기본 생성자 설정 자체가 `skip_non_finite_loss=True`, `oom_guard=True` 입니다.

**왜 나쁜가**
- 학습이 실제로는 깨졌는데도 run은 살아남습니다.
- history에는 `skipped_reason`만 남고, 근본 원인(traceback)은 사라집니다.
- 스케줄러, early stopping, phase selection이 "불안정한 run"을 정상 run으로 오판할 수 있습니다.
- 배치 일부만 계속 skip되면 데이터 분포 편향이 생기고, 이건 나중에 재현도 잘 안 됩니다.

**권장 조치**
- 기본값을 바꾸세요.
  - `skip_non_finite_loss=False`
  - `oom_guard=False`
- `PV26DetAssignmentUnavailable` catch 제거 권장
- 정말 남겨야 한다면 **명시적 opt-in 플래그**로만 허용:
  - 예: `--best-effort-train`
  - 기본은 strict fail-fast

**추천 형태**
- non-finite loss → 즉시 `FloatingPointError`
- assigner 실패 → 원인 그대로 예외 전파
- OOM → 즉시 `torch.cuda.OutOfMemoryError` 전파

이 파일이 전체 코드베이스에서 가장 먼저 손댈 곳입니다.

---

### 2) assigner 오류를 별도 예외로 감싸고, 상위에서 skip되게 연결됨

**파일**
- `model/engine/loss.py:45`
- `model/engine/loss.py:325-353`
- `model/engine/loss.py:546-558`

**현재 동작**
- assigner가 없으면 `PV26DetAssignmentUnavailable("task_aligned_assigner_missing")`
- assigner 내부에서 어떤 예외가 나도 `PV26DetAssignmentUnavailable("task_aligned_assigner_failed: ...")` 로 감쌉니다.
- 그리고 이 예외는 위 `_trainer_step.py`에서 잡혀서 step skip으로 이어집니다.

**왜 나쁜가**
- root cause가 런타임 로그에서 한 단계 추상화됩니다.
- shape mismatch, device mismatch, upstream contract 깨짐, AMP 문제 같은 서로 다른 원인이 전부 `assignment_unavailable`로 묶입니다.
- 결국 "학습은 돌았다"처럼 보이는데 detection supervision은 사실상 망가질 수 있습니다.

**권장 조치**
- `_build_det_assignment()` 안에서는 broad catch를 없애고 원인 예외를 그대로 터뜨리세요.
- 정말 메시지를 보강하고 싶으면 `raise RuntimeError(... ) from exc` 정도는 괜찮지만, 상위에서 skip하지는 말아야 합니다.
- assigner 부재는 환경 문제이므로 시작 시점에 fail-fast 하는 편이 낫습니다.

---

### 3) NMS가 깨져도 pure-python NMS로 조용히 대체됨

**파일**
- `model/engine/postprocess.py:244-266`

**현재 동작**
- `torchvision.ops.batched_nms` 호출이 어떤 이유로든 실패하면 broad catch 후 custom `_nms()`로 대체합니다.

**왜 나쁜가**
- `torchvision` ABI mismatch, CUDA/CPU extension mismatch, packaging 문제를 조용히 가립니다.
- 같은 모델인데 환경 따라 NMS 구현이 달라져 결과 비교가 흔들릴 수 있습니다.
- 특히 export/benchmark/debug 시 "왜 결과가 다른지"를 찾기 어려워집니다.

**권장 조치**
- 기본은 fail-fast로 바꾸세요.
- pure-python NMS는 테스트/디버그 전용 명시 플래그에서만 허용하는 편이 맞습니다.
  - 예: `allow_python_nms_fallback=False` 기본

---

### 4) stage 4 lane-only 학습이 실패하면 조용히 all-heads 학습으로 바뀜

**파일**
- `model/engine/trainer.py:168-194`

**현재 동작**
- `lane_family_modules()` 또는 lane 관련 모듈 탐지가 실패하면
  - `head_policy = "all_heads_fallback"`
  - 전체 head를 다시 학습 가능하게 열어버립니다.

**왜 나쁜가**
- 이건 예외처리보다 더 위험한 **의미론 변경 fallback** 입니다.
- 사용자는 lane family fine-tune을 의도했는데, 실제론 all-head fine-tune이 수행됩니다.
- 결과가 좋아도 나빠도 실험 계약이 깨집니다.

**권장 조치**
- lane-family 모듈을 찾지 못하면 즉시 `RuntimeError` 를 내세요.
- 실험 의도를 바꾸는 fallback은 없애는 게 맞습니다.

---

### 5) resume checkpoint가 아니면 다른 checkpoint를 대신 찾아서 재개함

**파일**
- `tools/od_bootstrap/teacher/runtime/resume.py:27-34`
- `tools/od_bootstrap/teacher/runtime/resume.py:79-96`
- `tools/od_bootstrap/teacher/runtime/resume.py:109-127`
- `tools/od_bootstrap/teacher/runtime/trainer.py:149-204`

**현재 동작**
- `torch.load()` 실패 시 `_load_checkpoint_payload()` 는 `None` 반환
- 지정 checkpoint가 resumable하지 않으면 teacher root 전체에서 "최신 resumable checkpoint"를 찾아 fallback
- sibling checkpoint까지 뒤집니다.
- `_restore_resume_training_args()` 는 내부의 모든 예외를 잡아 generic `FileNotFoundError` 로 바꿉니다.

**왜 나쁜가**
- 사용자가 `A.pt`로 resume했다고 생각하는데 실제로는 `B.pt`에서 시작할 수 있습니다.
- 실패 원인이 checkpoint 손상인지, train_args 누락인지, finalized checkpoint인지 다 가려집니다.
- 재현성, 실험 추적, lineage가 흐려집니다.

**권장 조치**
- `resolve_resume_checkpoint_path()`의 자동 fallback 제거
- 의미를 분리하세요.
  - `--resume path/to/exact.pt` → 정확히 그 파일만 허용, 아니면 실패
  - `--resume latest` → 그때만 최신 resumable checkpoint 탐색
- `_restore_resume_training_args()`에서 broad catch 후 generic 메시지로 덮지 말고 원인을 그대로 올리세요.

---

### 6) calibration class policy가 없으면 default policy로 조용히 진행됨

**파일**
- `tools/od_bootstrap/presets.py:159-174`
- `config/od_bootstrap_hyperparameters.yaml:149-151`

**현재 동작**
- `class_policy.yaml`이 없으면 hard-coded/default policy를 그대로 사용합니다.
- calibration에 일부 클래스가 빠져도 defaults와 merge해서 계속 갑니다.
- config 주석에도 fallback이라고 적혀 있습니다.

**왜 나쁜가**
- bootstrap 결과 품질이 calibration 상태에 따라 달라지는데, 파이프라인은 그냥 통과합니다.
- 팀은 calibration을 돌렸는지 안 돌렸는지 체감하기 어렵고, 데이터셋 생성 semantics가 흐려집니다.

**권장 조치**
- 기본은 fail-fast:
  - calibration/class_policy.yaml 없으면 종료
- 정말 초기 bring-up 용이면 명시 플래그로만 허용:
  - `--allow-default-class-policy`
- 일부 클래스 누락도 기본은 에러 권장

---

## P1 — strict 모드로 몰아넣거나 기본 동작에서 빼는 게 좋은 것

### 7) 이미지 크기 probe 실패 시 label 쪽 parsed size로 계속 감

**파일**
- `tools/od_bootstrap/source/raw_common.py:84-98`
- `tools/od_bootstrap/source/raw_common.py:143-164`

**현재 동작**
- PIL로 이미지 크기 읽기 실패 시 `identify`로 대체
- 그것도 실패하면 label JSON 안의 parsed size가 있으면 그걸 믿고 진행

**왜 애매한가**
- raw source 정리 단계에서는 편의상 이해는 되지만,
- 실제 이미지와 메타가 불일치할 때 잘못된 bbox/geometry를 조용히 먹을 수 있습니다.

**권장 조치**
- 운영 기본값은 strict:
  - 이미지 실제 크기 probe 실패 시 중단
- bring-up 단계에서만 명시적으로 허용:
  - `--allow-metadata-image-size`
- 최소한 로그에 sample 단위 치명 오류로 남기고 summary에서 fail 처리하는 게 낫습니다.

---

### 8) label parse 실패 / worker 실패 / chunk 실패를 누적만 하고 계속 진행

**파일**
- `tools/od_bootstrap/source/raw_common.py:224-238`
- `tools/od_bootstrap/source/aihub/workers.py:39-59`
- `tools/od_bootstrap/source/aihub/pipeline.py:300-325`
- `tools/od_bootstrap/source/bdd100k.py:392-411`
- `tools/od_bootstrap/source/bdd100k.py:844-860`

**현재 동작**
- 샘플 단위 실패를 `failure` row로 누적하고 전체 파이프라인은 계속 진행합니다.

**왜 나쁜가**
- 대량 데이터 파이프라인에서 흔한 패턴이긴 한데, strict 운영 철학과는 반대입니다.
- 실패율이 낮으면 묻히고, 데이터 품질 문제가 누적될 수 있습니다.

**권장 조치**
- 기본값을 strict로 두세요.
  - 샘플 실패 1건이라도 있으면 종료
- 정말 대규모 정리 작업에서만 opt-in:
  - `--best-effort-source-standardize`
  - `--max-failures N`
- summary에 실패 건수만 적는 방식은 기본 동작으로 두지 않는 게 좋습니다.

---

### 9) malformed cache / existing scene JSON 로드 실패 시 cache miss 취급

**파일**
- `tools/od_bootstrap/source/shared/resume.py:29-57`

**현재 동작**
- 기존 scene JSON이 깨졌거나 load 실패하면 그냥 `None` 반환
- 즉 cache miss처럼 취급됩니다.

**왜 나쁜가**
- resume/cache 일관성 깨짐이 조용히 숨겨집니다.
- 손상된 산출물이 있는지, 단순 미존재인지 구분이 안 됩니다.

**권장 조치**
- 파일이 존재하는데 load 실패하면 즉시 에러
- 진짜 cache miss와 corrupted cache를 분리

---

### 10) meta-train resume/retrain seed checkpoint fallback

**파일**
- `tools/pv26_train/scenarios.py:425-439`
- `tools/pv26_train/scenarios.py:447-474`

**현재 동작**
- manifest의 `best_checkpoint_path`가 없거나 깨지면 `phase_{i}/checkpoints/best.pt` fallback
- 시작 phase에 seed가 없으면 이전 phase best까지 거슬러 올라갑니다.

**왜 나쁜가**
- 이건 training lineage를 바꾸는 fallback입니다.
- "현재 phase의 정확한 best checkpoint"가 아니어도 그냥 다른 걸 줍습니다.

**권장 조치**
- exact resume/retrain에서는 manifest path만 신뢰
- phase chain fallback은 별도 명시 명령으로 분리
  - 예: `--derive-seed allow_previous_phase_best`

---

### 11) TensorBoard 사용 불가를 조용히 비활성화

**파일**
- `common/train_runtime.py:106-136`
- `model/engine/_trainer_step.py:295-314`

**현재 동작**
- SummaryWriter import/init 실패 시 writer를 `None`으로 두고 상태 dict만 남깁니다.
- graph add 실패도 `except Exception: pass`

**왜 애매한가**
- 선택적 부가기능이라 완전한 문제는 아니지만,
- 실험 인프라를 기대하는 운영 환경에선 조용한 비활성화가 좋지 않습니다.

**권장 조치**
- 운영 기본은 `tensorboard_required=True`로 두고 실패 시 종료
- 아니면 최소한 graph add 실패는 warning 이상으로 남기고 pass하지 않도록 조정

---

### 12) `read_jsonl()` missing file → 빈 리스트, `read_yaml()` empty root → 빈 dict

**파일**
- `common/io.py:24-28`
- `common/io.py:31-37`

**현재 동작**
- JSONL 파일이 없으면 `[]`
- YAML root가 `null`이면 `{}`

**왜 나쁜가**
- "정말 데이터가 없는 것"과 "파일이 없거나 비어있는 설정"이 구분되지 않습니다.
- `controller.replay(...)` 같은 흐름에서 history 파일 누락이 그냥 신규 상태처럼 보일 수 있습니다.

**권장 조치**
- 기본 유틸은 fail-fast로 두고,
- 호출부에서 "없어도 되는 파일"만 명시적으로 처리하도록 분리하세요.
  - 예: `read_optional_jsonl()` 별도 함수
  - `read_required_yaml()` / `read_optional_yaml()` 분리

---

### 13) overlay 렌더링이 ImageMagick 실패 시 Pillow fallback

**파일**
- `common/overlay.py:7-12`
- `common/overlay.py:247-250`
- `tools/pv26_train/cli.py:660-664`

**현재 동작**
- ImageMagick 실패 시 Pillow로 fallback
- preview overlay 실패는 `overlay_error`만 저장하고 전체 흐름은 계속 진행

**왜 애매한가**
- preview/debug artifact라서 치명도는 낮습니다.
- 다만 운영자가 overlay 결과를 신뢰할 때 renderer가 바뀌는 건 헷갈립니다.

**권장 조치**
- 미리보기는 지금처럼 optional로 둘 수 있음
- 대신 "fallback renderer 사용됨"을 artifact에 명시하거나, strict preview 모드에서는 즉시 실패시키는 게 낫습니다.

---

### 14) export 스크립트가 current layout 실패 시 legacy layout import 재시도

**파일**
- `tools/model_export/pv26_torchscript.py:112-133`

**현재 동작**
- 현재 모듈 layout import 실패 → legacy layout import 재시도

**왜 나쁜가**
- 코드 구조가 이미 정리된 저장소라면 오래된 호환 경로가 오히려 혼선을 줍니다.
- export는 특히 환경/버전 계약이 엄격해야 하는 편입니다.

**권장 조치**
- legacy import fallback 제거 권장
- 필요하면 별도 `legacy_export` 스크립트로 분리

---

## P2 — 남겨도 되지만, strict 철학이라면 정리할 수 있는 것

### 14-1) pv26 train CLI가 trunk API 호환용 import fallback을 유지 중

**파일**
- `tools/pv26_train/cli.py:42-54`
- `tools/pv26_train/cli.py:98-124`

**현재 동작**
- `build_yolo26_trunk`, `infer_pyramid_channels`, `resolve_yolo26_weights` import 실패 시 `None`으로 두고
- 이후 legacy/old API 가정을 써서 계속 진행합니다.

**왜 애매한가**
- trunk API가 이미 정리된 저장소라면, 이 fallback은 오래된 layout을 계속 끌고 가는 효과가 있습니다.
- backbone/head channel 계산 경로가 환경에 따라 달라질 수 있습니다.

**권장 조치**
- trunk API를 단일 경로로 고정하고 compatibility import 제거
- 정말 과거 브랜치 지원이 필요하면 별도 wrapper 모듈로 격리

---

### 15) 환경 점검/런처가 broad catch로 상태만 보여주고 계속 감

**파일**
- `tools/check_env/scan.py:125-180`
- `tools/check_env/scan.py:823-927`
- `tools/check_env/launch.py:45-56`
- `tools/check_env/launch.py:368-373`

**현재 동작**
- import/version/runtime check 실패를 에러 문자열로만 수집
- malformed manifest/summary는 candidate 없음으로 취급
- phase stress 기본 batch size 계산 실패 시 그냥 `40`
- config panel 렌더링 실패 시 실행은 계속

**평가**
- 이건 운영자용 TUI/스캐너라 best-effort가 어느 정도 이해됩니다.
- 다만 리포 전체 철학을 strict로 맞출 거면, 최소한 아래 둘은 정리하는 게 좋습니다.
  - `_default_phase_stress_batch_size()` 의 magic fallback `40`
  - malformed run metadata를 조용히 후보 제외하는 부분

**권장 조치**
- UI 레이어만 best-effort 허용
- core meaning을 바꾸는 fallback은 제거

---

### 16) `sync_timing_device()` 가 특정 device synchronize 실패 시 no-arg synchronize로 재시도

**파일**
- `common/train_runtime.py:15-24`

**현재 동작**
- `torch.cuda.synchronize(device)` 실패 → `torch.cuda.synchronize()` 재시도

**평가**
- 기능적 피해는 상대적으로 작지만, 실제 device 문제를 숨길 수 있습니다.

**권장 조치**
- strict 모드에서는 그냥 예외 전파
- 최소한 exception type을 제한하거나 warning 남기기

---

### 17) `checkpoint_audit` 의 안전 fallback들

**파일**
- `tools/od_bootstrap/build/checkpoint_audit.py:56-63`
- `tools/od_bootstrap/build/checkpoint_audit.py:112-120`

**현재 동작**
- parameter count 실패 → `None`
- `samefile()` 실패 → inode/device 비교 fallback

**평가**
- audit 도구라서 치명도는 낮습니다.
- 특히 `samefile()` → inode 비교는 괜찮은 편입니다.
- `param_count=None`도 audit report 맥락에서는 큰 문제는 아닙니다.

**권장 조치**
- 굳이 정리한다면 `param_count` 실패만 warning으로 드러내는 정도

---

## 유지 권장 — 이건 나쁜 방어코드로 보지 않음

### 1) 계약 검증

- shape/type/value 검증 후 `ValueError`, `TypeError` 터뜨리는 코드들
- 예: `model/engine/loss.py` 내부 supervision contract 검증

이건 사용자가 유지해도 된다고 한 "확인형 방어코드"에 정확히 해당합니다.

### 2) 트랜잭션 롤백

**파일**
- `tools/od_bootstrap/build/final_dataset.py:218-230`

staging publish 중 실패하면 backup을 복원합니다. 이건 "문제를 숨기는 fallback"이 아니라 **파일 시스템 상태 일관성을 지키는 롤백** 입니다. 유지하는 게 맞습니다.

### 3) 사용자 인터럽트/비대화형 UI 보조

- `KeyboardInterrupt` 처리
- TUI 입력 파싱
- progress backend 유무에 따른 표시 방식 차이

이건 런타임 semantics를 바꾸지 않기 때문에 우선순위가 낮습니다.

---

## UB/스노우볼 위험 관점에서 특히 문제인 지점

### 1) skip 기반 복구는 UB보다는 "침묵한 상태 오염"에 가깝다

엄밀한 C/C++식 UB는 아니어도, 현재 맥락에서는 더 위험할 수 있습니다.

- train loop는 계속 돈다
- optimizer/scheduler/history는 진행된다
- 일부 supervision만 빠진 상태가 누적된다
- 나중에 결과가 이상해져도 첫 원인 지점을 찾기 어렵다

즉, 사용자 표현대로 **스노우볼**이 굴러가기 좋은 형태가 맞습니다.

### 2) resume fallback은 실험 재현성을 직접 깨뜨린다

"같은 run을 이어서 학습했다"는 가정을 깨뜨리므로, 실험 관리 측면에서 가장 치명적인 부류입니다.

### 3) class policy fallback은 데이터 생성 semantics 자체를 바꾼다

이건 단순 예외처리 문제가 아니라 **데이터 계약 변경** 입니다.

### 4) NMS fallback은 환경 깨짐을 숨긴다

성능/결과가 미묘하게 달라질 수 있어서 디버깅을 더 어렵게 만듭니다.

---

## 가장 먼저 바꿀 10개

1. `model/engine/_trainer_step.py` 의 non-finite/assigner/OOM skip 제거
2. `model/engine/trainer.py` 기본값 `skip_non_finite_loss=True`, `oom_guard=True` 뒤집기
3. `model/engine/loss.py` assigner broad catch 제거
4. `model/engine/postprocess.py` batched_nms fallback 제거
5. `model/engine/trainer.py` 의 `all_heads_fallback` 제거
6. `tools/od_bootstrap/teacher/runtime/resume.py` 자동 checkpoint 대체 제거
7. `tools/od_bootstrap/teacher/runtime/trainer.py` broad catch → generic FileNotFound masking 제거
8. `tools/od_bootstrap/presets.py` default class policy fallback 제거 또는 opt-in화
9. `tools/pv26_train/scenarios.py` phase best / previous phase seed fallback 축소
10. `common/io.py` 의 `read_jsonl`/`read_yaml` optional semantics 분리

---

## 추천 리팩터링 원칙

### 원칙 A — core runtime은 무조건 strict

대상:
- training loop
- loss/assigner
- resume/checkpoint resolution
- dataset contract enforcement
- export contract

여기서는 fallback 금지, root cause 그대로 터뜨리기.

### 원칙 B — tooling/UI만 opt-in best-effort 허용

대상:
- check_env
- preview overlay
- progress backend
- audit/report 생성기

단, 이 경우도 기본 동작을 바꾸지 말고 **표시/도구 기능만 degrade** 해야 합니다.

### 원칙 C — optional과 required I/O를 함수 레벨에서 분리

예:
- `read_required_jsonl()`
- `read_optional_jsonl()`
- `resolve_exact_resume_checkpoint()`
- `resolve_latest_resumable_checkpoint()`

지금처럼 한 함수 안에서 알아서 fallback하지 말고, 호출자가 의도를 선택하게 해야 합니다.

---

## 결론

이 코드베이스에서 가장 불편한 "AI스러운 fallback"은 실제로 **코어 training/resume/class-policy** 쪽에 있습니다. 반대로 단순 UI 도구 쪽 best-effort는 상대적으로 덜 위험합니다.

사용자 취향대로 정리한다면,

- **학습 step skip 계열 전부 제거**
- **resume/checkpoint 자동 대체 제거**
- **class policy default fallback 제거**
- **NMS/environment fallback 제거**
- **optional artifact/preview만 제한적으로 best-effort 유지**

이 방향이 가장 깔끔합니다.

문서 철학에도 이미 적혀 있듯, 이 리포는 원래 "문서와 다른 임시 구현을 조용히 넣지 않는다", "일단 돌아가게를 이유로 contract를 흐리지 않는다" 쪽에 더 가깝습니다. 그래서 위 정리는 오히려 기존 철학에 잘 맞습니다.
