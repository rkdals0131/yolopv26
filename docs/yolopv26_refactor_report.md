# YOLOPV26 대대적 리팩토링 초안 보고서

## 0. 한 줄 결론

권장안은 **“철학 일치형 리팩토링(옵션 B)”** 이다.

핵심은 아래 5가지다.

1. `model/`은 **최종 PV26 학습/추론 런타임만** 남기고, 하위 디렉토리는 **3개(`data`, `net`, `engine`)** 로 줄인다.
2. raw AIHUB/BDD100K 표준화와 teacher 파이프라인은 **전부 `tools/od_bootstrap/` 소유** 로 옮긴다.
3. 체크인된 `.yaml` 설정 파일은 **대부분 삭제** 하고, 실제로 자주 건드리는 값만 **각 실행 파일 import 블록 바로 아래 preset 블록** 으로 올린다.
4. `smoke`, `dryrun` 용어와 모드는 **전부 제거** 하고, 필요 기능은 **정식 test / integration / e2e 테스트** 로 치환한다.
5. `OD_CLASSES`, `TL_BITS`, dataset key policy 같은 공용 계약은 standardizer 안이 아니라 **중립 공용 모듈** 로 뽑는다.

---

## 1. 현재 저장소 진단

### 1.1 구조 진단 요약

| 항목 | 현재 상태 |
|---|---:|
| `model/` 하위 디렉토리 수 | 9 |
| `model/` Python 파일 수 | 26 |
| `model/` `__init__.py` 수 | 10 |
| `tools/od_bootstrap/` 재귀 디렉토리 수 | 15 |
| `tools/od_bootstrap/` Python 파일 수 | 41 |
| `tools/od_bootstrap/` YAML 수 | 21 |
| 저장소 전체 YAML 수 | 24 |
| `run_*.py` 실행 스크립트 수 | 15 |
| `yaml.safe_load` 사용 지점 | 10 |
| `sys.path.insert(...)` 반복 지점 | 17 |
| `"smoke"` 문자열 등장 횟수 | 153 |
| `"dryrun"` 문자열 등장 횟수 | 5 |

### 1.2 `model/`의 실제 문제

현재 `model/`은 하위 디렉토리가 너무 잘게 쪼개져 있다. 특히 아래 디렉토리는 **실제 구현 파일 1개 + `__init__.py`** 패턴이다.

- `encoding/` → `pv26_target_encoder.py`
- `heads/` → `pv26_heads.py`
- `training/` → `pv26_trainer.py`
- `trunk/` → `ultralytics_yolo26.py`
- `viz/` → `overlay.py`

즉, **의미 단위보다 파일 단위로 디렉토리를 쪼갠 상태** 라서 탐색 비용만 늘고 있다.

### 1.3 `tools/od_bootstrap/`의 실제 문제

현재 `tools/od_bootstrap/`은 아래처럼 단계별로 또 쪼개져 있다.

- `calibration/`
- `common/`
- `config/`
- `eval/`
- `finalize/`
- `preprocess/`
- `smoke/`
- `sweep/`
- `train/`

문제는 이 구조가 “파이프라인 단계 설명”에는 좋지만, 실제 코드베이스 유지보수에는 과하게 잘게 나뉘어 있다는 점이다.  
게다가 `config/` 밑에도 또 6개 하위 폴더가 있어서 **디렉토리 두께가 과도하게 깊다.**

### 1.4 YAML 스프롤의 실제 문제

현재 YAML이 많은데, 실제로는 값 변화가 거의 없다.

#### OD bootstrap teacher train YAML
- 파일 수: 7개
- 펼친(flatten) key 수: 30개
- 이 중 **22개 key는 전 파일에서 동일**
- 실제로 달라지는 건 대체로 아래뿐이다.
  - `teacher_name`
  - `dataset.root`
  - `model.model_size`
  - `model.weights`
  - `model.class_names`
  - `train.epochs`
  - `train.batch`
  - `train.resume`

#### OD bootstrap eval YAML
- 파일 수: 4개
- 펼친 key 수: 20개
- 이 중 **16개 key는 전 파일에서 동일**
- 실제로 달라지는 건 대체로 아래뿐이다.
  - `teacher_name`
  - `dataset.root`
  - `model.checkpoint_path`
  - `model.class_names`

즉, 지금 상태는 **설정 파일을 위한 설정 파일 구조** 에 가깝고, 실제로 자주 조정하는 파라미터는 몇 줄 안 된다.

### 1.5 가장 중요한 구조적 문제: 소유권이 뒤집혀 있음

현재는 아래 의존이 존재한다.

- `model/loading/pv26_loader.py` 가 `model.preprocess.aihub_standardize` 에서 taxonomy를 가져옴
- `model/loss/spec.py` 가 `model.preprocess.aihub_standardize` 에서 taxonomy를 가져옴
- `model/encoding/pv26_target_encoder.py` 가 `model.preprocess.aihub_standardize` 에서 taxonomy를 가져옴
- `tools/od_bootstrap/sweep/materialize.py` 가 `model.preprocess.aihub_standardize` 에서 taxonomy를 가져옴
- `tools/od_bootstrap/finalize/final_dataset.py` 가 `model.preprocess.aihub_standardize` 에서 taxonomy를 가져옴

즉, **표준화 코드가 model/runtime 계약의 원천처럼 쓰이고 있다.**  
이건 방향이 반대다.

정상 구조는 아래여야 한다.

- **공용 계약** (`OD_CLASSES`, `TL_BITS`, lane taxonomy, dataset key policy)  
  ↓
- `tools/od_bootstrap/` 가 raw→canonical/exhaustive/final dataset 생성  
  ↓
- `model/` 은 최종 dataset를 읽고 학습/평가만 수행

### 1.6 `smoke/`가 이미 순수 보조 폴더가 아님

중요한 점 하나가 있다.

현재 `tools/od_bootstrap/preprocess/debug_vis.py` 는  
`tools.od_bootstrap.smoke.review.canonical_scene_to_overlay_scene` 를 import 하고 있다.

즉, `smoke/` 폴더는 이미 **삭제 가능한 임시 디렉토리** 가 아니라 **실제 생산 코드가 의존하는 폴더** 가 되어 있다.  
그래서 `smoke`를 제거하려면 단순 삭제가 아니라, **재사용 중인 함수부터 다른 곳으로 이사** 시켜야 한다.

---

## 2. 질문별 직접 결론

### 2.1 `model/` 병합해야 하나?
**예. 강하게 병합하는 게 맞다.**

현재 구조는 레이어 분리라기보다 파일 수만 늘리는 구조다.  
`model/`은 **3개 폴더** 가 가장 적절하다.

### 2.2 `tools/od_bootstrap/` 도 병합해야 하나?
**예. 역시 병합이 맞다.**

현재는 “단계 이름” 중심으로 폴더가 많고, config까지 따로 있어 깊이가 너무 깊다.  
**3개 폴더 + 1개 CLI 엔트리포인트** 정도가 가장 안정적이다.

### 2.3 YAML 줄여야 하나?
**예. 대부분 지워도 된다.**

특히 `train/`, `eval/`, `preprocess/`, `finalize/` 쪽 YAML은 거의 전부 코드 preset으로 옮길 수 있다.  
다만 **외부 툴이 요구하는 산출물 YAML**(예: ultralytics data.yaml, calibration 결과 class policy 출력)은 **생성 결과물** 로는 유지 가능하다.

### 2.4 `smoke`, `dryrun` 없애야 하나?
**예. 용어/모드/파일명 전부 정리하는 게 맞다.**

이 저장소는 이미 test가 꽤 많은 편이다.  
이제는 `smoke`가 아니라 **unit / integration / e2e / gpu test** 로 가는 게 더 일관된다.

### 2.5 OD bootstrap 바깥의 standardize 코드는 이제 obsolete인가?
**개발 철학 기준으로는 맞다.**

당신이 정의한 철학이 아래라면:

> 원본 AIHUB + BDD100K 읽기 → 전처리 → teacher 3개 학습 → exhaustive OD dataset 제작 → lane 합치기 → PV26 train

그러면 raw source 표준화는 더 이상 `model/` 소유가 아니다.  
그건 **dataset build pipeline** 의 일부이고, 따라서 **`tools/od_bootstrap/` 소유** 가 맞다.

단, 아래 둘은 남겨야 한다.

- **공용 taxonomy / dataset contract**
- **최종 dataset loader가 읽는 canonical/exhaustive/final scene schema**

즉, **standardizer는 이동**, **contract는 중립화** 가 정답이다.

### 2.6 common 함수 재사용은 어떻게 할까?
아래 3종은 바로 공용화 대상이다.

1. **schema/taxonomy**
   - `OD_CLASSES`
   - `TL_BITS`
   - `LANE_CLASSES`
   - `LANE_TYPES`
   - dataset key / supervision policy

2. **입출력 유틸**
   - `read_json`, `write_json`
   - `link_or_copy`
   - `resolve_latest_root`
   - `now_iso`

3. **CLI/경로 유틸**
   - `resolve_path`
   - 경로 기본값
   - manifest/write helper

### 2.7 결론적으로 대대적 리팩토링 해도 되나?
**예. 지금이 오히려 정리 타이밍으로 보인다.**

이유는 두 가지다.

- 개발 철학이 이미 명확하게 바뀌었다.
- 테스트 코드가 이미 꽤 있어서, 무대뽀 리팩토링이 아니라 **테스트로 감싸면서 정리** 할 수 있다.

---

## 3. 리팩토링 목표 원칙

### 3.1 구조 원칙
- `model/` 은 **학습/추론 런타임만**
- `tools/od_bootstrap/` 은 **raw→final dataset 빌드 파이프라인 전부**
- 공용 계약은 **중립 shared/common 모듈**

### 3.2 설정 원칙
- 체크인된 YAML 최소화
- 자주 바꾸는 값만 각 실행 파일 상단에 모음
- 주석으로 `# ===== User-tunable parameters =====` 구간 명시
- 실행 시에는 **실제로 사용된 preset을 JSON manifest로 출력** 해서 재현성 확보

### 3.3 테스트 원칙
- `smoke-first` → **`test-first`**
- `smoke`, `dryrun` CLI 제거
- 필요한 검증은 `pytest` 테스트로 귀속
- 작은 fixture 기반 integration test를 정식으로 둠

---

## 4. 대안 비교

## 옵션 A. 보수적 병합안

### 구조
```text
common/
  pv26_schema.py
  io.py
  paths.py

model/
  data/
  net/
  engine/
  preprocess/

tools/od_bootstrap/
  data/
  teacher/
  shared/
  config/
```

### 장점
- 기존 코드 이동량이 적다.
- diff가 비교적 작다.
- 단기적으로 덜 위험하다.

### 단점
- `model/preprocess/` 가 계속 남아서 철학이 덜 깔끔하다.
- YAML/config 스프롤이 일부 계속 남는다.
- “raw 표준화는 bootstrap 소유”라는 핵심 원칙이 반쯤만 반영된다.

### 평
- **임시 봉합용** 으로는 가능하지만, 이번 기회에 하는 대대적 정리안으로는 아쉽다.

---

## 옵션 B. 철학 일치형 리팩토링안 **(권장)**

### 구조
```text
common/
  pv26_schema.py
  io.py
  paths.py

model/
  data/
    dataset.py
    transform.py
    target_encoder.py
    preview.py
  net/
    trunk.py
    heads.py
  engine/
    loss.py
    postprocess.py
    metrics.py
    evaluator.py
    trainer.py

tools/
  run_pv26_train.py
  check_env.py
  od_bootstrap/
    data/
      aihub.py
      bdd100k.py
      source_prep.py
      teacher_dataset.py
      image_list.py
      exhaustive_od.py
      final_dataset.py
      debug_vis.py
    teacher/
      train.py
      eval.py
      calibrate.py
      ultralytics_runner.py
      policy.py
    cli.py
    presets.py

test/
  model/
  od_bootstrap/
  fixtures/
```

### 장점
- 당신이 설명한 개발 철학과 가장 정확히 맞는다.
- `model/` 이 아주 선명해진다.
- raw/source 관련 코드가 전부 bootstrap으로 모인다.
- YAML 제거가 자연스럽다.
- `smoke` 제거도 구조적으로 쉬워진다.

### 단점
- import 경로 수정 범위가 넓다.
- 테스트/문서 수정도 꽤 필요하다.
- 첫 1~2주 동안 diff가 크게 보일 수 있다.

### 평
- **이번 리팩토링의 본안** 으로 가장 적절하다.

---

## 옵션 C. 공격적 평탄화안

### 구조
```text
common/
  ...

model/
  dataset.py
  transform.py
  target_encoder.py
  trunk.py
  heads.py
  loss.py
  postprocess.py
  metrics.py
  evaluator.py
  trainer.py

tools/od_bootstrap/
  aihub.py
  bdd100k.py
  source_prep.py
  teacher_dataset.py
  image_list.py
  exhaustive_od.py
  final_dataset.py
  debug_vis.py
  train.py
  eval.py
  calibrate.py
  cli.py
  presets.py
```

### 장점
- 폴더 수는 최소다.
- 찾기 쉬워 보일 수 있다.

### 단점
- 파일이 커지고 루트가 금방 지저분해진다.
- merge conflict가 잦아진다.
- 검색/탐색은 쉬워도 의미 계층은 약해진다.

### 평
- **지금 코드량에서는 과한 평탄화** 다.  
- 개인 프로젝트면 가능하지만, 협업/증분 변경에는 비추천이다.

---

## 5. 권장안 상세 설계

## 5.1 `model/` 권장 구조 (3폴더)

```text
model/
  data/
    dataset.py        # loader + dataset key policy
    transform.py      # letterbox, box/point transform
    target_encoder.py # encode_pv26_batch
    preview.py        # overlay/preview helper
  net/
    trunk.py          # ultralytics trunk adapter
    heads.py          # PV26Heads
  engine/
    loss.py           # spec + runtime 통합 또는 loss_runtime 중심
    postprocess.py
    metrics.py
    evaluator.py
    trainer.py
```

### 이 구조가 좋은 이유
- `data`: 입력/타깃/시각화
- `net`: 순수 신경망 구조
- `engine`: 학습/평가/후처리

즉, 현재 9개 폴더를 **의미 3축** 으로 접는다.

## 5.2 `model/`에서 없앨 것

### 삭제/이동 대상
- `model/preprocess/` 전체 → `tools/od_bootstrap/data/` 로 이동
- 대부분의 subpackage re-export `__init__.py`
- `model.__getattr__` lazy import 패턴

### 왜?
지금 `model/__init__.py` 와 여러 subpackage `__init__.py` 에 `__getattr__` lazy export가 많이 들어가 있다.  
이건 import surface를 예쁘게 보이게는 하지만, 리팩토링 중에는 **검색성과 추적성을 떨어뜨린다.**

이번 정리에서는:

- `from model.data.dataset import PV26CanonicalDataset`
- `from model.net.trunk import build_yolo26n_trunk`

같이 **직접 import** 하는 게 낫다.

---

## 6. `tools/od_bootstrap/` 권장 구조

## 6.1 권장 구조 (3폴더 + 2파일)

```text
tools/od_bootstrap/
  data/
    aihub.py
    bdd100k.py
    source_prep.py
    teacher_dataset.py
    image_list.py
    exhaustive_od.py
    final_dataset.py
    debug_vis.py
  teacher/
    train.py
    eval.py
    calibrate.py
    ultralytics_runner.py
    policy.py
  cli.py
  presets.py
```

## 6.2 의미

### `data/`
dataset artifact를 만드는 코드 전부

- raw source standardize
- canonical bundle 준비
- teacher train dataset 제작
- exhaustive OD materialize
- final merged dataset 제작
- debug visualization

### `teacher/`
teacher 모델 lifecycle 전부

- teacher train
- teacher eval
- calibration
- Ultralytics 연결
- class policy

### `cli.py`
엔트리포인트 하나로 통합

예시:

```bash
python -m tools.od_bootstrap prepare-sources
python -m tools.od_bootstrap build-teacher-datasets
python -m tools.od_bootstrap train --teacher mobility
python -m tools.od_bootstrap eval --teacher mobility
python -m tools.od_bootstrap calibrate
python -m tools.od_bootstrap build-exhaustive-od
python -m tools.od_bootstrap build-final-dataset
```

### `presets.py`
체크인 설정의 대부분을 이 파일로 이동

---

## 7. 파일 이동 매핑 초안

| 현재 경로 | 권장 이동 경로 |
|---|---|
| `model/loading/pv26_loader.py` | `model/data/dataset.py` |
| `model/loading/transform.py` | `model/data/transform.py` |
| `model/loading/sampler.py` | `model/data/dataset.py` 또는 `model/data/sampler.py` |
| `model/encoding/pv26_target_encoder.py` | `model/data/target_encoder.py` |
| `model/trunk/ultralytics_yolo26.py` | `model/net/trunk.py` |
| `model/heads/pv26_heads.py` | `model/net/heads.py` |
| `model/eval/postprocess.py` | `model/engine/postprocess.py` |
| `model/eval/metrics.py` | `model/engine/metrics.py` |
| `model/eval/pv26_evaluator.py` | `model/engine/evaluator.py` |
| `model/loss/runtime.py` | `model/engine/loss.py` |
| `model/loss/spec.py` | `model/engine/loss.py` 또는 `common/pv26_schema.py` 와 일부 분리 |
| `model/training/pv26_trainer.py` | `model/engine/trainer.py` |
| `model/viz/overlay.py` | `model/data/preview.py` 또는 `common/overlay.py` |
| `model/preprocess/aihub_standardize.py` | `tools/od_bootstrap/data/aihub.py` |
| `model/preprocess/bdd100k_standardize.py` | `tools/od_bootstrap/data/bdd100k.py` |
| `model/preprocess/aihub_common.py` | `tools/od_bootstrap/data/source_common.py` 또는 `common/io.py` + source helper로 분리 |
| `tools/od_bootstrap/preprocess/sources.py` | `tools/od_bootstrap/data/source_prep.py` |
| `tools/od_bootstrap/preprocess/teacher_dataset.py` | `tools/od_bootstrap/data/teacher_dataset.py` |
| `tools/od_bootstrap/sweep/image_list.py` | `tools/od_bootstrap/data/image_list.py` |
| `tools/od_bootstrap/sweep/materialize.py` | `tools/od_bootstrap/data/exhaustive_od.py` |
| `tools/od_bootstrap/finalize/final_dataset.py` | `tools/od_bootstrap/data/final_dataset.py` |
| `tools/od_bootstrap/smoke/review.py` | `common/overlay_scene.py` 또는 `tools/od_bootstrap/data/review.py` |
| `tools/run_aihub_standardize.py` | 삭제 |
| `tools/run_bdd100k_standardize.py` | 삭제 |
| `tools/run_pv26_tiny_overfit_smoke.py` | 삭제 |

---

## 8. 공용 모듈 설계

## 8.1 가장 먼저 뽑아야 할 파일: `common/pv26_schema.py`

이 파일은 이번 리팩토링의 첫 커밋이어야 한다.

### 여기로 옮길 것
- `OD_CLASSES`
- `OD_CLASS_TO_ID`
- `TL_BITS`
- `LANE_CLASSES`
- `LANE_TYPES`
- dataset key 상수
- `SOURCE_MASK_BY_DATASET`
- `DET_SUPERVISION_BY_DATASET`
- exhaustive dataset key mapping

### 이유
지금은 이 정보들이 `aihub_standardize.py` 안에 들어 있어서 **runtime과 preprocessing이 엉켜 있다.**

`common/pv26_schema.py` 로 뽑으면:

- model이 standardizer를 안 봐도 됨
- od_bootstrap도 standardizer를 안 봐도 됨
- contract 변경 지점이 1개가 됨

## 8.2 `common/io.py`
공통 파일 입출력 유틸

- `read_json`
- `write_json`
- `write_text`
- `link_or_copy`
- `now_iso`

## 8.3 `common/paths.py`
공통 경로 유틸

- `resolve_path`
- `resolve_latest_root`
- repo root / dataset root 계산

---

## 9. YAML 정리 방안

## 9.1 원칙
체크인된 YAML은 **입력 preset** 으로는 거의 없애고,  
필요하면 **실행 결과 manifest** 만 남긴다.

## 9.2 삭제 권장 대상

### 루트
- `config/pv26_meta_train.default.yaml`
- `config/pv26_meta_train.smoke.yaml`
- `config/pv26_meta_train.stage3_vram_stress.yaml`

### OD bootstrap
- `tools/od_bootstrap/config/train/*`
- `tools/od_bootstrap/config/eval/*`
- `tools/od_bootstrap/config/preprocess/*`
- `tools/od_bootstrap/config/finalize/*`
- `tools/od_bootstrap/config/sweep/model_centric.default.yaml`
- `tools/od_bootstrap/config/sweep/model_centric.smoke.yaml`
- `tools/od_bootstrap/config/sweep/model_centric.dryrun.yaml`
- `tools/od_bootstrap/config/sweep/class_policy.template.yaml`
- `tools/od_bootstrap/config/sweep/class_policy.smoke.yaml`
- `tools/od_bootstrap/config/calibration/class_policy.default.yaml`

## 9.3 유지 가능한 YAML
아래는 **체크인 preset** 이 아니라 **런타임 산출물** 로는 유지 가능하다.

- Ultralytics용 generated `data.yaml`
- calibration 결과 `class_policy.yaml`
- run manifest / summary JSON/YAML

## 9.4 대체 방식

예를 들어 `tools/od_bootstrap/teacher/train.py` 상단에:

```python
# ===== User-tunable parameters =====
DEFAULT_DEVICE = "cuda:0"
DEFAULT_WORKERS = 8
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs" / "od_bootstrap" / "train"

TEACHER_PRESETS = {
    "mobility": {
        "dataset_root": REPO_ROOT / "seg_dataset" / "pv26_od_bootstrap" / "teacher_datasets" / "mobility",
        "model_size": "s",
        "weights": "yolo26s.pt",
        "class_names": ("vehicle", "bike", "pedestrian"),
        "epochs": 200,
        "batch": 20,
    },
    "signal": {
        ...
    },
    "obstacle": {
        ...
    },
}
# ===============================
```

이 정도면 실제 tweak 포인트가 import 블록 바로 아래에 모이고,  
YAML 7~8개를 헤맬 이유가 없다.

---

## 10. `smoke`, `dryrun` 제거 방안

## 10.1 지울 대상

### 파일/폴더
- `tools/od_bootstrap/smoke/` 전체
- `tools/run_pv26_tiny_overfit_smoke.py`
- `*.smoke.yaml`
- `*.dryrun.yaml`

### 코드/문서 용어
- `stage_0_smoke`
- `smoke-first`
- `smoke review`
- `smoke image list`
- `dry_run`

## 10.2 단, 바로 삭제하면 안 되는 것
`smoke/` 안에 있는 기능 중 일부는 재사용 가치가 있다.

### 살려서 이사할 것
- `canonical_scene_to_overlay_scene`  
  → `common/overlay_scene.py` 또는 `tools/od_bootstrap/data/review.py`
- checkpoint audit가 유효하다면  
  → `tools/od_bootstrap/teacher/audit.py`
- subset 이미지 리스트가 유효하다면  
  → test fixture generator 또는 `tools/od_bootstrap/data/sample_manifest.py`

## 10.3 대체 테스트 체계

### 기존 “smoke” 역할을 대체할 테스트
1. **unit**
   - 순수 함수
   - box/policy/schema/transform

2. **integration**
   - raw fixture → canonical output
   - canonical → teacher dataset
   - teacher predictions → exhaustive dataset
   - final dataset build

3. **e2e**
   - 아주 작은 fixture로 파이프라인 전체 1회 통과

4. **gpu(optional)**
   - trainer 1~2 step
   - ultralytics 연동

### 명명 규칙
- `smoke` 대신 `integration`, `e2e`, `gpu`, `regression`
- `dryrun` 대신 **없앰**  
  필요한 검증은 test가 맡는다.

---

## 11. 표준화 코드 소유권 정리

## 11.1 왜 `model/preprocess` 가 어색한가

지금 `model/preprocess` 는 사실상 아래 일을 한다.

- raw AIHUB/BDD 읽기
- canonical dataset 생성
- meta report 생성
- debug overlay 생성

이건 **학습 모델의 내부 전처리** 가 아니라 **dataset build pipeline** 이다.

즉, 이름과 위치 둘 다 어색하다.

## 11.2 권장 원칙

### `model/` 의 책임
- final dataset loader
- transform
- target encode
- network
- loss
- trainer/evaluator

### `tools/od_bootstrap/` 의 책임
- raw data 이해
- raw → canonical
- canonical → teacher dataset
- teacher train/eval/calibration
- exhaustive OD materialization
- final merged dataset build

이렇게 경계를 자르면 mental model이 아주 선명해진다.

---

## 12. 테스트 재배치 제안

현재 model test 중 일부가 standardizer를 직접 import 해서 fixture를 만든다.  
이건 refactor 후에는 의존 방향이 어색해진다.

### 권장 재배치

#### `test/model/`
- loader
- transform
- target encoder
- loss
- trainer
- evaluator

이 테스트들은 **synthetic canonical fixture builder** 를 써서 dataset를 만든다.  
raw AIHUB/Bdd 표준화 코드를 직접 import 하지 않는다.

#### `test/od_bootstrap/`
- aihub/bdd standardize
- source prep
- teacher dataset build
- teacher train/eval/calibration
- exhaustive OD build
- final dataset build

즉, **raw→dataset build는 bootstrap test**, **final dataset consumer는 model test** 로 나눈다.

---

## 13. 권장 실행 흐름 (리팩토링 후)

```text
raw AIHUB + BDD100K
  -> tools.od_bootstrap.data.(aihub / bdd100k)
  -> canonical bundle
  -> teacher dataset build
  -> teacher train/eval/calibrate
  -> exhaustive OD dataset
  -> final PV26 exhaustive OD lane dataset
  -> tools/run_pv26_train.py
```

여기서 `model/` 은 마지막 두 줄만 관심 있으면 된다.

---

## 14. 구현 순서 제안

## Phase 1. shared contract 추출
### 목표
- `common/pv26_schema.py`
- `common/io.py`
- `common/paths.py`

### 이 단계에서 하는 일
- `OD_CLASSES`, `TL_BITS`, `LANE_CLASSES`, `LANE_TYPES` 이동
- loader/loss/encoder/sweep/finalize import 교체
- 동작 변화 없이 import 방향만 정리

### 이유
가장 큰 구조적 얽힘을 제일 먼저 자른다.

---

## Phase 2. standardizer 이사
### 목표
- `model/preprocess/*` → `tools/od_bootstrap/data/*`

### 이 단계에서 하는 일
- `tools/run_aihub_standardize.py`
- `tools/run_bdd100k_standardize.py`

를 삭제하거나, 잠깐 shim으로만 남긴 뒤 최종 삭제

### 결과
`model/` 에서 raw source 책임이 빠진다.

---

## Phase 3. `model/` 3폴더화
### 목표
- `data/`, `net/`, `engine/`

### 이 단계에서 하는 일
- `loading/`, `encoding/`, `viz/` → `data/`
- `trunk/`, `heads/` → `net/`
- `loss/`, `eval/`, `training/` → `engine/`
- re-export `__init__.py` 제거
- 직접 import 방식으로 전환

---

## Phase 4. YAML 제거 + preset화
### 목표
- 체크인 YAML 대거 삭제
- `presets.py` 도입

### 이 단계에서 하는 일
- `train/scenario.py`, `eval/scenario.py`, `calibration/scenario.py`, `sweep/scenario.py`
  같은 YAML parser 계층 제거
- 실행 파일 import 블록 아래에 user-tunable preset 배치

---

## Phase 5. `smoke`, `dryrun` purge
### 목표
- 용어/파일/코드/문서/테스트에서 전부 제거

### 이 단계에서 하는 일
- reusable 함수 이사
- test 이름 정리
- docs 철학 수정
- `stage_0_smoke` 제거

---

## Phase 6. 문서 정리
### 바꿔야 할 문서 성격
- `smoke-first` → `test-first`
- `standardization layer in model` → `bootstrap-owned pipeline`
- 실행 커맨드 전부 최신화

---

## 15. 당장 첫 커밋으로 추천하는 것

가장 먼저 아래 3개만 해도 전체가 훨씬 쉬워진다.

### Commit 1
`common/pv26_schema.py` 생성 후 아래 import 전환
- `model/loading/pv26_loader.py`
- `model/loss/spec.py`
- `model/encoding/pv26_target_encoder.py`
- `tools/od_bootstrap/d/materialize.py`
- `tools/od_bootstrap/finalize/final_dataset.py`
- `tools/od_bootstrap/preprocess/teacher_dataset.py`

### Commit 2
`tools/od_bootstrap/data/` 만들고 standardize 코드 이동  
`model/preprocess/` 는 deprecated shim만 잠깐 남김

### Commit 3
`tools/od_bootstrap/smoke/review.py` 의 재사용 함수 분리  
`preprocess/debug_vis.py` 가 더 이상 `smoke/` 를 import 하지 않게 정리

이 3개가 끝나면 나머지는 훨씬 mechanical 해진다.

---

## 16. 최종 추천안

### 최종 선택
**옵션 B. 철학 일치형 리팩토링안**

### 이유
- 당신의 현재 개발 철학과 정확히 맞다.
- old canonical-only 시대의 흔적을 제대로 걷어낼 수 있다.
- `model/` 과 `od_bootstrap/` 의 경계가 가장 선명해진다.
- YAML / smoke / dryrun 정리를 한 번에 끝낼 수 있다.
- 이후 신규 팀원이 들어와도 “어디가 raw pipeline이고 어디가 training runtime인지” 바로 이해된다.

### 최종 목표 구조
```text
common/
model/
  data/
  net/
  engine/
tools/
  run_pv26_train.py
  od_bootstrap/
    data/
    teacher/
    cli.py
    presets.py
test/
  model/
  od_bootstrap/
  fixtures/
```

---

## 17. 마지막 판단

당신이 지금 하려는 건 단순한 폴더 미화가 아니라 **프로젝트 경계 재정의** 다.

이번 리팩토링의 핵심은:

- 디렉토리 수 줄이기
- YAML 줄이기
- smoke 없애기

이 3개 자체가 아니라,

> **“PV26 model runtime”과 “OD bootstrap dataset build pipeline”을 완전히 분리하는 것**

이다.

이 방향으로 가면 이후 코드 수정 기준도 매우 단순해진다.

- raw source를 만지면 `tools/od_bootstrap`
- final dataset를 읽어 학습하면 `model`
- 둘 다 쓰는 계약은 `common`

이 기준이 서면, 나머지 세부 리팩토링은 거의 자동으로 정리된다.
