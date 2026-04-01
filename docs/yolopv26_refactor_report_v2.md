# YOLOPV26 refactor_2 구조 조사 및 개선안 보고서

## 0. 조사 범위와 확인 방법

대상은 `yolopv26-refactor_2.zip` 전체 저장소이며, 디렉토리 구조, 각 폴더의 Python 소스, 설정 파일, 문서, 테스트를 같이 봤다.  
추가로 이전 `yolopv26-refactor.zip`와 간단 비교를 해서 이번 refactor_2가 실제로 얼마나 정리되었는지도 같이 봤다.

이 문서는 원래 `check_env.py` TUI 적용 전에 받은 리뷰를 바탕으로 시작했지만, 현재 저장소 상태에서 바로 읽어도 오해가 없도록 stale 수치와 충돌하는 권고를 함께 보정했다.

이번 조사에서 실제로 확인한 것:

- 저장소 전체 트리/파일 수/LOC 비교
- `model/`, `tools/od_bootstrap/`, `tools/run_pv26_train.py`, `common/`, `config/`, `test/` 정적 리뷰
- `python -m compileall` 기준 문법 컴파일 확인
- 일부 테스트 실제 실행

실행 결과 요약:

- `python -m compileall`: 성공
- `pytest -q test/test_docs_sync.py test/test_portability_runtime.py`: **14 passed**
- `pytest -q test/od_bootstrap/test_sweep_schema.py test/od_bootstrap/test_sweep_policy.py test/od_bootstrap/test_sample_helpers.py test/od_bootstrap/test_final_dataset.py test/od_bootstrap/test_teacher_dataset.py`: **13 passed**
- `pytest -q test/od_bootstrap`: **44 passed / 2 failed**
  - 남은 실패 2개는 `ultralytics` trainer dependency 부족이 아니라, sweep preset/class policy가 현재 workspace의 `runs/od_bootstrap/calibration/default/class_policy.yaml`를 반영하는 구조 때문에 테스트 fixture가 repo 상태를 충분히 격리하지 못한 데서 나온다.
  - 즉, 현재 기준으로는 **od_bootstrap의 핵심 데이터/스키마/머티리얼라이즈 계층은 잘 버티고 있고**, 남은 일은 teacher runtime 폴백 확대보다 sweep 관련 테스트를 deterministic하게 닫는 쪽에 가깝다.

## 1. 한 줄 총평

이번 `refactor_2`는 **방향이 맞고, 이전 refactor보다 분명히 좋아졌다.**  
특히 네가 원래 원하던 방향인 아래 네 가지는 거의 제대로 반영되어 있다.

1. `model/`을 `data / net / engine` 3축으로 줄인 것  
2. raw standardization 책임을 `tools/od_bootstrap` 안으로 밀어 넣은 것  
3. `smoke`, `dryrun` 류를 active code에서 거의 없앤 것  
4. formal test와 docs sync를 넣어서 “조용히 구조가 다시 망가지는 것”을 막은 것

다만 지금 남아 있는 큰 문제는 **디렉토리 수 자체가 아니라 `od_bootstrap` 내부의 간접 계층, 설정 중복, giant module`**이다.  
즉, “폴더는 줄였는데, 복잡도는 wrapper + `_impl` + YAML 중복 + 거대한 파일로 남아 있다”가 정확한 진단이다.

## 2. 이전 refactor 대비 객관적 변화

| 항목 | 이전 refactor | refactor_2 | 해석 |
|---|---:|---:|---|
| 전체 디렉토리 수 | 31 | 14 | 눈에 띄게 정리됨 |
| 전체 Python 파일 수 | 107 | 110 | 파일 수는 비슷함 |
| 전체 YAML 수 | 24 | 3 | 설정 폭발이 크게 줄어듦 |
| `model/` 하위 디렉토리 수 | 9 | 3 | 네가 원하던 방향 거의 달성 |
| `tools/od_bootstrap/` 하위 디렉토리 수 | 15 | 3 | 폴더 난립 정리 성공 |
| `tools/od_bootstrap/` YAML 수 | 21 | 0 | bootstrap 내부 설정 폴더 삭제 성공 |
| `tools/od_bootstrap/` Python LOC | 7,320 | 11,198 | 폴더/YAML은 줄었지만 실제 구현은 커짐 |

이 표에서 중요한 포인트는 하나다.

**이번 refactor_2는 “폴더 구조 청소”는 성공했지만, 복잡도가 사라진 것은 아니다.**  
복잡도는 이제 `tools/od_bootstrap` 내부 Python 코드로 이동했다.

## 3. 폴더별 진단

### 3.1 `common/`

좋은 점:

- `pv26_schema.py`가 dataset key, supervision mask, exhaustive dataset key mapping을 한 곳에 모으고 있다.
- `user_config.py`가 YAML 읽기/merge를 중앙화한다.
- `io.py`, `boxes.py`, `overlay.py`는 범용 재사용 포인트가 맞다.

아쉬운 점:

- `tools/od_bootstrap/common/`가 사실상 여기의 얇은 re-export wrapper다.
- `now_iso`, `write_json`, `link_or_copy`, path/coerce 계열은 아직도 각 파일 안에 중복 정의가 많다.

판단:

- `common/`은 **좋은 축**이다.
- 대신 `tools/od_bootstrap/common/`는 지우고 여기로 완전히 흡수하는 게 맞다.

### 3.2 `config/`

좋은 점:

- 이전처럼 preset YAML 폴더가 터져 있지 않고, 지금은 `user_paths.yaml`, `od_bootstrap_hyperparameters.yaml`, `pv26_train_hyperparameters.yaml` 3개만 남았다.
- 구조만 보면 많이 좋아졌다.

아쉬운 점:

- **아직도 설정의 실제 source of truth가 두 군데다.**
- `tools/od_bootstrap/presets.py` 안에 기본값이 이미 있고, 같은 값이 `od_bootstrap_hyperparameters.yaml`에도 다시 있다.
- `tools/run_pv26_train.py` 안에도 기본 preset/phase/default가 이미 있고, 같은 값이 `pv26_train_hyperparameters.yaml`에도 다시 있다.

즉 현재 구조는:

- 코드에 default 있음
- YAML에 거의 같은 default 또 있음
- 둘을 merge 함

이 방식은 “유연해 보이지만”, 실제론 **중복 유지보수 비용**만 만든다.

판단:

- default가 코드와 YAML에 나뉘어 있는 부담은 여전히 있다.
- 다만 현재 저장소의 실제 UX는 `check_env.py` interactive TUI, README, docs sync test가 모두
  `user_paths.yaml`, `od_bootstrap_hyperparameters.yaml`, `pv26_train_hyperparameters.yaml` 3개를 공식 조절 지점으로 전제하는 상태다.
- 그래서 지금 당장 config를 코드 상수 중심으로 회수하는 것은 우선순위가 아니다.
- 먼저 해야 할 일은 이 3개 YAML의 역할을 분명히 유지하고, TUI/help/README/test가 같은 조절 지점을 일관되게 가리키도록 맞추는 것이다.

유저 의견:
- GPT는 config를 더 줄이라고 하지만 유저는 지금은 나쁘지 않다고 생각한다.
- 사용자 입장에서는 차라리 그때그때 사용자가 진짜 바꿔야되는 값들만 옆에 인라인 한국어 주석으로 달아두고 안 쓰는 주석은 그냥 간단하게 방치해두면 될 것 같다고 생각한다. 아니면 변경 필요 없다고 인라인주석 달거나. 
- 따라서 현재 config 및 파라미터 조정 구조는 양호하며 변경 필요성이 매우 낮다는 유저의 평가.


### 3.3 `docs/`

좋은 점:

- 문서 수는 많지만, “지금 구조를 설명하는 문서”로 일관돼 있다.
- 특히 `test/test_docs_sync.py`가 아주 좋다.
- 이 테스트 덕분에 예전 구조(`model/preprocess`, `tools/run_aihub_standardize.py`, `tools/od_bootstrap/config/` 등)가 문서에 다시 섞이는 걸 막고 있다.

판단:

- 이 저장소에서 **가장 잘 짠 부분 중 하나**다.
- 문서를 줄이는 게 목적이 아니라면 지금 방향 유지가 맞다.

### 3.4 `model/`

좋은 점:

- 이번 refactor_2의 가장 깔끔한 성과다.
- 예전 `encoding / eval / heads / loading / loss / preprocess / training / trunk / viz`에서
  지금 `data / net / engine`으로 바뀐 것은 매우 좋다.
- 네가 원했던 “큰 축 3개”가 실제로 구현됐다.

아쉬운 점:

- `model/engine/trainer.py`가 **1,666 LOC**로 너무 크다.
- `model/engine/spec.py`는 사실상 `_loss_spec.py` wrapper다.
- `model/data/preview.py`도 `common.overlay` thin wrapper다.

판단:

- 구조적 방향은 이미 맞다.
- 여기서 더 손댈 것은 “폴더 개편”이 아니라 **큰 파일을 조금만 분해하는 정도**다.
- 즉 `model/`은 이제 거의 안정 구간으로 봐도 된다.

### 3.5 `test/`

좋은 점:

- 네가 싫어한 `smoke`, `dryrun` 대신 정식 테스트가 자리 잡고 있다.
- 특히 `test/od_bootstrap/`가 별도 하위 폴더로 정리되어 있고, 스키마/정책/데이터셋/머티리얼라이즈/teacher 툴링이 분리돼 있다.
- docs sync test까지 있는 점이 좋다.

보완점:

- 현재 남아 있는 `od_bootstrap` 실패 2개는 teacher runtime dependency 부족이 아니라 sweep preset/class policy가 workspace 상태를 흡수하는 정합성 문제다.
- 특히 `build_sweep_preset()`이 `runs/od_bootstrap/calibration/default/class_policy.yaml`를 우선 읽는 정책이 있으므로, 테스트는 이를 명시적으로 격리해야 한다.

판단:

- 테스트 철학은 맞다.
- 남은 일은 “test를 더 만들기”보다 **환경/산출물 의존 경계를 더 명확히 드러내고, 테스트 fixture를 deterministic하게 만드는 것**이다.

유저의 의견:
- GPT는 계속 의존성이 없는 환경에서도 graceful하게 뭐 폴백을 만들고 어쩌고 하는데
- 폴백도 어지간히 하고 graceful하게 이어지는것은 개발자 입장에서 꽤나 경계해야하는 **위험** 한 접근법이다. 
- 예외사항이 발생하면 원인과 로깅을 명확히 해야지 스리슬쩍 냄새가 나니 덮는다 식 대응은 스노우볼을 크게 굴리는 작업이다. 
- check env에서 tui interactive하게 상황 스캔이 가능하므로 환경이 맞지 않으면 그냥 기본 에러메시지정도를 잘 띄워주고, 오류가 나면 확실히 오류를 짚어야지 
- 실시간 작동/멈추면 안 됨 이런 목표가 절대로 아니므로 문제가 있으면 명확히 밝히는 접근법으로 가야한다. 이것은 문서화에도 개발철학으로 넣을 것. 

### 3.6 `tools/`

좋은 점:

- top-level 도구가 `check_env.py`, `run_pv26_train.py`, `od_bootstrap/`로 단순해졌다.
- `check_env.py`가 현재는 interactive TUI 상태 허브 역할까지 맡으면서, runtime 점검과 다음 액션 선택을 한 화면에서 처리하게 됐다.
- 네 철학대로 bootstrap과 final dataset build의 책임이 `od_bootstrap` 안으로 들어왔다.
- 실제로 docs sync test가 예전 standalone standardize script 부활을 막고 있다.

보완점:

- `tools/run_pv26_train.py`가 **1,499 LOC**로 여전히 너무 크다.
- 그리고 여기엔 아직 `LEGACY_CANONICAL_BDD_ROOT`, `include_bdd` 같은 backward compatibility 가지가 남아 있다.
- 네 현재 철학이 “최종 학습 입력은 exhaustive OD + lane merged dataset”이라면, 이 legacy branch는 치우는 편이 맞다.
- `stage3_vram_stress` preset도 active config에 남아 있는데, 네 취향 기준으로는 test/dev 영역으로 보내는 게 더 일관적이다.

판단:

- `tools/` 전체는 좋아졌지만, 다음 정리 포인트는 `run_pv26_train.py`와 `od_bootstrap/` 내부다.

## 4. `tools/od_bootstrap/` 상세 진단

## 4.1 지금 잘 된 점

현재 `od_bootstrap`는 기능적으로는 꽤 명확하다.

- single CLI entrypoint: `python -m tools.od_bootstrap`
- source prep → teacher dataset → teacher train/eval/calibrate → exhaustive OD → final dataset build 흐름이 맞다
- provenance 필드, class policy, review/debug tooling이 분리되어 있다
- raw standardization이 바깥 레이어가 아니라 bootstrap 안에 들어왔다
- formal tests가 붙어 있다

즉 **파이프라인 철학은 맞다.**  
문제는 철학이 아니라 내부 레이아웃과 구현 스타일이다.

## 4.2 지금 가장 큰 구조적 문제

### A. wrapper + `_impl` 2중 구조가 너무 많다

`tools/od_bootstrap` 안의 Python 파일은 총 48개인데, 그중 **19개가 15줄 이하의 얇은 wrapper**다.  
또 `_impl` 계열 파일이 **14개**나 있다.

대표 예:

- `data/debug_vis.py` → `_debug_vis_impl.py`
- `data/exhaustive_od.py` → `_exhaustive_od_impl.py`
- `data/final_dataset.py` → `_final_dataset_impl.py`
- `teacher/calibrate.py` → `_calibrate_impl.py`
- `teacher/eval.py` → `_eval_impl.py`
- `teacher/policy.py` → `_policy_impl.py`
- `teacher/ultralytics_runner.py` → `_ultralytics_runner_impl.py`

이 구조는 폴더를 줄이는 데는 성공했지만, **읽는 사람 입장에서는 오히려 한 번 더 점프해야 해서 피로하다.**  
게다가 `import *` re-export가 많아서 실제 정의 위치를 따라가기 더 어려워진다.

내 판단은 이렇다.

- 공개 API가 필요한 건 맞다.
- 하지만 지금은 “패키지 안정화”보다 “코드 가독성”이 더 중요하다.
- 따라서 **wrapper를 대량으로 유지할 가치가 크지 않다.**

### B. `data/` 폴더가 너무 많은 책임을 갖고 있다

현재 `tools/od_bootstrap/data/` 안에는 사실상 네 종류의 일이 섞여 있다.

1. raw source standardization
2. teacher dataset build
3. sweep/exhaustive/final materialization
4. review/debug/audit

대략 LOC로 보면 아래 정도다.

- source 계열: 약 **4,294 LOC**
- teacher dataset build: 약 **421 LOC**
- materialize 계열: 약 **1,397 LOC**
- QA/debug 계열: 약 **780 LOC**

즉 이름은 `data/`지만, 실제로는 **source + build + qa**를 한 폴더에 몰아넣은 상태다.  
지금 디렉토리 수가 적은 건 좋은데, 그 대가로 `data/`가 “무엇이든 다 들어가는 폴더”가 되어버렸다.

### C. 공통 기능 재사용이 아직 부족하다

중복이 보이는 대표 예:

- `tools/run_pv26_train.py`와 `tools/od_bootstrap/presets.py`에  
  `_coerce_mapping`, `_coerce_bool`, `_coerce_int`, `_coerce_float`, `_coerce_str`가 **동일 구현**으로 중복
- `model/engine/trainer.py`와 `tools/od_bootstrap/teacher/_ultralytics_runner_impl.py`에  
  `_flatten_scalar_tree`가 **동일 구현**으로 중복
- `_now_iso`는 코드베이스에 여러 번 반복
- `_write_json`, `_link_or_copy`, `_default_io_workers`도 반복

이건 단순 미관 문제가 아니라,  
나중에 로깅 포맷/JSON 출력/typing 정책/설정 coercion을 바꾸려 할 때 **수정 포인트가 여러 군데로 찢어진다**는 뜻이다.

### D. giant module 두 개가 복잡도를 끌어올린다

특히 아래 둘이 크다.

- `tools/od_bootstrap/data/_aihub_standardize_impl.py`: **2,334 LOC**
- `tools/od_bootstrap/teacher/_ultralytics_runner_impl.py`: **1,327 LOC**

둘 다 파일 하나가 너무 많은 일을 한다.

`_aihub_standardize_impl.py`는 안에서:

- lane / traffic / obstacle worker
- inventory
- readme generation
- failure manifest
- qa summary
- debug vis generation

까지 다 한다.

`_ultralytics_runner_impl.py`는 안에서:

- checkpoint resume 탐색
- trainer class 구성
- dataloader patch
- tensorboard payload
- timing/profile
- progress renderer
- latest checkpoint alias 갱신

까지 다 한다.

이 둘은 “폴더 수”보다 “파일 책임”이 문제다.

### E. dev/legacy residue가 아직 남아 있다

대표적으로:

- `tools/run_pv26_train.py`의 `LEGACY_CANONICAL_BDD_ROOT`, `include_bdd` backward compatibility
- `config/pv26_train_hyperparameters.yaml`와 `tools/run_pv26_train.py` 안의 `stage3_vram_stress`
- `tools/od_bootstrap/data/checkpoint_audit.py` 안의 **날짜 박힌 고정 checkpoint path**

이런 것들은 지금 프로젝트의 “주요 정식 파이프라인”이라기보다, 개발/점검 흔적에 가깝다.  
남겨도 되지만, **core path와 같은 레벨에 두면 설계가 흐려진다.**

## 4.3 `od_bootstrap`에서 묶을 수 있는 것

아래는 현재 파일들을 “같은 책임끼리” 다시 묶는 제안이다.

| 현재 파일 | 권장 묶음 | 이유 |
|---|---|---|
| `tools/od_bootstrap/common/boxes.py`, `tools/od_bootstrap/common/paths.py` | 삭제 후 `common.boxes`, `common.paths` 직접 사용 | 현재는 wrapper만 수행 |
| `data/aihub.py` + `_aihub_standardize_impl.py` | `source/aihub.py` | 공개/구현 이중화 제거 |
| `data/bdd100k.py` + `_bdd100k_standardize_impl.py` | `source/bdd100k.py` | 동일 |
| `data/source_prep.py` + `_source_prep_impl.py` + `source_common.py` | `source/prepare.py` | source orchestration 한 곳으로 |
| `_teacher_dataset_impl.py` + `teacher_dataset.py` + `teacher/data_yaml.py` | `teacher/dataset.py` | teacher dataset 관련 책임 통합 |
| `_train_impl.py` + `train.py`, `_eval_impl.py` + `eval.py`, `_calibrate_impl.py` + `calibrate.py` | `teacher/ops.py` 또는 `teacher/train.py`, `teacher/eval.py`, `teacher/calibrate.py` 단일 파일화 | wrapper 제거 |
| `_ultralytics_runner_impl.py` + `ultralytics_runner.py` | `teacher/runtime.py` | runtime 경계 한 곳 |
| `_image_list_impl.py` + `sample_manifest.py` + `artifacts.py` + `sweep_types.py` | `build/manifest.py` + `build/types.py` | materialization metadata/contract 응집 |
| `_sweep_impl.py` + `sweep.py` | `build/sweep.py` | wrapper 제거 |
| `_exhaustive_od_impl.py` + `exhaustive_od.py` | `build/exhaustive.py` | wrapper 제거 |
| `_final_dataset_impl.py` + `final_dataset.py` | `build/final_dataset.py` | wrapper 제거 |
| `_debug_vis_impl.py` + `debug_vis.py` + `review.py` + `checkpoint_audit.py` | `build/qa.py` 또는 `qa/` | QA/debug/inspection은 core pipeline과 분리 |

핵심은 하나다.

**지금 `od_bootstrap`는 폴더 수는 적지만, 책임 묶음은 흐리다.**  
“파일 타입별”이 아니라 “파이프라인 단계별”로 다시 묶는 게 맞다.

## 5. 추천 목표 구조

내가 가장 추천하는 구조는 아래다.  
디렉토리는 `3개`만 유지하고, 나머지는 파일 단위로 정리하는 방식이다.

```text
tools/od_bootstrap/
  __main__.py
  cli.py
  config.py          # 지금 presets.py + 일부 user_config/coerce helper 흡수
  shared.py          # 또는 io.py / manifest.py 등 root-level support file

  source/
    aihub.py
    bdd100k.py
    prepare.py
    raw_common.py

  teacher/
    dataset.py
    runtime.py
    policy.py
    ops.py
    types.py

  build/
    manifest.py
    sweep.py
    exhaustive.py
    final_dataset.py
    qa.py
```

이 구조의 장점:

- 폴더 수는 여전히 적다
- `source / teacher / build`가 실제 파이프라인 단계와 맞는다
- 지금 `data/`에 뒤섞인 책임이 분리된다
- `common/` wrapper 폴더를 지울 수 있다
- wrapper + `_impl` 2중 구조를 없앨 수 있다

### 보수적 대안

리스크를 더 낮추고 싶으면 폴더 이름은 유지하고 아래만 먼저 해도 된다.

- `tools/od_bootstrap/common/` 삭제
- 모든 `_impl` wrapper 제거
- `data/` 안을 파일명만 재정리
- `checkpoint_audit.py`를 core path 밖으로 이동
- `presets.py`와 `run_pv26_train.py`의 중복 config helper 추출

즉, **폴더 개편 없이도 1차 효과는 충분히 볼 수 있다.**

## 6. 설정 정리 메모

현재 기준 권장안은 “config를 더 줄이자”가 아니라 **현재 3개 YAML + TUI help + README 계약을 안정적으로 유지하자** 쪽이다.

### 현재 기준 권장안

- `config/user_paths.yaml`, `config/od_bootstrap_hyperparameters.yaml`, `config/pv26_train_hyperparameters.yaml` 3개는 유지한다.
- interactive TUI와 README, docs sync test가 이 3개 파일을 공식 조절 지점으로 같이 가리키도록 맞춘다.
- 코드와 YAML 사이 default 중복은 장기적으로 줄일 수 있지만, 그 작업은 legacy residue와 테스트 정합화를 먼저 끝낸 뒤에 다시 판단한다.

### 장기 검토 메모

- 나중에 config를 더 줄이고 싶다면, 현재 UX 계약을 먼저 다시 설계한 뒤 한 번에 옮기는 편이 낫다.
- 즉석에서 `user_paths.yaml`만 남기거나 `local_overrides.yaml`을 끼워 넣는 식의 절반짜리 축소는 지금 단계의 우선순위가 아니다.

## 7. 지금 코드에서 “잘 짠 부분”으로 남겨야 하는 것

이건 유지하는 게 맞다.

### 7.1 `model/ = data / net / engine`

이건 이번 refactor_2의 가장 좋은 성과다.  
여기서 다시 세분화 쪽으로 되돌아가면 안 된다.

### 7.2 `common/pv26_schema.py`

dataset key, source mask, exhaustive mapping, supervision policy가 한 곳에 모여 있는 건 좋다.  
오히려 `od_bootstrap` 쪽 metadata/schema도 이런 방향으로 더 정리하면 된다.

### 7.3 `test/test_docs_sync.py`

정말 좋다.  
문서와 현재 구조를 같이 락 걸어버리는 방식이라, refactor 후 다시 옛 구조가 스며드는 걸 막는다.

### 7.4 provenance / manifest 쪽 설계

`tools/od_bootstrap/data/artifacts.py`, `sweep_types.py` 류는 방향이 좋다.  
exhaustive OD의 provenance를 남기는 철학은 유지해야 한다.

### 7.5 single CLI entrypoint

`python -m tools.od_bootstrap` 단일 진입점은 좋다.  
여러 개의 scattered script로 다시 돌아갈 필요 없다.

## 8. 남겨두면 계속 불편할 부분

### 8.1 `run_pv26_train.py`의 legacy compatibility

지금 철학대로면 최종 입력은 `pv26_exhaustive_od_lane_dataset`다.  
그런데 `run_pv26_train.py`에는 아직 legacy BDD canonical branch가 남아 있다.

이건 “혹시 몰라서 남겨둔 것”에 가깝고, 앞으로는 오히려 헷갈림을 만든다.  
정리하는 게 맞다.

### 8.2 `stage3_vram_stress`

이건 이름만 다를 뿐, 네가 싫어하는 `smoke/dryrun` 계열의 개발 편의 preset에 가깝다.  
정식 기능이라면 test/dev 영역으로 보내고, 정식 기능이 아니라면 없애는 게 더 일관적이다.

### 8.3 `checkpoint_audit.py`의 고정 checkpoint path

코드 안에 날짜 박힌 체크포인트 경로를 기본값으로 넣어두는 건 장기적으로 좋지 않다.  
다만 현재는 README와 docs sync test가 이 파일을 공식 tooling으로 계속 언급하고 있으므로, 즉시 core path 밖으로 이동시키는 것은 우선순위가 아니다.  
먼저 이 도구를 계속 공식 지원할지, 아니면 dev helper로 내릴지를 결정해야 한다.

## 9. 우선순위별 실행 순서

### 1차: 동작 변경 없이 바로 할 수 있는 것

1. stale 문서/테스트 결과 정정
2. sweep preset/class policy 테스트를 workspace 상태와 분리
3. `tools/od_bootstrap/common/` 삭제
4. wrapper + `_impl` 쌍 제거
5. `import *` re-export 제거
6. `common`으로 `_now_iso`, `write_json`, `link_or_copy`, config coercion helper 추출

이 단계만 해도 체감이 꽤 클 것이다.

### 2차: `od_bootstrap` 구조 재배치

1. `data/`를 `source/`와 `build/`로 재편
2. `teacher/`는 유지하되 runtime / dataset / policy / ops 기준으로 파일 정리
3. `AIHUB` giant standardizer를 source 파일들로 조금 분해

이 단계가 핵심 리팩토링이다.

### 3차: 설정 철학 정리

1. 현재 3개 YAML + TUI help + README 계약을 유지할지
2. 이후 config 축소가 정말 필요한지
3. 필요하다면 어떤 UX로 대체할지

이 단계는 지금 당장 손대기보다, 앞선 정합화 작업 뒤에 다시 판단하는 편이 맞다.

### 4차: 남은 대형 파일 정리

- `tools/run_pv26_train.py`
- `model/engine/trainer.py`
- `tools/od_bootstrap/teacher/_ultralytics_runner_impl.py`
- `tools/od_bootstrap/data/_aihub_standardize_impl.py`

이 네 파일을 “기능 단위”로 쪼개면 구조 완성도가 확 올라간다.

## 10. 최종 결론

지금 `refactor_2`는 **초안이 아니라 이미 꽤 괜찮은 1차 완성본**에 가깝다.  
특히 네가 처음 원했던 큰 방향은 대부분 반영됐다.

정리하면:

- `model/` 구조: **거의 합격**
- standardization의 `od_bootstrap` 귀속: **맞는 방향**
- smoke/dryrun 제거: **대체로 달성**
- formal test / docs sync: **좋음**
- 남은 핵심 문제: **`od_bootstrap` 내부 복잡도와 설정 중복**

그래서 다음 수정 방향은  
“다시 큰 폴더 리팩토링을 하자”보다,

**`tools/od_bootstrap`를 `source / teacher / build` 3단계로 재정렬하고, wrapper + `_impl` + YAML 중복을 줄이는 것**  
이 제일 효과가 크다.

내 추천 1순위는 이거다:

> **추천안:** 문서/테스트 정합화 → sweep fixture 격리 → 그 다음 `od_bootstrap/common` 삭제와 wrapper 정리 → 필요하면 `data/`를 `source / build`로 재배치

이 순서대로 가면, 지금 잘 돌아가는 흐름을 크게 깨지 않으면서도  
“정말 잘 짠 코드” 쪽으로 한 단계 더 갈 수 있다.
