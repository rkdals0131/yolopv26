# YOLOPV26 main 코드 구조·재사용성·위험점 조사 보고서

## 0. 조사 범위

이번 보고서는 `yolopv26-main.zip` 기준으로 현재 저장소를 다시 정적 조사한 결과다.  
이번에는 이전처럼 폴더 개편 자체보다, 네가 말한 아래 기준에 맞춰 봤다.

1. **소스코드 역할 분리**가 충분히 됐는가
2. **모듈화와 재사용**이 더 가능한 부분이 남아 있는가
3. **중복 코드**가 실질적으로 줄었는가, 아직 남은 중복은 어떤 종류인가
4. **위험한 결합점**이나 향후 유지보수 때 터질 만한 지점이 있는가
5. user-facing UX는 이미 `tools/check_env.py` interactive launcher로 크게 개선됐다고 보고, 이번에는 **코드 내부 경계**를 중심으로 본다

추가로, 단순 눈검사만 하지 않고 아래 정도는 같이 확인했다.

- `common/`, `model/`, `tools/` 전반 정적 구조 조사
- 생산 코드/테스트 코드 규모와 큰 파일 분포 확인
- 반복 helper 이름/사설(private) cross-import 패턴 확인
- 선택 테스트 실행
  - `test/test_common_config_coercion.py`
  - `test/test_common_scalars.py`
  - `test/test_docs_sync.py`
  - `test/test_overlay.py`
  - `test/test_run_pv26_train.py` 일부
  - `test/od_bootstrap/test_preprocess_sources.py` 일부
  - `test/test_portability_runtime.py` 일부
  - `test/od_bootstrap/test_final_dataset.py`
  - `test/test_pv26_loss_spec.py`
  - `test/test_pv26_postprocess.py`
- 선택 실행 기준 **30개 테스트 전부 통과**
- `python -m compileall -q common model tools` 통과

즉, 이번 평가는 “대충 구조만 본 인상평”이 아니라, **실제 코드 결합 방식과 일부 핵심 테스트까지 확인한 상태**의 초안이다.

---

## 1. 한 줄 총평

지금 `main`은 **상위 구조는 거의 완성 단계**라고 봐도 된다.  
이제 더 중요한 건 디렉토리 대수술이 아니라, **코드 경계 정리**다.

내 판단으로는 지금 남은 핵심은 딱 이쪽이다.

- 반복되는 `json/path/time/link` helper 정리
- `private helper`를 다른 모듈이 끌어다 쓰는 **cross-import mesh** 제거
- `check_env.py`, `run_pv26_train.py`, `aihub.py`, `bdd100k.py`, `ultralytics_runner.py` 같은 **orchestration 거대 파일 분해**
- `dict[str, Any]` 중심 manifest/row 전달을 좀 더 **명시적인 계약**으로 바꾸기
- teacher runtime 쪽과 PV26 trainer 쪽의 **progress/tensorboard/runtime helper 중복** 줄이기

즉, 이제는 “폴더를 또 갈아엎는 단계”가 아니라  
**재사용 가능한 공통부를 드러내고, 사설 구현이 경계 밖으로 새는 부분을 막는 단계**다.

결론부터 말하면,

- **더 큰 top-level 리팩토링은 굳이 안 해도 된다.**
- 대신 **코드 단위의 경계 정리**는 아직 꽤 할 수 있다.
- 특히 `tools/od_bootstrap/source`와 `tools/run_pv26_train.py`, `tools/check_env.py`가 다음 타깃이다.

---

## 2. 지금 상태에서 이미 잘 된 점

## 2.1 `model/` 상위 구조는 더 안 건드려도 된다

현재 `model/`은 아래 3축으로 정리돼 있다.

```text
model/
  data/
  net/
  engine/
```

이건 네가 처음 원했던 방향과 거의 정확히 맞다.

- `data/`: dataset, sampler, target encoder, transform
- `net/`: trunk, heads
- `engine/`: trainer, evaluator, loss, metrics, postprocess, spec

예전처럼 `encoding/`, `loading/`, `viz/`, `training/` 같은 미세 디렉토리가 난립하는 상태는 아니다.  
지금은 “어디 코드를 찾아야 하는가”가 충분히 자연스럽다.

## 2.2 `tools/od_bootstrap/` 상위 구조도 큰 방향은 맞다

현재 구조는 아래처럼 읽힌다.

```text
tools/od_bootstrap/
  source/
  teacher/
  build/
  cli.py
  presets.py
```

이건 현재 개발 철학과 잘 맞는다.

- `source/`: 원본 AIHUB + BDD100K 표준화
- `teacher/`: teacher train / eval / calibration
- `build/`: teacher dataset, exhaustive OD, final dataset, review/debug-vis

즉,

**원본 정리 → teacher 계층 → exhaustive/final materialization**

이라는 큰 책임 분리가 이미 자리를 잡았다.

## 2.3 config는 지금 정도면 충분히 괜찮다

이번 `main` 기준으로는 config surface가 이미 많이 정리됐다.

- `config/user_paths.yaml`
- `config/od_bootstrap_hyperparameters.yaml`
- `config/pv26_train_hyperparameters.yaml`

그리고 실제 코드는

- `common/user_config.py`
- `tools/od_bootstrap/presets.py`
- `tools/pv26_train_config.py`

를 통해 읽고 병합한다.

즉, 예전처럼 YAML 파일이 폴더 여러 개에 흩어져 있고, 뭘 어디서 바꾸는지 감이 안 오는 상태는 아니다.  
이번 조사 기준에서는 **config가 가장 큰 문제는 아니다.**

## 2.4 테스트 문화가 생각보다 좋다

정량적으로 보면 Python 파일 119개 중 테스트 파일이 37개다.  
그리고 `test/test_docs_sync.py`, `test/test_portability_runtime.py`, `test/od_bootstrap/*`처럼 단순 수치 테스트를 넘어 **문서 동기화, 런타임 상태, bootstrap 동작**까지 테스트하고 있다.

이건 꽤 좋은 신호다.

단순히 “학습 코드만 있고 테스트는 없다”가 아니라,

- 데이터 표준화
- bootstrap
- trainer
- evaluator
- docs sync
- launcher/runtime summary

까지 이미 테스트 레일이 깔려 있다.

## 2.5 `build-final-dataset`는 예전보다 훨씬 건강해졌다

`tools/od_bootstrap/build/final_dataset.py`는 이제 staging root를 만든 뒤 publish하는 방식으로 가고 있다.  
예전처럼 결과물 디렉토리를 중간에 덮어쓰기만 하는 형태보다 훨씬 낫다.

특히 아래 점이 좋다.

- staging root 생성
- publish marker 기록
- backup 후 rename publish
- 완료 후 marker 갱신

즉, 최종 artifact에 대해서는 이미 **재실행 안정성**을 어느 정도 신경 쓴 상태다.

이건 지금 저장소에서 꽤 중요한 개선점이다.

---

## 3. 정량 스냅샷

이번 조사에서 눈에 띈 수치를 먼저 적어두면 이렇다.

### 3.1 파일 규모

- 전체 Python 파일: **119개**
- 생산 코드: **82개**
- 테스트 코드: **37개**

### 3.2 라인 수가 큰 주요 파일

| 파일 | 대략 라인 수 | 해석 |
| --- | ---: | --- |
| `tools/check_env.py` | 1530 | TUI + env check + workspace scan + action launcher까지 한 파일에 들어있음 |
| `tools/run_pv26_train.py` | 1339 | meta-train preset, resume, preview, phase execution까지 한 파일 |
| `tools/od_bootstrap/source/bdd100k.py` | 1241 | BDD 전처리 전 과정을 한 파일에서 관리 |
| `tools/od_bootstrap/presets.py` | 891 | bootstrap 전체 preset composition root |
| `tools/od_bootstrap/teacher/calibrate.py` | 793 | calibration 전 과정이 응집돼 있음 |
| `model/engine/loss.py` | 783 | multitask loss 구현이 크게 응집돼 있음 |
| `tools/od_bootstrap/teacher/ultralytics_runner.py` | 762 | Ultralytics 적응 계층 + runtime helper 연결점 |
| `model/engine/_trainer_epochs.py` | 688 | epoch-level runtime/reporting 비중이 큼 |
| `tools/od_bootstrap/source/aihub.py` | 682 | AIHUB 전처리 orchestration monolith |
| `model/engine/trainer.py` | 632 | trainer façade + 내부 helper 재노출 |

이 숫자만 봐도, 남은 작업이 “폴더 추가/삭제”보다 **큰 orchestration 파일 분해** 쪽이라는 게 드러난다.

### 3.3 반복 helper 이름

생산 코드에서 이름이 반복되는 helper가 꽤 있다.

- `_write_json`: **8곳**
- `_now_iso`: **7곳**
- `_link_or_copy`: **3곳**
- `_load_json`: **4곳**
- `_make_anchor_grid`: **2곳**
- `_decode_anchor_relative_boxes`: **2곳**
- `_move_to_device`: **2곳**
- `_raw_batch_for_metrics`: **2곳**
- `_augment_lane_family_metrics`: **2곳**
- `_resample_points`: **2곳**
- `deep_merge_mappings` 계열: **2곳**

이 중 전부를 강제로 합칠 필요는 없지만,  
적어도 지금은 **공통으로 빼도 되는 것과 남겨도 되는 것이 섞여 있다.**

### 3.4 private cross-import

생산 코드에서 `_private_helper`를 다른 모듈이 import하는 패턴이 **121건** 나왔다.

특히 몰려 있는 곳은 아래다.

- `tools/od_bootstrap/source/aihub.py`
- `tools/od_bootstrap/source/bdd100k.py`
- `tools/od_bootstrap/source/aihub_*_worker.py`
- `tools/od_bootstrap/teacher/ultralytics_runner.py`
- `model/engine/_trainer_epochs.py`
- `model/engine/trainer.py`

이건 지금 저장소에서 가장 강한 구조적 냄새다.  
지금은 `_`가 “이 모듈 내부 전용”이 아니라, 사실상 **shared internal API**로 쓰이고 있다.

---

## 4. 지금 단계에서의 핵심 판단

## 4.1 더 이상 top-level 폴더 대공사는 필요 없다

이건 분명히 말할 수 있다.

지금 `common/`, `model/`, `tools/`, `test/` 아래 구조는 이미 충분히 읽힌다.  
`model`도 `data/net/engine`, `od_bootstrap`도 `source/teacher/build`로 정리돼 있어서,

**여기서 또 상위 디렉토리 구조를 갈아엎는 건 얻는 것보다 잃는 게 더 클 가능성**이 높다.

즉, 지금은

- 폴더 수를 더 줄이는 단계가 아니라
- 각 폴더 안의 **코드 경계와 재사용 지점**을 정리하는 단계다.

## 4.2 남은 문제는 “위치”보다 “경계”다

코드는 이미 대체로 제자리를 찾았다.  
그런데 그 안에서 아직 이런 문제가 남아 있다.

- 사설 helper가 실제로는 공유 helper처럼 쓰임
- 비슷한 IO/time/path helper가 여러 군데에 퍼져 있음
- orchestration 파일이 너무 커서 내부 흐름이 한 번에 들어가지 않음
- JSON manifest row가 대부분 느슨한 dict라 key drift에 취약함

즉, 이제는 “어느 폴더에 둘까”보다

- **어떤 것을 진짜 공용 API로 승격할지**
- **어떤 것은 진짜 private로 닫을지**
- **어떤 파일을 orchestration / domain logic / shared utility로 나눌지**

를 정하는 게 핵심이다.

---

## 5. 폴더별 상세 진단

## 5.1 `common/`

### 좋은 점

`common/`은 지금 비교적 절제되어 있다.

- `io.py`
- `paths.py`
- `user_config.py`
- `config_coercion.py`
- `pv26_schema.py`
- `overlay.py`
- `boxes.py`
- `scalars.py`

이 정도면 과도하게 커지지도 않았고, repo-wide 공용 기능이 이 안으로 모이는 구조도 이해 가능하다.

### 남은 개선 포인트

다만 지금 상태를 보면 `common/`이 이미 있는데도 실제로는 아래 helper들이 로컬 중복으로 남아 있다.

- `write_json`
- `append_jsonl`
- `now_iso`
- `timestamp_token`
- path resolve
- deep merge

즉, `common/`은 기반은 잘 깔았는데, **실제 재사용 흡수력이 아직 100%는 아니다.**

### 권장 방향

- repo 전체에서 재사용하는 건 `common/`으로 올리고
- bootstrap 내부에서만 재사용하는 건 `tools/od_bootstrap/.../shared_*.py`로 둔다

이 선만 명확히 하면 된다.

내 판단으로 `common/`은 지금 급하게 뜯을 곳은 아니고, **새 shared helper를 받아줄 착지점**으로 쓰는 게 좋다.

`2026-04-03` team wave 기준으로는 이 방향의 첫 단추를 이미 채웠다.
`deep_merge_mappings`는 `common.user_config` public helper로 승격됐고,
`tools/pv26_train_config.py`가 이를 재사용하도록 바뀌었다.
다만 `now_iso`, `timestamp_token`, `write_json`, path resolve`류는 아직 부분 중복이 남아 있다.

그 다음 wave에서는 `common.paths`에 `resolve_optional_path`, `resolve_latest_root`가 정착했고,
`common.io`에는 `write_jsonl`이 추가돼 `tools/od_bootstrap/build/`와
`tools/pv26_train_config.py` 일부 call-site가 공용 helper를 재사용하게 됐다.
즉, path/JSONL helper는 한 단계 더 정리됐고, 남은 큰 축은 time helper와 `append_jsonl`류다.

그 다음 wave에서는 `build/sweep.py`, `build/debug_vis.py`, `build/teacher_dataset.py`,
`source/prepare.py`가 semantics-compatible `common.io` helper를 직접 재사용하게 됐다.
즉, 공용 helper의 착륙 범위는 조금 더 넓어졌지만,
`append_jsonl`, 일부 time helper, 정책 차이가 있는 writer는 아직 로컬 구현이 남아 있다.

그 다음 wave에서는 `common/train_runtime.py`가 추가되면서
trainer/teacher runtime이 공유하는 duration formatting, device sync timing,
rolling timing summary, tensorboard writer/scalar helper가 한 레이어에 모이기 시작했다.
동시에 `model/engine/_trainer_io.py`도 `common.io`를 직접 재사용하도록 정리됐다.
다만 progress rendering helper와 repo-wide `append_jsonl` 흡수까지 끝난 것은 아니라서,
공용 helper 추출은 이제 마무리 단계만 남은 상태다.

그 다음 wave에서는 `common.io`에 `write_json_sorted`, `append_jsonl_sorted`, `write_jsonl_sorted`가 추가되면서
trainer history와 source shared IO가 더 이상 call-site마다 `sort_keys=True`를 다시 적지 않게 됐다.
즉, 3순위 helper 공통화는 이제 남은 local wrapper/정책 차이만 정리하면 되는 단계다.

---

## 5.2 `model/`

### 좋은 점

상위 구조는 좋다.  
특히 `data / net / engine` 3축은 더 안 건드려도 된다.

- `model/data/dataset.py`
- `model/data/sampler.py`
- `model/data/target_encoder.py`
- `model/net/trunk.py`
- `model/net/heads.py`
- `model/engine/trainer.py`
- `model/engine/evaluator.py`
- `model/engine/loss.py`
- `model/engine/postprocess.py`
- `model/engine/metrics.py`

이 배치는 충분히 자연스럽다.

### 남은 문제

문제는 `engine/` 내부에서 생긴다.

#### 1) trainer 내부가 `_trainer_*` 분할 + private 재노출 구조다

`model/engine/trainer.py`는 상위 façade 역할을 하면서도 아래 모듈들의 private helper를 대량으로 끌어와 alias로 다시 붙인다.

- `_trainer_checkpoint.py`
- `_trainer_epochs.py`
- `_trainer_fit.py`
- `_trainer_io.py`
- `_trainer_reporting.py`
- `_trainer_step.py`

이 구조는 의도는 이해된다.  
파일을 쪼개서 trainer를 관리하기 쉽게 만들려는 건 맞다.

그런데 실제로는

- `trainer.py`가 private helper를 많이 재노출하고
- 다른 모듈도 private helper를 서로 import하고
- public surface와 internal surface가 조금 섞여 있다

는 문제가 있다.

즉, 지금 `engine/`은 폴더 구조는 좋지만 **internal API 경계가 아직 흐리다.**

#### 2) 수학/geometry helper가 중복된다

대표적으로 아래가 있다.

- `_make_anchor_grid`: `loss.py`, `postprocess.py`
- `_decode_anchor_relative_boxes`: `loss.py`, `postprocess.py`
- `_move_to_device`: `trainer.py`, `evaluator.py`
- `_raw_batch_for_metrics`: `_trainer_epochs.py`, `evaluator.py`
- `_augment_lane_family_metrics`: `_trainer_epochs.py`, `evaluator.py`
- `_resample_points`: `target_encoder.py`, `metrics.py`

이 중 일부는 tensor/numpy backend가 달라서 완전 통합이 불편할 수 있다.  
그래도 최소한 아래 정도는 정리 가능하다.

- `model/engine/shared_geometry.py`
- `model/engine/shared_batch.py`
- `model/data/polyline_ops.py`

같은 공용 모듈을 두고, 진짜 같은 알고리즘은 같은 위치에 두는 편이 낫다.

### 결론

`model/`은 상위 디렉토리는 합격이다.  
남은 일은 **engine 내부 helper 경계 정리 + 수학 helper 공용화**다.

`2026-04-03` team wave 기준으로는 batch/helper 쪽이 먼저 정리됐다.
`model/engine/batch.py`가 추가되면서 `move_to_device`, raw batch unwrap,
lane family metric augmentation이 trainer/evaluator/_trainer_epochs 사이에서 공용화됐다.
그 다음 wave에서는 `model/engine/_det_geometry.py`가 추가되면서
anchor grid / anchor-relative decode도 `loss.py`, `postprocess.py`에서 공용화됐다.
그 다음 wave에서는 `model/engine/trainer.py`도 한 단계 더 정리됐다.
facade가 노출하던 private re-export를 테스트/호출자가 실제로 쓰는 compatibility shim 집합으로 줄이고,
내부 호출은 `_trainer_*` 모듈을 직접 참조하도록 바뀌었다.
그 다음 wave에서는 `_loss_spec.py`도 section-builder 형태로 다시 쪼개져
`build_loss_spec()`가 fresh nested mutable payload를 조립하는 구조가 더 읽히게 됐다.
그 다음 wave에서는 `model/engine/trainer_reporting.py`가 추가돼
trainer/test 호출부가 `_trainer_reporting.py` implementation detail 대신 public shared surface를 통하게 됐고,
누락돼 있던 progress/reporting helper도 `_trainer_reporting.py` 안으로 복구됐다.
즉, engine 쪽은 public shared module 승격이 실제로 한 번 들어갔고,
그 다음 wave에서는 `_trainer_epochs.py`의 loader/progress bookkeeping helper 일부가 `_trainer_progress.py`로 빠지면서
파일 크기가 다시 줄고 epoch loop가 orchestration 중심으로 조금 더 읽히게 됐다.
이제 남은 핵심은 더 넓은 public/internal surface 경계와 남은 internal shim 정리다.

---

## 5.3 `tools/check_env.py`

### 좋은 점

이건 user-facing UX 관점에서는 꽤 성공적이다.

- runtime 점검
- raw root 점검
- source prep / teacher / calibration / exhaustive / final dataset / pv26 상태 스캔
- 추천 액션 제시
- interactive launcher
- strict/json 모드

즉, **운영 허브 역할**은 충분히 잘 한다.

### 구조적 문제

하지만 파일 하나가 1530줄이다.  
지금은 돌아가도, 앞으로 launcher가 더 커지면 유지보수가 갑자기 어려워질 수 있다.

실제로 이 파일은 한곳에서 아래를 전부 한다.

- env check
- manifest parsing
- workspace scan
- action catalog
- blockers/advisory 계산
- rich TUI render
- input handling
- subprocess action launch
- resume candidate handling

이건 역할이 너무 많다.

### 권장 방향

이건 당장 급한 건 아니지만, 다음 정도로만 쪼개면 유지보수성이 확 올라간다.

```text
tools/
  check_env.py              # entrypoint
  check_env_scan.py         # workspace scan / summary load
  check_env_actions.py      # action catalog / blockers / advisory
  check_env_tui.py          # render / input / interactive loop
  check_env_launch.py       # subprocess launch / resume 실행
```

중요한 건 **기능을 바꾸는 게 아니라 파일 경계를 정리하는 것**이다.

### 우선순위 판단

- UX 자체: 이미 좋음
- 코드 구조: 아직 한 번 더 정리 가능
- 우선순위: **중간**

`2026-04-03` 후속 wave 기준으로는 여기서도 실제 진전이 있었다.
`tools/check_env.py`는 launch/compat facade로 남기고,
workspace scan은 `tools/check_env_scan.py`,
action catalog는 `tools/check_env_actions.py`,
rich rendering은 `tools/check_env_tui.py`로 분리됐다.
그 다음 wave에서는 input handling / subprocess launch / resume candidate 흐름도
`tools/check_env_launch.py`로 이동하면서 facade가 더 얇아졌다.
즉, 현재 남은 큰 축은 launch 계층 추가 분해보다 실제 런타임 시나리오 coverage와 유지보수 세부 정리다.

즉, 이건 “필수 수정”은 아니고 **나중에 덩치 줄이기용**이다.

---

## 5.4 `tools/run_pv26_train.py`

이 파일은 현재 저장소에서 가장 대표적인 orchestration monolith 중 하나다.

### 왜 큰가

지금 이 파일 안에는 아래가 다 있다.

- preset 조립
- scenario 로딩
- resume scenario 복구
- phase transition 제어
- dataloader 생성
- trainer/evaluator 생성
- preview overlay 생성
- manifest/summary 작성
- stage3 VRAM stress probe
- CLI entrypoint

즉, 실행 스크립트라기보다 **PV26 meta-train application layer 전체**가 들어 있다.

### 문제점

#### 1) import alias가 너무 많다

`tools/pv26_train_config.py`, `tools/pv26_train_artifacts.py`에서 함수들을 underscore alias로 많이 끌고 온다.

읽는 입장에서는

- 이게 이 파일 로컬 helper인지
- 외부 helper인지
- public API인지
- private implementation인지

가 한눈에 안 들어온다.

`2026-04-03` 후속 정리 기준으로 이 항목은 해소됐다.
현재 `run_pv26_train.py` 내부 호출은 `pv26_train_config.py`, `pv26_train_artifacts.py`의 public module API를 기준으로 읽히고,
기존 import surface 호환이 필요한 `_scenario_phase_defaults`, `_resolve_phase_selection`, `_validate_meta_train_scenario`,
`_phase_entry_is_completed`, `_recover_phase_entry_from_run_dir`만 로컬 compatibility wrapper로 남겨둔 상태다.

#### 2) `site.addsitedir(REPO_ROOT)`는 실행은 되지만 구조 냄새다

스크립트를 직접 실행하기 위해 repo root를 path에 넣는 방식은 실용적이긴 하다.  
그런데 이건 결국 **packaging/entrypoint 경계가 아직 느슨하다**는 뜻이기도 하다.

지금 당장 문제는 아니지만, 이 방식은 나중에 import 문제를 가리기도 쉽다.

#### 3) meta-train preset assembly와 runtime execution이 한 파일에 있다

이건 분리 가치가 크다.

### 권장 분해 방향

```text
tools/
  run_pv26_train.py         # stable thin facade / CLI entrypoint
  pv26_train_scenario.py    # preset assembly + scenario/resume loading
  pv26_train_runtime.py     # phase runtime orchestration
  pv26_train_stress.py      # stage3 VRAM stress probe / summary
```

이렇게 나누면 읽기 좋아질 뿐 아니라, `test/test_run_pv26_train.py`가 잡고 있는
public facade(import surface)와 내부 orchestration 경계를 분리해 유지하기도 쉬워진다.

특히 이번 2b 분해에서는 아래 경계를 먼저 고정하는 편이 안전하다.

- facade에 남길 것
  - `load_meta_train_scenario()`
  - `load_meta_train_resume_scenario()`
  - `run_stage3_vram_stress()`
  - `run_meta_train_scenario()`
  - `main()`
- `pv26_train_scenario.py`로 옮길 것
  - preset lookup / scenario validation
  - `meta_manifest.json` snapshot restore
  - legacy resume compatibility check
- `pv26_train_stress.py`로 옮길 것
  - stage 3 batch/iter override config
  - probe execution
  - OOM summary / recommendation payload
- 회귀 gate
  - `test/test_run_pv26_train.py`
  - `test/test_portability_runtime.py`
  - `test/test_docs_sync.py`

### 우선순위 판단

이건 `check_env.py`보다 더 우선순위가 높다.  
왜냐하면 실제 학습 실행/재개/preview/manifest 로직의 중심이기 때문이다.

내 판단상 **가장 먼저 줄여야 할 큰 파일 중 하나**다.

---

## 5.5 `tools/od_bootstrap/source/`

여기가 현재 저장소에서 **가장 큰 정리 여지**가 남아 있는 구간이다.

### 지금 좋은 점

- AIHUB worker 분리 (`aihub_lane_worker.py`, `aihub_traffic_worker.py`, `aihub_obstacle_worker.py`)
- raw helper 분리 (`raw_common.py`)
- source metadata/report 분리 (`aihub_reports.py`, `aihub_source_meta.py`)
- prepare entrypoint 분리 (`prepare.py`)

즉, 예전보다 훨씬 낫다.

### 그런데 여전히 남은 가장 큰 문제

#### 1) `bdd100k.py`가 `aihub.py`의 private helper를 많이 import한다

이건 이번 조사에서 제일 눈에 띄는 구조적 냄새였다.

`tools/od_bootstrap/source/bdd100k.py`는 아래 같은 것을 `aihub.py`에서 가져다 쓴다.

- `LiveLogger`
- `_iter_task_chunks`
- `_parallel_chunk_size`
- `_default_workers`
- `_generate_debug_vis`
- `_link_or_copy`
- `_write_json`
- `_write_text`
- `_bbox_to_yolo_line`
- `_counter_to_dict`

즉, 겉으로는 `source/aihub.py`와 `source/bdd100k.py`가 병렬인 형제 모듈인데,  
실제로는 BDD 쪽이 AIHUB implementation에 깊게 의존하고 있다.

이건 구조적으로 좋지 않다.

왜 문제냐면,

- AIHUB 쪽 리팩토링이 BDD 쪽까지 깨뜨릴 수 있고
- private helper라 이름상으로는 건드려도 될 것처럼 보이는데 실제론 shared API 역할을 하고 있고
- source adapter 둘이 서로 독립적이지 않다

는 뜻이기 때문이다.

#### 2) worker들이 `raw_common.py`, `aihub_worker_common.py`의 private helper를 대량 import한다

이것도 비슷한 문제다.

- `_base_scene`
- `_counter_to_dict`
- `_sample_id`
- `_extract_annotations`
- `_safe_slug`
- `_normalize_text`
- `_load_json`

같은 것들이 여러 worker에서 반복 import된다.

이 경우 해결 방향은 단순하다.

**private helper를 shared helper로 승격**하면 된다.

예를 들면,

```text
tools/od_bootstrap/source/
  shared_io.py
  shared_parallel.py
  shared_scene.py
  shared_summary.py
```

같은 식으로 public shared module을 두고,

- underscore helper는 진짜 그 파일 내부에서만 쓰고
- cross-module reuse가 필요한 것은 underscore를 떼서 shared module로 올리는 편이 훨씬 낫다

#### 3) `source/types.py`가 pure types가 아니다

`source/types.py`는 이름상 타입 모듈인데, 실제로는 `aihub.py`와 `bdd100k.py`에서 기본 경로 상수를 import해 온다.

즉, 이 모듈은 타입 선언만 하는 곳이 아니라 **implementation defaults에 의존하는 곳**이다.

이건 나중에 import-time coupling을 만들기 쉽다.

### 권장 방향

`source/`는 아래처럼 정리하는 게 좋다.

```text
source/
  aihub_pipeline.py
  bdd_pipeline.py
  raw_common.py
  shared_io.py
  shared_parallel.py
  shared_scene.py
  shared_summary.py
  aihub_lane_worker.py
  aihub_traffic_worker.py
  aihub_obstacle_worker.py
  prepare.py
  types.py
  constants.py
```

`2026-04-03` team wave 기준으로는 이 방향의 일부가 실제 코드로 들어갔다.
`shared_resume.py`가 existing-output summary skeleton을 담당하고,
`shared_source_meta.py`가 BDD README/tree/source inventory render를 담당한다.
즉, source 공통골격 중 resume/meta 출력은 public shared API로 이동했고,
남은 큰 잔여물은 debug-vis manifest write 쪽이다.

핵심은 “더 쪼개자”가 아니라,

- **서로 끌어다 쓰는 private helper를 shared module로 빼자**
- `aihub.py`/`bdd100k.py`는 pipeline coordinator로만 남기자

는 것이다.

### 우선순위 판단

이번 조사 기준으로 **가장 우선순위가 높은 코드 정리 대상**은 여기다.

---

## 5.6 `tools/od_bootstrap/teacher/`

### 좋은 점

여기는 이미 구조가 꽤 괜찮다.

- `train.py`
- `eval.py`
- `calibrate.py`
- `policy.py`
- `data_yaml.py`
- `runtime_*`
- `*_types.py`

즉, teacher lifecycle이 분리돼 있다.

### 남은 문제

#### 1) `ultralytics_runner.py`가 너무 많은 runtime helper를 감싼다

이 파일은 실제로는 아래를 다 엮는다.

- dataloader kwargs
- progress helper
- tensorboard helper
- resume helper
- artifact refresh helper
- callback builder
- trainer subclass

그리고 이 과정에서 `runtime_progress.py`, `runtime_tensorboard.py`, `runtime_resume.py`, `runtime_artifacts.py`의 private helper를 다수 alias로 끌어온다.

즉, 여기도 사실상

**private helper mesh + orchestration monolith**

구조다.

#### 2) `build_teacher_runtime_callbacks()`의 dependency injection 인자가 너무 많다

이 함수는 테스트 가능성을 높이려는 의도는 좋다.  
그런데 지금은 callable 인자가 너무 많아서 읽는 사람이 오히려 구조를 추적하기 어려워졌다.

이럴 때는 함수 인자를 13개 넘게 늘리기보다,

- `TeacherRuntimeSupport` dataclass
- 또는 `TeacherRuntimeDeps` 객체

하나로 묶는 편이 보통 더 읽기 쉽다.

#### 3) teacher runtime과 PV26 trainer runtime이 비슷한 helper를 따로 가진다

중복되는 축은 아래다.

- `format_duration`
- `sync_timing_device`
- tensorboard writer build
- tensorboard scalar write
- timing profile 요약
- progress rendering

teacher와 PV26 trainer가 프레임워크가 달라 완전 통합은 어렵다.  
하지만 최소한 **공통 runtime helper 계층**은 조금 더 정리 가능하다.

### 결론

teacher 폴더는 상위 구조는 좋다.  
남은 작업은

- `ultralytics_runner.py` 덩치 줄이기
- runtime helper/public shared API 정리
- dependency injection을 객체 단위로 묶기

정도다.

`2026-04-03` 후속 wave 기준으로는 runtime helper/public API 정리의 첫 단계가 들어갔다.
`ultralytics_runner.py`는 이제 `runtime_progress`, `runtime_tensorboard`,
`runtime_resume`, `runtime_artifacts`의 public/shared helper 이름을 우선 사용하고,
underscore alias는 compatibility shim으로만 남겨둔 상태다.
그 다음 wave에서는 `TeacherRuntimeSupport`가 `build_teacher_runtime_callbacks()`의
callback dependency surface를 한 객체로 묶어줬다.
그 다음 wave에서는 `_make_teacher_trainer()`도 resume restore,
extended-LR scheduler build, runtime-state setup helper를 바깥으로 빼내며 조금 더 읽히는 구조가 됐다.
즉, alias mesh / callback DI / trainer helper extraction은 한 단계 더 정리됐고,
그 다음 wave에서는 `common/train_runtime.py`를 매개로 teacher runtime 쪽도
`runtime_progress.py`, `runtime_tensorboard.py`, `calibrate.py`가 duration/timing/tensorboard helper를 공유하게 됐다.
남은 핵심은 calibrate 흐름의 추가 축소와 runner 본체 덩치다.

그 다음 wave에서는 `runtime_artifacts.py`가 teacher train summary / latest artifact publication까지 맡게 되면서
`ultralytics_runner.py`가 artifact summary write를 직접 잡지 않게 됐다.
즉, runner 분해는 이제 callback/dataloader/trainer subclass 쪽 큰 덩치를 줄이는 단계가 남아 있다.

우선순위는 `source/`보다는 약간 낮지만, **분명히 손볼 가치가 있다.**

---

## 5.7 `tools/od_bootstrap/build/`

### 좋은 점

여기는 전체적으로 꽤 괜찮다.

- `teacher_dataset.py`
- `exhaustive_od.py`
- `final_dataset.py`
- `debug_vis.py`
- `image_list.py`
- `review.py`
- `checkpoint_audit.py`
- `artifacts.py`

역할 분리가 비교적 선명하다.

특히 좋은 건,

- `image_list.py`가 sample UID 계약을 잡아주고
- `artifacts.py`가 bootstrap run manifest를 정리하고
- `final_dataset.py`가 publish contract를 가지며
- `debug_vis.py`가 별도 툴로 분리돼 있다는 점이다.

### 남은 문제

#### 1) manifest row가 여전히 대부분 느슨한 dict다

예를 들면 아래 계층에서 JSON row dict를 많이 돌린다.

- `debug_vis.py`
- `teacher_dataset.py`
- `exhaustive_od.py`
- `final_dataset.py`
- `sweep.py`

이건 파이프라인이 커질수록 key 이름이 조금씩 어긋날 위험이 있다.

지금 단계에서는 dataclass보다도 **TypedDict**가 잘 맞는다.

예를 들면,

- `TeacherDatasetManifestRow`
- `ExhaustiveSampleRow`
- `FinalDatasetSampleRow`
- `DebugVisItemRow`
- `TeacherPredictionRow`

정도만 둬도 IDE 추적성과 리팩토링 안전성이 훨씬 좋아진다.

그 다음 wave에서는 그 방향의 첫 실제 적용이 들어왔다.
`debug_vis.py`와 `teacher_dataset.py`가 `TypedDict` 기반 manifest/item row를 쓰기 시작하면서
적어도 debug/teacher dataset 경로는 loose dict 면적을 줄였다.
다만 `exhaustive_od.py`, `final_dataset.py`, `sweep.py`까지 한 번에 다 끝난 것은 아니다.

#### 2) IO helper가 아직 중복된다

- `_write_json`
- `_link_or_copy`
- `_default_io_workers`
- `resolve_latest_root`

같은 것들이 build 내부에도 반복된다.

다만 여기서는 주의가 필요하다.

- `common.io.link_or_copy()`는 symlink fallback 정책이고
- `teacher_dataset.py` / `final_dataset.py`는 hardlink/copy 정책이고
- `final_dataset.py`는 overwrite 금지 정책이 있고
- 어떤 곳은 existing이면 skip 정책이 있다

즉, 이름은 같아도 정책은 다르다.

현재 local `link_or_copy` surface를 정확히 적으면 아래 파일들이다.

- `common/io.py`
- `source/shared_io.py`
- `source/aihub.py`
- `build/teacher_dataset.py`
- `build/final_dataset.py`
- `teacher/runtime_artifacts.py`
- `teacher/data_yaml.py`

이 중에서도 차이가 크다.

- `common/io.py`는 기존 target을 지우고 symlink fallback을 시도한다.
- `source/shared_io.py` / `source/aihub.py`는 hardlink/copy + existing skip 계약이다. (`source/shared_io.py`는 parent-dir 준비와 JSON read는 `common.io`를 재사용한다.)
- `build/teacher_dataset.py` / `build/final_dataset.py`는 copy_images 여부와 overwrite 금지 publish 흐름을 함께 가진다.
- `teacher/runtime_artifacts.py`는 latest alias를 refresh하기 위해 remove + relink를 허용한다.
- `teacher/data_yaml.py`는 파일 하나가 아니라 tree staging이라 symlink/copytree 축이다.

그래서 무조건 하나로 합치기보다,

- low-level atomic/json helper는 공통화하고
- link policy는 각 빌더가 갖는 방식

이 더 낫다.

### 결론

`build/`은 구조 자체는 좋다.  
여기는 대공사보다

- manifest row typing
- low-level IO helper 정리
- naming consistency

정도면 충분하다.

`2026-04-03` 후속 wave 기준으로는 low-level IO/path helper 정리가 한 차례 더 진행됐다.
`common.paths.resolve_latest_root`, `resolve_optional_path`,
`common.io.write_jsonl`을 build 쪽에서 재사용하도록 맞췄고,
`final_dataset.py`의 overwrite/publish처럼 정책 차이가 큰 helper는 그대로 로컬에 남겨뒀다.
즉, 정책이 같은 helper만 공용화하고 나머지는 builder-local로 남기는 방향이 실제 코드에 반영됐다.

source 쪽에서도 비슷한 방향으로 한 단계 더 나갔다.
`shared_debug.build_debug_vis_manifest()`와 관련 `TypedDict`가 들어오면서
debug-vis index payload가 더 이상 loose dict에만 기대지 않게 됐고,
AIHUB/BDD caller가 같은 manifest contract를 재사용하게 됐다.

그 다음 wave에서는 `check_env_scan.py`, `tools/od_bootstrap/cli.py`,
`test_final_dataset.py`, `test_sweep_runner.py`, `test_run_generate_debug_vis.py` 등이
`exhaustive_od.py`, `final_dataset.py`의 summary/manifest filename constant를 재사용하도록 정리됐다.
즉, build 쪽은 loose dict typing 확장에 더해 summary/publish naming consistency도 한 단계 더 정리된 상태다.

---

## 6. 지금 남아 있는 재사용/중복 개선 포인트

이번 조사 기준으로, 실제로 “지금 당장 손대면 효과 큰 것”만 추리면 아래 정도다.

## 6.1 가장 먼저 뽑아낼 shared helper

### A. 시간/JSON/path helper

후보:

- `now_iso`
- `timestamp_token`
- `write_json`
- `append_jsonl`
- `resolve_latest_root`
- `resolve_optional_path`
- `deep_merge_mappings`

이건 지금 정말 여러 곳에 있다.  
최소한 아래 둘 중 하나는 필요하다.

- `common/io.py`, `common/paths.py`, `common/config_merge.py` 확장
- 또는 `tools/od_bootstrap/shared_io.py`, `shared_paths.py` 추가

`2026-04-03` rank-3 audit 기준으로 production definition 잔량은
`now_iso` 3곳, `timestamp_token` 2곳, `write_json` 5곳, `append_jsonl` 3곳이다.
다만 이 숫자를 그대로 “다 common으로 몰아넣으면 된다”로 읽으면 안 된다.

- `model/engine/_trainer_io.py`
- `tools/od_bootstrap/teacher/runtime_progress.py`
- `tools/od_bootstrap/source/shared_io.py`

이 셋은 이미 common helper 위에 얹힌 thin compatibility shim에 가깝다.
`2026-04-03` wave 9 follow-up에서는 이 셋이 local wrapper body를 더 줄이고
direct alias / re-export(`trainer_io`, `runtime_progress`)와 common `read_json` / `ensure_parent_dir`
재사용(`shared_io`)만 남도록 더 얇아졌다.
반대로 아래 셋은 아직 로컬 정책/계약 차이가 남아 있다.

- `source/raw_common.py`의 UTC timestamp contract
- `teacher/calibrate.py`의 `default=str` JSON 직렬화
- `build/final_dataset.py`의 overwrite 금지 publish semantics

즉, rank-3에서 해야 할 일은 “개수를 0으로 만들기”보다
shim과 policy-sensitive wrapper를 분리해서 문서/코드 기준선을 고정하는 쪽에 더 가깝다.

### B. source pipeline 공통골격

후보:

- `LiveLogger`
- parallel chunking
- existing output summary skeleton
- debug-vis manifest write
- README/tree markdown render

특히 **BDD가 AIHUB private helper를 import하는 상황**은 꼭 정리하는 게 좋다.

### C. engine 공통 수학/helper

후보:

- anchor grid 생성
- anchor-relative box decode
- raw batch unwrap
- lane family metric augmentation
- move_to_device

### D. runtime monitor helper

teacher와 PV26 trainer 양쪽에서 쓰는 축:

- duration formatting
- tensorboard writer build
- scalar flatten write
- device sync timing helper
- rolling timing summary

여긴 `common/train_runtime.py` 같은 별도 공용 레이어가 가능하다.

---

## 6.2 지금은 남겨도 되는 중복

반대로, 중복처럼 보여도 억지로 합치지 않는 게 좋은 것도 있다.

### A. `target_encoder._resample_points` vs `metrics._resample_points`

이건 로직은 비슷해 보여도

- 하나는 torch tensor encoder용
- 하나는 numpy 기반 metric 계산용

이라 구현을 완전히 합치면 오히려 코드가 더 지저분해질 수 있다.

이 경우엔 “공용 계약 + 테스트 공유”까지만 하고, 구현은 나눠도 된다.

### B. source worker별 annotation 파싱

- lane
- obstacle
- traffic

이 셋은 데이터 의미가 다르다.  
여기는 지나친 공통화보다 worker별 명시성이 더 중요하다.

즉, **공통화는 skeleton/helper 수준까지만** 가고, 실제 annotation 의미 해석은 분리 유지가 맞다.

### C. teacher runtime과 PV26 runtime의 전체 추상화 통합

둘 다 학습이지만,

- teacher는 Ultralytics trainer 적응 계층
- PV26는 직접 구현 trainer

라서 전체 runner abstraction을 합치는 건 오히려 손해일 수 있다.

여기는 **tensorboard/progress/timing helper만 통합**하는 선이 적절하다.

---

## 7. 지금 보이는 위험한 점

## 7.1 `_private helper`가 사실상 shared API가 된 상태

이게 제일 크다.

원래 `_foo()`는 “이 파일 내부용”이라는 신호인데, 지금은 다른 모듈들이 많이 가져다 쓴다.  
이 상태가 지속되면 나중에 refactor할 때

- 이름만 바꿨는데 다른 모듈이 깨지고
- 파일을 나눴는데 import 경로가 연쇄 붕괴하고
- IDE에서 public/private 경계 추적이 어려워진다

즉, 이건 앞으로 코드베이스가 더 커질수록 위험해진다.

## 7.2 느슨한 dict 계약

manifest row, sample row, prediction row, summary row 대부분이 dict라서,  
다음 같은 버그가 런타임까지 숨어들기 쉽다.

- key typo
- optional field 누락
- value 타입 drift
- 경로/string 혼합

지금은 테스트가 있어서 버틸 수 있지만, 코드가 더 커지면 이 방식은 한계가 온다.

## 7.3 큰 orchestration 함수는 회귀 위험이 높다

대표적으로 아래가 그렇다.

- `tools/od_bootstrap/source/aihub.py: run_standardization()`
- `tools/od_bootstrap/source/bdd100k.py: run_standardization()`
- `tools/od_bootstrap/teacher/calibrate.py: calibrate_class_policy_scenario()`
- `tools/od_bootstrap/teacher/ultralytics_runner.py: _make_teacher_trainer()`
- `tools/run_pv26_train.py: run_meta_train_scenario()`
- `model/engine/_loss_spec.py: build_loss_spec()`

이런 함수는 잘 돌아갈 때는 좋다.  
그런데 수정점이 늘면 회귀가 한 번에 커진다.

## 7.4 import-time coupling

특히 아래는 조심해야 한다.

- `source/types.py`가 implementation defaults를 import함
- `bdd100k.py`가 `aihub.py`의 internal helper를 import함
- `run_pv26_train.py`가 repo root를 path에 직접 추가함

이건 당장 에러가 난다는 뜻은 아니다.  
하지만 “구조상 조용히 불안정한 지점”이라는 뜻이다.

---

## 8. 비추천하는 리팩토링

지금 시점에서 오히려 하지 않는 게 좋은 것도 적어두는 편이 낫다.

## 8.1 top-level 폴더를 또 뒤집는 것

지금 구조는 충분히 좋다.  
여기서 `model/`이나 `tools/od_bootstrap/` 상위 폴더를 또 갈아엎는 건 비추천이다.

## 8.2 모든 중복을 하나로 합치려는 것

이건 오히려 더 나빠질 수 있다.

특히

- torch/numpy backend가 다르거나
- teacher/PV26 프레임워크가 다르거나
- source별 annotation semantics가 다르면

중복 일부는 유지하는 게 맞다.

## 8.3 `common/`에 다 몰아넣는 것

repo-wide truly common만 `common/`에 두고,  
bootstrap 내부에서만 쓰는 건 bootstrap shared 모듈로 두는 게 맞다.

지금 단계에서 `common/`을 과대 팽창시키는 건 별로다.

## 8.4 config 체계를 또 흔드는 것

이번 조사 기준으로 config는 이미 충분히 정리된 축이다.  
여길 또 뜯는 것보다 source/engine/orchestration 정리가 훨씬 가치가 크다.

---

## 9. 추천 수정 우선순위

## 9.1 1순위: `source/`의 private cross-import 정리

가장 먼저 할 만한 일은 이거다.

- `bdd100k.py -> aihub.py` private import 제거
- worker들이 쓰는 helper를 `shared_*.py`로 승격
- `types.py`를 implementation dependency에서 분리

이건 재사용성, 경계 명확성, 유지보수 안전성을 한 번에 올린다.

## 9.2 2순위: `tools/run_pv26_train.py` 분해

- preset assembly
- resume recovery
- preview generation
- phase execution
- CLI entrypoint

를 나누는 것만으로도 코드가 크게 좋아질 가능성이 높다.

## 9.3 3순위: 공통 IO/path/config merge helper 정리

- `_write_json`
- `_now_iso`
- `deep_merge_mappings`
- `resolve_optional_path`
- `resolve_latest_root`

이건 손대기 쉽고 효과도 바로 보인다.

## 9.4 4순위: manifest row TypedDict 도입

이건 런타임 동작을 크게 바꾸지 않으면서도,  
코드 추적성과 안전성을 많이 올릴 수 있다.

## 9.5 5순위: `check_env.py`, `ultralytics_runner.py` 파일 경계 정리

이건 급하진 않지만, 코드 수명이 길어질수록 미리 해두는 편이 좋다.

---

## 10. 최종 결론

내 결론은 명확하다.

### 1) 지금 저장소는 “얼추 완성”이라는 표현이 꽤 맞다

특히 상위 폴더 구조는 이미 충분히 좋다.

- `model/` 구조 좋음
- `od_bootstrap/` 상위 구조 좋음
- config surface 괜찮음
- tests 좋음
- launcher UX 좋음

즉, **또 한 번의 대대적 구조 개편은 필요 없어 보인다.**

### 2) 하지만 코드 내부 정리는 아직 남아 있다

남은 핵심은 아래 네 가지다.

- private helper mesh 제거
- 공용 helper 추출
- orchestration monolith 분해
- typed contract 강화

이건 “보이기 좋은 리팩토링”이 아니라,  
실제로 앞으로 수정할 때 덜 깨지고 덜 헷갈리게 만드는 작업이다.

### 3) 가장 먼저 손댈 곳은 `tools/od_bootstrap/source/`와 `tools/run_pv26_train.py`

초기 우선순위는 이 둘이 맞았고, `2026-04-03` cleanliness wave들에서 실제로 가장 많이 정리된 축도 이 둘이었다.
현재 시점에서 남은 다음 타깃을 다시 고르면 순서는 조금 바뀐다.

- `common/` + `common/train_runtime.py` 마감 (`append_jsonl`, time helper, progress helper residue)
- `model/engine/` public/internal surface 추가 정리
- `tools/od_bootstrap/teacher/ultralytics_runner.py` bulk 축소
- build/source summary row static contract 마감

즉, 초반의 큰 경계 정리는 꽤 진행됐고, 지금은 shared helper와 internal surface를 마감하는 단계다.

### 4) 요약하면

지금은 “폴더를 또 바꾸자”가 아니라,

**공용부를 올바르게 승격하고, 사설 구현이 경계 밖으로 새는 걸 막는 단계**다.

그 단계만 지나면 이 저장소는 꽤 오래 버티는 형태가 될 가능성이 높다.

---

## 11. 한 장 요약

- 상위 구조는 이미 충분히 좋다. 더 큰 폴더 개편은 비추천.
- 지금 남은 건 **경계 정리**다.
- 가장 큰 냄새는 **private cross-import**다.
- 초기 최우선 타깃이던 `tools/od_bootstrap/source/`와 `tools/run_pv26_train.py`는 큰 경계 정리가 이미 진행됐다.
- 지금 가장 효과 큰 다음 타깃은 3순위 빈칸 마감이다: `common/` helper commonization, local wrapper 흡수, common vs bootstrap-shared 경계 기준 고정이 먼저다.
- 그 다음이 `model/engine/` internal surface 정리다.
- `common/`은 새 shared helper를 받아주는 방향으로 확장하면 된다.
- `build/`는 구조는 좋고, manifest typing + summary/publish naming contract 마감 정도면 충분하다.
- teacher runtime과 PV26 trainer runtime은 전체 통합보다 **공통 runtime helper만 공유**하는 편이 맞다.
- 선택 테스트 30개와 compileall 기준으로는 현재 코드 건강도는 꽤 좋은 편이다.
