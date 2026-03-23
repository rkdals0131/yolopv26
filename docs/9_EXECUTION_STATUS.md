# Execution Status

## 운영 규칙

- 이 문서는 구현 중 항상 갱신하는 live tracker다.
- 코드 변경 전에 해당 phase와 next action을 맞춘다.
- 테스트가 끝나면 결과와 날짜를 기록한다.

## 현재 기준

- 날짜: `2026-03-24`
- phase: `phase 16 portability-and-validation-hardening`
- current focus: `portability/import boundary/validation runtime hardening 완료, full-dataset 본학습 전 pilot 실행 판단 단계`

## 완료된 항목

- [x] old pivot 문서 세트 제거
- [x] docs 전면 개정
- [x] `model/` 루트로 패키지 정리
- [x] `aihub_common.py + aihub_standardize.py` 구조 정리
- [x] AIHUB standardization pipeline 구현
- [x] source README 자동 생성
- [x] source inventory / conversion report 구현
- [x] debug overlay 구현
- [x] loss spec 코드/문서 반영
- [x] sample contract 문서 고정
- [x] transform contract 문서 고정
- [x] detector-TL attr binding 규약 고정
- [x] query count 문서/코드 sync 정리
- [x] docs 상대경로 링크 정리
- [x] `N_gt_det / Q_det` 용어 분리
- [x] dataset raw / vehicle camera reference / network input 용어 분리
- [x] task별 invalid 처리 우선순위 고정
- [x] raw output / export bundle 계층 분리
- [x] BDD100K det-only canonical pipeline 구현
- [x] canonical dataset loader runtime 구현
- [x] shared online letterbox transform 구현
- [x] ragged sample collate 구현
- [x] target encoder runtime 구현
- [x] Ultralytics YOLO26 trunk adapter baseline 구현
- [x] official `yolo26n.pt` real-load smoke 확인
- [x] PV26 custom heads skeleton 구현
- [x] trunk + custom heads forward smoke 확인
- [x] multitask loss runtime 구현
- [x] trunk + custom heads + loss backward smoke 확인
- [x] trunk adapter default trainable 상태 regression 고정
- [x] trainer skeleton runtime 구현
- [x] evaluator skeleton runtime 구현
- [x] tiny overfit smoke command 구현
- [x] tiny overfit loss 감소 확인
- [x] final detector assignment integration
- [x] lane family Hungarian matching integration
- [x] dataset-balanced sampler helper 구현
- [x] trainer checkpoint save/load 구현
- [x] trainer history summary / JSONL logging 구현
- [x] inference postprocess runtime 구현
- [x] evaluator predict-batch runtime 구현
- [x] batch-level eval metric runtime 구현
- [x] full-epoch trainer wiring 구현
- [x] epoch-level val loop 구현
- [x] best / last checkpoint write 구현
- [x] fit smoke command 구현
- [x] AIHUB/BDD resume scan 구현
- [x] AIHUB/BDD failure manifest 구현
- [x] AIHUB/BDD QA summary 구현
- [x] trainer AMP / grad accumulation / grad clip 구현
- [x] trainer auto resume 구현
- [x] trainer non-finite / OOM guard 구현
- [x] pilot training command 구현
- [x] repo-relative + env override dataset root 정리
- [x] preprocess image-size probing에 PIL 우선 fallback 추가
- [x] eval / training / trunk lazy import 정리
- [x] torchvision NMS pure PyTorch fallback 구현
- [x] validation sequential eval loader 구현
- [x] evaluator single-forward validation path 구현
- [x] env preflight command 구현
- [x] ultralytics-missing 환경용 runtime skip test 정리
- [x] unit test 통과
- [x] real-data smoke 통과
- [x] git commit 생성

## 다음 작업

- [ ] full-dataset 전처리 실제 실행 계획 확정
- [ ] pilot subset 본학습과 metric 해석
- [ ] preflight 결과를 기준으로 full-train 환경 lock 파일 정리
- [ ] export / ROS 정교화

## 최근 검증

- [x] `python3 -m unittest discover -s test -v`
- [x] `python3 -m model.preprocess.aihub_standardize --workers 2 --max-samples-per-dataset 4 --debug-vis-count 2`
- [x] `python3 -m model.preprocess.bdd100k_standardize --workers 8 --max-samples-per-split 2 --debug-vis-count 2`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_loader.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_target_encoder.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_yolo26_trunk.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_heads.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_trunk_features.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_loss_runtime.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_balanced_sampler.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_trainer.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_evaluator.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_eval_metrics.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_postprocess.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_tiny_overfit.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_aihub_standardize.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_bdd100k_standardize.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_portability_runtime.py' -v`
- [x] `python3 tools/run_yolo26_trunk_smoke.py`
- [x] `python3 tools/check_env.py`
- [x] `python3 tools/run_pv26_tiny_overfit_smoke.py --steps 4`
- [x] `python3 tools/run_pv26_fit_smoke.py --epochs 1 --train-batches 1 --val-batches 1`
- [x] `python3 tools/run_pv26_pilot_train.py --epochs 1 --train-batches 1 --val-batches 1 --run-dir /tmp/pv26_pilot_smoke`
- [x] `python3 -m model.preprocess.aihub_standardize --workers 1 --max-samples-per-dataset 1 --debug-vis-count 1`
- [x] `python3 -m model.preprocess.bdd100k_standardize --workers 1 --max-samples-per-split 1 --debug-vis-count 1`
- [x] detector assignment 통합 후 targeted tests 재통과
- [x] lane Hungarian 통합 후 targeted tests 재통과
- [x] docs sync test 추가 후 `python3 -m unittest discover -s test -v` 재통과
- [x] loss runtime 추가 후 `python3 -m unittest discover -s test -v` 재통과

## 최근 결정

- trunk는 official pretrained `yolo26n` 기준
- head는 PV26 custom implementation
- raw-space standardized dataset 유지
- `800x608` transform은 loader 단계 온라인 적용
- lane 학습/추론 계약은 AIHUB 기준 유지
- loader sample contract는 `image / det_targets / tl_attr_targets / lane_targets / source_mask / valid_mask / meta`로 고정
- lane/stop/crosswalk query count는 `12 / 6 / 4`로 고정
- TL attr supervision은 detector assignment 결과를 재사용
- `N_gt_det`는 GT row count, `Q_det`는 detector prediction slot count로 고정
- standardized dataset loader는 variable dataset raw에서 `800x608`으로 직접 transform한다
- loader 전에 BDD100K도 canonical standardization 레이어로 맞춘다
- BDD100K는 `det only` source로 쓰고 `trafficLightColor`는 scene hint로만 보존한다
- loader collate는 image만 stack하고 ragged target은 list 형태로 유지한다
- target encoder는 `det padded GT + TL GT bits/mask + lane family fixed query tensor`를 만든다
- trunk adapter는 `ultralytics>=8.4.0` 가드, detect-head 분리, partial state load helper를 기준선으로 둔다
- current smoke env is `ultralytics 8.4.25 + torch 2.10.0 + torchvision 0.25.0 + numpy 1.26.4`
- current custom heads skeleton uses `P3/P4/P5 = 64/128/256 channels` and `Q_det=9975` at `800x608`
- current trunk feature extractor returns detect-source pyramid directly from Ultralytics graph using indices `[16, 19, 22]`
- current detector loss runtime uses task-aligned assignment on real trunk/head outputs
- current synthetic `q_det != canonical` tests keep a `prefix positive fallback`
- current lane/stop-line/crosswalk loss runtime uses Hungarian matching against valid GT rows
- build_yolo26n_trunk returns trunk parameters with `requires_grad=True` by default
- current trainer skeleton can run `encoded batch -> backward -> optimizer.step` on real trunk+heads
- current trainer runtime includes balanced sampler helper, checkpoint save/load, and history JSONL logging
- current trainer runtime also includes epoch fit loop, val loop, best / last checkpoint write, and run summary output
- current trainer runtime also includes AMP, grad accumulation, grad clip, auto resume, and non-finite / OOM guard
- current evaluator runtime returns batch loss summary / GT count summary and supports postprocessed prediction bundles
- current evaluator runtime also returns batch-level detector AP50/precision/recall, TL bit F1/combo accuracy, and lane family matching metrics
- current tiny overfit smoke uses `stage_1_frozen_trunk_warmup`, mixed canonical train batch, and confirms best loss < first loss
- current loop is closed end-to-end from offline standardization through loader / train / loss / inference / evaluation
- current standardization runtime writes `failure_manifest` and `qa_summary`, and reuses existing outputs through resume scan
- current defaults no longer depend on host-specific absolute repo paths
- current validation path uses sequential eval loader and avoids double-forward in epoch validation
- current postprocess tolerates missing `torchvision.ops.nms` through pure PyTorch fallback
