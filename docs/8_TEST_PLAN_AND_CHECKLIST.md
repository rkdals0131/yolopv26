# Test Plan And Checklist

## 문서/구현 공통 체크리스트

- [ ] 문서와 코드가 일치한다
- [ ] obsolete 파일이 남아 있지 않다
- [ ] README가 현재 entrypoint를 가리킨다
- [ ] docs 내부 링크가 상대경로 기준으로 유지된다
- [ ] sample/transform contract 문서와 loss spec이 같은 query count를 사용한다
- [ ] `N_gt_det`와 `Q_det`가 문서와 spec에서 분리돼 있다
- [ ] interpolation, padding fill, float coordinate policy가 문서에 고정돼 있다

## standardization 체크리스트

- [x] source README 생성
- [x] source inventory 생성
- [x] conversion report 생성
- [x] debug overlay 생성
- [x] lane / traffic source 모두 처리
- [x] 원본 dataset intact 유지
- [x] BDD100K det-only canonical source 처리
- [x] resume scan / skip-existing 처리
- [x] failure manifest 생성
- [x] QA summary 생성

## loader 체크리스트

- [x] standardized scene/det를 모두 읽는다
- [x] online resize/pad를 공통 처리한다
- [x] partial label source를 구분한다
- [x] tiny subset iteration이 돈다

## target encoder 체크리스트

- [x] det target
- [x] TL bit target
- [x] lane vector target
- [x] stop-line target
- [x] crosswalk target
- [x] transformed-space consistency

## pretrained trunk 체크리스트

- [x] adapter skeleton + version gate
- [x] official `yolo26n.pt` load 가능
- [ ] backbone load 성공
- [ ] neck load 성공
- [ ] custom head random init 확인
- [ ] freeze/unfreeze stage 확인

## custom head 체크리스트

- [x] det raw head skeleton
- [x] TL attr raw head skeleton
- [x] lane query head skeleton
- [x] stop-line query head skeleton
- [x] crosswalk query head skeleton
- [x] documented raw output shape 일치

## loss 체크리스트

- [x] det loss finite
- [x] TL attr masked loss finite
- [x] lane loss finite
- [x] stop-line loss finite
- [x] crosswalk loss finite
- [x] no-positive batch 안전 처리
- [x] final detector assignment runtime
- [x] synthetic assignment sanity test
- [x] lane Hungarian matching runtime
- [x] stop-line Hungarian matching runtime
- [x] crosswalk Hungarian matching runtime

## regression test 체크리스트

- [x] unit test 통과
- [x] forward regression 통과
- [x] backward regression 통과
- [x] tiny overfit 통과
- [ ] debug sample 시각화 확인

## trainer / evaluator 체크리스트

- [x] stage-wise freeze policy skeleton
- [x] optimizer group 분리
- [x] 1-step train runtime
- [x] dataset-balanced sampler helper
- [x] checkpoint save/load
- [x] history summary / JSONL logging
- [x] training live/profiling log: epoch/iteration/ETA/elapsed/epoch start + n-iter avg wait/load/fwd/loss/bwd and p50/p99
- [x] full epoch fit loop
- [x] val loop
- [x] best / last checkpoint write
- [x] AMP option
- [x] grad accumulation
- [x] grad clip
- [x] auto resume
- [x] non-finite / OOM guard
- [x] validation sequential eval loader
- [x] batch loss summary evaluator
- [x] GT count summary evaluator
- [x] inference postprocess decode
- [x] evaluator predict-batch runtime
- [x] detector/TL/lane family metric runtime
- [x] evaluator single-forward validation path
- [x] torchvision NMS runtime test
- [x] prepared PV26 dataset train/infer e2e test
- [x] prepared PV26 dataset runtime sanity test
- [x] env preflight command

## 명령 체크리스트

- [x] `python3 -m unittest discover -s test -v`
- [x] `python3 -m tools.od_bootstrap prepare-sources`
- [x] `python3 -m tools.od_bootstrap build-teacher-datasets`
- [x] `python3 -m unittest discover -s test -v`가 docs sync test까지 포함해 통과
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
- [x] `python3 -m unittest discover -s test -p 'test_portability_runtime.py' -v`
- [x] `python3 tools/check_env.py`
- [x] `python3 -m unittest discover -s test -p 'test_aihub_standardize.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_bdd100k_standardize.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_loader.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_train_infer_e2e.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_runtime_sanity.py' -v`
- [x] `python3 tools/run_pv26_train.py --preset default`
- [x] model regression command
- [x] training regression command
- [x] `python3 tools/check_env.py --check-yolo-runtime`
- [x] `python3 tools/run_pv26_train.py`
