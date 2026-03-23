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

- [ ] det loss finite
- [ ] TL attr masked loss finite
- [ ] lane loss finite
- [ ] stop-line loss finite
- [ ] crosswalk loss finite
- [ ] no-positive batch 안전 처리

## smoke test 체크리스트

- [ ] unit test 통과
- [x] forward smoke 통과
- [ ] backward smoke 통과
- [ ] tiny overfit 통과
- [ ] debug sample 시각화 확인

## 명령 체크리스트

- [x] `python3 -m unittest discover -s test -v`
- [x] `python3 -m model.preprocess.aihub_standardize --max-samples-per-dataset 4 --debug-vis-count 2`
- [x] `python3 -m model.preprocess.bdd100k_standardize --max-samples-per-split 2 --debug-vis-count 2`
- [x] `python3 -m unittest discover -s test -v`가 docs sync test까지 포함해 통과
- [x] `python3 -m unittest discover -s test -p 'test_pv26_target_encoder.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_yolo26_trunk.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_heads.py' -v`
- [x] `python3 -m unittest discover -s test -p 'test_pv26_trunk_features.py' -v`
- [x] `python3 tools/run_yolo26_trunk_smoke.py`
- [ ] loader smoke command
- [x] model smoke command
- [ ] training smoke command
