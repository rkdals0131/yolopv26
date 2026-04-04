# Road-Marking Head Rewrite Checklist

## 문서 목적

- 이 문서는 [13_ROAD_MARKING_HEAD_REWRITE_DESIGN.md](13_ROAD_MARKING_HEAD_REWRITE_DESIGN.md)를 실제 구현 순서로 쪼갠 sequential checklist다.
- implementer는 아래 순서대로만 진행한다.
- phase를 건너뛰거나, 뒤 phase 결정을 앞 phase에서 임의로 당겨서 바꾸지 않는다.

## 공통 규칙

- [ ] `13_ROAD_MARKING_HEAD_REWRITE_DESIGN.md`의 tensor contract를 먼저 고정한다
- [ ] public prediction surface는 계속 `points_xy`로 유지한다
- [ ] trunk rewrite를 끼워 넣지 않는다
- [ ] stop_line / crosswalk를 lane row-anchor 표현으로 억지 통일하지 않는다
- [ ] architecture break 이후 checkpoint exact resume 불가 정책을 문서/코드에 함께 반영한다
- [ ] `yolo26n.pt` / `yolo26s.pt` variant compatibility를 유지한다

## phase 0. doc lock

### 목표

- 설계 문서를 코드보다 먼저 고정한다.

### checklist

- [ ] design doc에 lane / stop_line / crosswalk internal shape가 모두 적혀 있다
- [ ] lane query count `24`, stop_line query count `8`, crosswalk query count `8`이 문서에 고정돼 있다
- [ ] lane anchor row count `16`과 row 정의가 문서에 적혀 있다
- [ ] stop_line `2 endpoints + width`, crosswalk `4-corner quad`가 문서에 고정돼 있다
- [ ] old checkpoint exact resume unsupported 정책이 문서에 적혀 있다
- [ ] visibility source policy가 `raw visibility`와 `pseudo-visibility`를 구분해서 적혀 있다
- [ ] docs map과 live tracker가 새 문서를 가리킨다

### exit criteria

- [ ] 구현자가 architecture / tensor / migration 관련 추가 결정을 할 필요가 없다

## phase A. lane row-anchor rewrite

### 목표

- lane path를 pooled MLP에서 spatial row-anchor branch로 교체한다.

### implementation checklist

- [ ] standardization 단계에서 source lane visibility 존재 여부를 audit한다
- [ ] source에 실제 visibility가 있으면 canonical scene에 그대로 보존한다
- [ ] source에 실제 visibility가 없으면 canonical scene에 `pseudo-visibility`를 기록하고 문서/테스트 명칭을 맞춘다
- [ ] shared spatial fusion stem 추가
- [ ] lane memory branch 추가
- [ ] lane pooled MLP 제거
- [ ] lane query decoder 추가
- [ ] lane internal contract를 `B x 24 x 38`로 전환
- [ ] lane target encoder를 row-anchor 기반으로 전환
- [ ] source visibility 또는 pseudo-visibility를 loader -> target encoder -> loss까지 일관되게 전달
- [ ] lane Hungarian cost를 visible-anchor masked geometry 기준으로 교체
- [ ] lane loss를 `obj + color/type + masked x + visibility + smoothness`로 교체
- [ ] lane visibility TV/span regularizer 추가
- [ ] lane postprocess를 anchor-row reconstruction으로 교체
- [ ] lane decode에서 longest contiguous visible run rule 추가
- [ ] lane dedupe 추가
- [ ] old `12 x 54` lane-specific assumptions 제거

### verification checklist

- [ ] `test/test_pv26_heads.py`가 새 lane shape를 검증한다
- [ ] `test/test_pv26_target_encoder.py`가 raw visibility preservation을 검증한다
- [ ] `test/test_pv26_loss_spec.py`가 새 lane contract를 검증한다
- [ ] `test/test_pv26_loss_runtime.py`가 lane loss finite/backward를 검증한다
- [ ] `test/test_pv26_postprocess.py`가 lane `points_xy` reconstruction과 dedupe를 검증한다
- [ ] `test/test_pv26_evaluator.py`가 lane metric path 유지 여부를 검증한다
- [ ] lane-only tiny overfit smoke test가 loss 감소를 보인다

### exit criteria

- [ ] lane branch는 pooled vector를 전혀 읽지 않는다
- [ ] lane public output은 계속 `points_xy`다
- [ ] lane regression gates가 모두 통과한다

## phase B. stop_line spatial geometry rewrite

### 목표

- stop_line을 pooled MLP에서 geometry-aware spatial head로 바꾼다.

### implementation checklist

- [ ] geometry memory branch 추가
- [ ] stop_line small query decoder 추가
- [ ] stop_line pooled MLP 제거
- [ ] stop_line internal contract를 `B x 8 x 6`으로 전환
- [ ] stop_line target encoder를 `2 endpoints + width`로 전환
- [ ] stop_line width-valid mask를 encoded batch contract에 추가
- [ ] 2-point GT는 width unsupervised auxiliary prior로 처리한다
- [ ] stop_line matching cost를 endpoint/width/angle/length 기준으로 전환
- [ ] stop_line loss를 `obj + endpoint + width + geometry regularizer`로 전환
- [ ] stop_line postprocess를 centerline `points_xy` 재구성으로 전환
- [ ] stop_line dedupe 추가

### verification checklist

- [ ] new stop_line target encoding unit test 추가
- [ ] new stop_line loss runtime unit test 추가
- [ ] new stop_line decode unit test 추가
- [ ] evaluator metric path가 stop_line `points_xy`를 계속 읽는지 검증한다

### exit criteria

- [ ] stop_line branch는 pooled vector를 읽지 않는다
- [ ] stop_line public output은 계속 `points_xy`다
- [ ] stop_line metric regression이 기존 경로를 유지한다

## phase C. crosswalk quad rewrite

### 목표

- crosswalk를 pooled MLP에서 quad-aware spatial head로 바꾼다.

### implementation checklist

- [ ] crosswalk small query decoder 추가
- [ ] crosswalk pooled MLP 제거
- [ ] crosswalk internal contract를 `B x 8 x 9`로 전환
- [ ] crosswalk target encoder를 ordered quad 기준으로 전환
- [ ] crosswalk matching cost를 `corner L1 + polygon IoU` 기준으로 전환
- [ ] crosswalk loss를 `obj + corner + polygon consistency`로 전환
- [ ] crosswalk postprocess를 quad `points_xy` reconstruction으로 전환
- [ ] crosswalk dedupe 추가

### verification checklist

- [ ] new crosswalk target encoding unit test 추가
- [ ] new crosswalk loss runtime unit test 추가
- [ ] new crosswalk decode unit test 추가
- [ ] evaluator metric path가 crosswalk `points_xy`를 계속 읽는지 검증한다

### exit criteria

- [ ] crosswalk branch는 pooled vector를 읽지 않는다
- [ ] crosswalk public output은 계속 `points_xy`다
- [ ] crosswalk metric regression이 기존 경로를 유지한다

## phase D. checkpoint / export / hardening

### 목표

- architecture break를 안전하게 land 시킨다.

### implementation checklist

- [ ] checkpoint version / spec version을 올린다
- [ ] old exact resume unsupported error를 명시적으로 추가한다
- [ ] shape-aware partial load path를 trunk / det / TL 기준으로 분리한다
- [ ] `load_matching_state_dict()`를 migration helper로 연결한다
- [ ] lane / stop_line / crosswalk head random init 정책을 명시한다
- [ ] stage 4 freeze policy를 새 모듈 트리에 맞게 다시 정의한다
- [ ] road-marking 파트를 freeze policy가 잡을 수 있도록 모듈 경계를 정리한다
- [ ] TorchScript raw-head export에 crosswalk tensor를 포함시킨다
- [ ] TorchScript export metadata를 새 tensor shape에 맞게 갱신한다
- [ ] docs sync를 새 문서 / 새 contract 기준으로 갱신한다
- [ ] execution status에 migration risk와 verification 결과를 기록한다

### verification checklist

- [ ] `test/test_docs_sync.py`
- [ ] `test/test_pv26_trainer.py`
- [ ] `test/test_pv26_evaluator.py`
- [ ] `test/test_pv26_postprocess.py`
- [ ] `test/test_pv26_loss_spec.py`
- [ ] `test/test_pv26_loss_runtime.py`
- [ ] `test/test_run_pv26_train.py`
- [ ] checkpoint migration targeted test
- [ ] old exact resume rejection targeted test
- [ ] export smoke test
- [ ] compileall smoke test

### exit criteria

- [ ] new architecture generation checkpoint가 저장/복원된다
- [ ] old generation exact resume가 모호하지 않게 실패한다
- [ ] public prediction surface와 evaluator metric path가 유지된다

## 권장 실행 순서

1. phase 0 doc lock
2. phase A lane
3. phase B stop_line
4. phase C crosswalk
5. phase D hardening

이 순서는 바꾸지 않는다.

## 최종 검증 명령 묶음

- [ ] `python3 -m pytest -q test/test_docs_sync.py`
- [ ] `python3 -m pytest -q test/test_pv26_heads.py test/test_pv26_target_encoder.py test/test_pv26_loss_spec.py test/test_pv26_loss_runtime.py test/test_pv26_postprocess.py test/test_pv26_evaluator.py test/test_pv26_trainer.py`
- [ ] `python3 -m pytest -q test/test_run_pv26_train.py`
- [ ] `python3 -m compileall -q common model tools test`

## phase 진행 원칙

- lane을 먼저 안정화하기 전에는 stop_line / crosswalk rewrite를 시작하지 않는다.
- stop_line / crosswalk는 shared geometry memory를 쓰지만 final parameterization은 공유하지 않는다.
- public output surface를 깨는 변경은 hardening phase 이전에도 이후에도 허용하지 않는다.
