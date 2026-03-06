> **Archived (2026-03-05)** — kept for history; may be outdated.  
> Canonical docs: `docs/PV26_PRD.md`, `docs/PV26_DATASET_CONVERSION_SPEC.md`, `docs/PV26_DATASET_SOURCES_AND_MAPPING.md`, `docs/PV26_IMPLEMENTATION_STATUS.md`.

# YOLO26 Trunk 리스크 정리 + 리팩터링 옵션 + (n→s/m…) + TensorRT 계획

작성일: 2026-03-05

이 문서는 PV26에서 사용하는 Ultralytics YOLO26 트렁크(`PV26MultiHeadYOLO26`)의 “구조적 리스크”가 왜 생기는지 쉬운 말로 풀고, 앞으로 `n → s → m → l → x`로 스케일업 및 TensorRT 배포까지 고려한 리팩터링 옵션을 정리합니다.

---

## 1) 지금 베이스가 `yolo26n-seg`인가? (결론: **아님**, 현재는 **detect 트렁크**)

PV26 코드는 Ultralytics의 **DetectionModel(탐지 트렁크)** 을 사용합니다.

- PV26 코드: `PV26MultiHeadYOLO26`는 `DetectionModel(self.yolo26_cfg, ...)`로 생성합니다. (`pv26/multitask_model.py`)
- Ultralytics YOLO26 모델 YAML에서:
  - `yolo26.yaml`은 **detect 모델**이고, `model=yolo26n.yaml` 같은 별칭은 `scales`에서 **scale=n**으로 `yolo26.yaml`을 호출한다는 주석이 들어 있습니다.
  - `yolo26-seg.yaml`은 **seg 모델**이고, `model=yolo26n-seg.yaml` 별칭은 `yolo26-seg.yaml`을 **scale=n**으로 호출한다는 주석이 들어 있습니다.

즉, “PV26는 seg 트렁크(yolo26n-seg)를 그대로 쓰는 구조”가 아니라,
**detect 트렁크(yolo26n)를 가져오고, DA/RM/lane-subclass는 PV26에서 별도 head로 붙이는 구조**입니다.

참고: Ultralytics 공식 YOLO26 모델 문서도 detect/segment 등 task를 구분합니다.

---

## 2) “남은 구조적 리스크”가 뜻하는 바 (쉬운 설명)

PV26의 `PV26MultiHeadYOLO26`는 트렁크에서 **특정 feature map 2개**를 꺼내서(P3/8 backbone, P3/8 head) DA/RM head 입력으로 씁니다.

현재 방식은 “Ultralytics 내부 구조”에 **꽤 강하게 의존**합니다:

1) **고정 인덱스 의존 (`model[4]`가 P3라는 가정)**  
   - “Ultralytics 모델 안에서 4번째 레이어 출력이 P3(backbone)일 것이다”를 전제로 forward hook을 겁니다.  
   - 그런데 Ultralytics는 YAML로 모델을 조립하는데, YAML/버전/옵션이 조금만 바뀌면 레이어 개수/순서가 바뀔 수 있습니다.  
   - 그러면 `model[4]`가 더 이상 P3가 아닐 수 있고, 그 순간 **P3를 못 잡거나(에러)**, 혹은 **엉뚱한 feature를 잡아(성능 문제)** 로 이어질 수 있습니다.

2) **딕셔너리 키/리스트 순서 의존 (`preds['one2many']['feats'][0]`)**  
   - “`one2many` 딕셔너리에 `feats` 리스트가 있고, 그 0번째가 P3(head)일 것이다”를 전제로 합니다.  
   - Ultralytics의 출력 포맷은 버전/설정에 따라 dict/tuple이 섞일 수 있고, 키/구조도 바뀔 수 있습니다. (Ultralytics 코드도 출력 포맷을 여러 케이스로 처리합니다.)

3) **forward hook + Python dict 갱신(부작용)이 `torch.compile`/export에 불리할 수 있음**  
   - 지금은 forward 중에 hook이 Python 레벨에서 실행되고, `_feat` dict가 갱신됩니다(“부작용”).  
   - `torch.compile`은 모델을 “그래프 형태로” 안정적으로 묶고 최적화하려고 하는데, 이런 부작용이 많으면 **그래프가 끊기거나(compile 이득 감소)**, 경우에 따라 **compile 실패** 가능성이 커집니다.  
   - Ultralytics도 export/추론 파이프라인에서 “정적인 텐서 출력”을 강하게 선호합니다(특히 TensorRT/ONNX).

한 문장으로:  
**“Ultralytics 내부 레이아웃(레이어 번호, dict 키, 리스트 순서)에 기대서 feature를 꺼내는 방식”은 버전/모델 스케일/배포(export/compile)에서 깨지기 쉽다**는 뜻입니다.

---

## 3) 리팩터링 옵션 제안 (단계별)

여기서 목표는 2가지입니다:

- (안정성) Ultralytics 버전/CFG 스케일이 바뀌어도 P3 feature를 안정적으로 얻기
- (배포성) `torch.compile`, ONNX/TensorRT export 경로에서 side-effect를 최소화하기

### Option 1) “최소 변경” — 고정 인덱스 제거(Shape/Stride 기반으로 P3 찾기)

핵심 아이디어:
- dry forward 때 여러 레이어 출력을 잠깐 수집하고,
- **입력 대비 1/8 해상도(H,W)가 되는 feature map을 P3(backbone)로 선택**합니다.

장점
- 코드 변경량이 작고, `model[4]` 하드코딩보다 훨씬 덜 취약
- `n/s/m/l/x`로 스케일 바뀌어도 stride=8 feature는 보통 유지됨

단점
- 여전히 hook/부작용 패턴이 남습니다(compile/export 관점에서는 “차선책”)

권장 적용
- “빠르게 안전도만 올리고 싶다”면 1순위

### Option 2) “중간 변경” — hook을 없애고, 트렁크의 forward를 ‘명시적으로’ 따라가며 feature 추출

핵심 아이디어:
- Ultralytics가 내부에서 레이어를 순회하며 feature를 모으는 로직을 참고해,  
  **우리가 필요한 P3(backbone/head)를 forward 내부에서 텐서로 직접 캐치**합니다(딕셔너리 side effect 없이).
- 즉, “모델의 실행 흐름을 따라가며” 중간 feature를 꺼내는 방식입니다.

장점
- hook/`_feat` dict 같은 부작용 제거 → `torch.compile`/export 친화도 증가
- “레이어 번호 하드코딩”을 줄일 수 있음

단점
- Ultralytics 내부 forward 로직 변경에 영향을 받을 수 있음(그래도 Option1보다 낫고, 버전 pinning과 같이 쓰면 안정적)

권장 적용
- “TensorRT까지 고려하면 결국 이쪽이 필요할 확률이 높다”는 현실적인 선택지

### Option 3) “장기/가장 안전” — Ultralytics 버전 pin + 구조 고정(또는 트렁크를 레포에 vendor)

핵심 아이디어:
- Ultralytics 의존도를 줄이거나(벤더링), 최소한 **특정 커밋/버전으로 고정(pin)** 해서
  “내부 레이아웃 변화” 리스크를 통제합니다.

장점
- 장기적으로 가장 덜 깨짐
- TensorRT 파이프라인(ONNX → TRT)에서도 재현성이 좋아짐

단점
- 유지보수 비용이 늘고, upstream 개선을 자동으로 못 가져옴

권장 적용
- “상용 배포 + TRT 엔진 재현성 + 장기 운영”까지 가면 결국 필요해질 가능성이 큼

---

## 4) 스케일업 계획: `n → s → m → l → x`

Ultralytics YOLO26는 `yolo26.yaml` 안에 **scales(n/s/m/l/x)** 를 정의하고, `yolo26n.yaml` 같은 별칭으로 scale을 선택하는 방식입니다.

PV26 관점에서 중요한 포인트:

- 트렁크 스케일을 키우면(예: `yolo26s.yaml`, `yolo26m.yaml`)  
  **feature 채널 수가 바뀌어** DA/RM head 입력 채널도 달라집니다.
- 현재 PV26 코드는 init에서 dry forward로 채널 수를 추론해 head를 구성하므로, “채널 수 변화” 자체는 자동 대응 가능한 편입니다.  
  (문제는 “P3를 어디서 안정적으로 가져오느냐” 쪽이 더 큼 → 3절 리팩터링이 중요)

추천 운영 방안(현실적):
1) **학습은 s/m로 올려보기** (성능 vs 비용)
2) **추론/배포는 TensorRT FP16** 기본, 필요하면 INT8(캘리브레이션) 옵션
3) 트렁크 크기별로 “정해진 입력 해상도(예: 960×544)”를 유지해서 엔진을 안정화

---

## 5) TensorRT 배포(엔진)까지 고려한 설계 포인트

Ultralytics는 export 모드에서 TensorRT 엔진(`format=engine`)을 공식 지원합니다.

하지만 PV26는 “Ultralytics의 detect/seg task를 그대로 쓰는 모델”이 아니라,
**Ultralytics detect 트렁크 + PV26 커스텀 head** 구조라서, 배포 경로를 두 가지로 나눠 생각하는 게 안전합니다.

### Track A) “Ultralytics 그대로 export” (가능하면 가장 쉬움)

- 전제: PV26 전체 모델을 Ultralytics export 파이프라인이 이해할 수 있는 형태로 맞춘다.
- 장점: 공식 지원(export 옵션, 엔진 빌드, 호환성)이 많음
- 단점: PV26 커스텀 head/출력 포맷 때문에 맞추는 비용이 발생할 수 있음

### Track B) “PV26 전용 ONNX/TensorRT 파이프라인” (실무적으로 많이 선택)

아이디어:
- PV26 모델 forward가 **“순수 텐서 출력(고정 구조)”** 을 내도록 정리하고,
- 그걸 ONNX로 내보낸 뒤 TensorRT로 엔진을 만든다.

이 트랙에서 특히 중요한 것:
- **hook/딕셔너리/가변 길이 출력**을 줄일수록 export가 쉬워집니다.
- detect 출력은 NMS 결과처럼 길이가 변하는 형태보다,
  **고정 shape의 raw head 출력**(혹은 end2end가 제공하는 고정 텐서들)로 두는 편이 보통 export에 유리합니다.

Ultralytics export 문서(모드/포맷/제약)는 Track A/B 공통으로 참고 가치가 큽니다.

---

## 6) 권장 로드맵(현실적인 순서)

1) (이미 완료) lane-subclass를 lane marker로 게이팅하는 **Option-A 후처리 정책 확정**
2) **Option 1**: `model[4]` 하드코딩 제거(Shape/Stride로 P3 선택)
3) 스케일업 실험: `n → s → m` (학습/검증 파이프라인 정리)
4) TensorRT 계획 확정:
   - Track A가 가능한지(투자 대비 효율) 판단
   - 아니면 Track B로 “export-friendly forward + ONNX + TRT” 구축
5) 장기 운영이면 **Option 2 또는 Option 3**로 구조 안정화(compile/export 포함)

---

## 7) 공식 참고 링크 (문서/오픈소스 구조)

- Ultralytics YOLO26 모델 문서: `https://docs.ultralytics.com/models/yolo26/`
- Ultralytics Export 모드(ONNX/TensorRT 포함): `https://docs.ultralytics.com/modes/export/`
- 오픈소스 모델 YAML(Detect): `https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml`
- 오픈소스 모델 YAML(Seg): `https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-seg.yaml`
- 오픈소스 모델/task 로딩 코드(출력 포맷 처리 포함): `https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py`
