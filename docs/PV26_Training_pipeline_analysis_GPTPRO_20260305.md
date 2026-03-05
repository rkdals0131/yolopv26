압축 파일 기준으로 `pv26/multitask_model.py`, `pv26/criterion.py`, `pv26/torch_dataset.py`, `tools/train/train_pv26.py`를 중심으로 검토했습니다. 테스트는 23개가 수집되는 구조였고, stub/데이터셋/criterion 계열은 통과했습니다. 다만 실제 `PV26MultiHeadYOLO26` 경로는 이 환경에 `ultralytics`가 없어 실행 확인 대신 정적 리뷰로 판단했습니다.

총평부터 말씀드리면, 이 코드는 생각보다 꽤 잘 정리되어 있습니다.
“대충 붙인 멀티태스크 코드”가 아니라, 데이터셋 규약, partial-label 처리, 손실 마스킹, 학습 루프가 서로 맞물리도록 신경 쓴 흔적이 분명합니다. 연구용 1차 베이스라인으로는 충분히 괜찮습니다.

다만, 실제 성능과 유지보수성을 깎아먹을 수 있는 핵심 리스크가 몇 군데 있습니다.

## 좋았던 점

1. **partial-label 설계가 일관적입니다.**
   `Pv26ManifestDataset`에서 supervision이 없는 채널은 all-255 ignore 마스크로 내보내고 (`pv26/torch_dataset.py:310-335`), criterion에서는 `has_*` 플래그로 손실을 정확히 가립니다 (`pv26/criterion.py:120-132`, `400-474`). 멀티태스크 데이터셋에서 이 부분이 흐트러지면 학습이 금방 망가지는데, 여기서는 꽤 깔끔합니다.

2. **데이터 경계가 좋습니다.**
   이미지는 CPU에서 `uint8`로 유지하다가 device에서 정규화하고 (`tools/train/train_pv26.py:523-531`), collate 단계에서 detection flat target도 미리 만들어 둡니다 (`tools/train/train_pv26.py:286-324`). 이런 구조는 실제 학습 throughput에 유리합니다.

3. **4-head의 역할 분리는 논리적으로 맞습니다.**
   DA는 얕은 P3/backbone에서, RM과 lane-subclass는 head P3에서 뽑는 구조 (`pv26/multitask_model.py:241-285`) 자체는 YOLOPv2 계열 감각으로 보면 꽤 자연스럽습니다.
   “DA는 디테일 + RM은 더 의미론적 피처”라는 생각이 반영되어 있습니다.

4. **학습 파이프라인의 기본기**가 있습니다.
   AMP, `channels_last`, checkpoint/resume, compile fallback, profile 옵션 등은 실험용 코드 치고 꽤 단정합니다.

## 핵심적으로 아쉬운 점

### 1) 가장 큰 리스크는 Ultralytics 내부 구조에 너무 강하게 결합되어 있다는 점입니다

`pv26/multitask_model.py:241-259`을 보면,

* backbone P3를 `self.det_model.model[4]`에 하드코딩해 hook으로 잡고,
* head P3는 `preds["one2many"]["feats"][0]`를 직접 꺼내며,
* hook 결과를 `self._feat`라는 Python dict에 저장합니다 (`232-239`, `268-279`).

이 방식은 기능적으로는 돌아갈 수 있지만, 다음 세 가지 문제가 있습니다.

* Ultralytics 버전이 바뀌면 쉽게 깨집니다.
* `torch.compile` 관점에서 아주 compile-friendly한 패턴이 아닙니다.
* 무엇보다 `__init__`에서 channel 추론용 dry forward를 **train 모드**로 돌립니다 (`248-253`).

마지막 항목이 은근히 중요합니다. train 모드에서 zero dummy를 한 번 흘리면, BN running stats가 초기화 직후부터 조금 오염될 수 있습니다. 큰 사고는 아닐 수 있지만, 굳이 감수할 이유가 없습니다.

여기는 `eval()` 상태에서 dry forward를 하거나, 더 좋게는 YOLO trunk가 필요한 feature를 명시적으로 return하도록 감싸는 쪽이 맞습니다. 그리고 `ultralytics` 버전과 `yolo26n.yaml` 의존성을 명확히 pinning하는 것이 좋습니다. 지금 저장소 내부에는 yaml이 보이지 않아 외부 패키지/버전에 사실상 묶여 있습니다.

### 2) lane-subclass head는 현재 구조상 성능 손해를 보기 쉬운 설계입니다

현재 lane-subclass는 RM과 완전히 독립된 별도 head로 나오고 (`pv26/multitask_model.py:282-284`), 손실은 full-image 기준의 plain cross-entropy입니다 (`pv26/criterion.py:444-474`).

이 조합의 문제는 두 가지입니다.

* lane pixel이 극히 희소하므로 background가 지나치게 우세합니다.
* subclass가 lane marker 존재 여부와 직접 연결되어 있지 않아, 계층적 일관성이 약합니다.

쉽게 말해,
“이 픽셀이 lane인지 아닌지”와
“lane이라면 white/yellow, solid/dashed 중 무엇인지”를
완전히 독립적으로 배우게 하고 있습니다.

이 문제는 실제로 thin structure에서 성능을 깎기 쉽습니다.
제 판단으로는 lane-subclass loss를 적어도 **lane-positive 영역 중심**으로 바꾸는 것이 좋습니다. 방법은 여러 가지가 있는데, 예를 들면:

* GT lane-marker가 1인 위치에서만 subclass loss를 주기
* subclass CE에 class weight 또는 focal 계열을 넣기
* inference 때도 subclass를 RM lane-marker와 논리적으로 묶기

지금 구조는 “분류 head를 하나 더 붙였다”는 점에서는 좋지만, 얇은 선 구조를 배우는 방식으로는 아직 거칠습니다.

### 3) best checkpoint 선택 기준이 멀티태스크 모델답지 않습니다

`tools/train/train_pv26.py:457-460`을 보면 `map50`이 있으면 그것만으로 best checkpoint를 고릅니다.

즉, 기본 설정에서는 `best.pt`가 사실상 **detection best**입니다.
그런데 이 모델은 detection-only 모델이 아니라 4-head 멀티태스크입니다.

그래서 실제로는

* detection은 조금 좋아졌는데
* DA/RM/lane-subclass는 나빠진 checkpoint

가 best로 저장될 수 있습니다.

이건 운영 철학의 문제이기도 하지만, 지금 구조에서는 꽤 중요한 왜곡입니다.
최소한 다음 중 하나는 필요합니다.

* `best_det.pt`, `best_seg.pt`, `best_total.pt`를 따로 저장
* composite score 사용
* 혹은 `val_total`과 `map50`을 함께 조건에 넣기

멀티태스크 모델을 detection metric 하나로만 선발하는 것은 아깝습니다.

### 4) detection subset supervision은 현재 “조용히 틀릴 가능성”이 있습니다

`pv26/criterion.py:164-170`을 보면 Ultralytics loss 경로에서 `scope_code != 2`인 샘플을 전부 keep합니다. 즉, `subset`도 사실상 `full`처럼 들어갑니다.

반면 dense stub path는 `subset`을 더 보수적으로 다룹니다 (`pv26/criterion.py:380-385` 쪽 동작).
즉, arch에 따라 subset 의미가 달라집니다.

현재 데이터셋이 full-only라면 당장 문제는 없을 수 있습니다. 하지만 이건 전형적인 잠복 버그입니다.
나중에 subset label이 들어오는 순간 false negative가 detection trunk에 들어갈 수 있습니다.

이 부분은 TODO로 남겨둘 문제가 아니라,
**지원하지 않으면 즉시 에러를 내는 편이 더 안전합니다.**

### 5) optimizer 구성이 너무 평평합니다

`tools/train/train_pv26.py:551-562`에서 모든 trainable parameter에 동일 LR/weight decay를 걸고 있습니다.

이 구조는 특히 `--det-pretrained`를 쓰는 순간 아쉬움이 커집니다.

* pretrained detection trunk
* random init segmentation heads

를 같은 LR로 같이 돌리면, 초반에 새 head의 noisy gradient가 trunk를 흔들 수 있습니다.
게다가 BN/bias까지 동일 weight decay를 먹는 구성도 일반적으로는 예쁘지 않습니다.

여기는 적어도

* trunk와 new head를 param group으로 분리
* pretrained trunk LR은 더 낮게
* BN/bias는 decay 제외 또는 완화

정도는 들어가는 편이 좋습니다.
지금 구조는 “돌아가는 기본형”으로는 괜찮지만, 수렴성과 재현성 면에서는 손해를 볼 가능성이 큽니다.

### 6) lane-subclass mIoU 산식은 희소 클래스에서 지나치게 비관적으로 나올 수 있습니다

`tools/train/train_pv26.py:1003-1010`, `1063-1079`를 보면 class별 lane-subclass IoU를 계산할 때, 해당 클래스가 실제로 전혀 없는 경우에도 `union==0`이면 0.0으로 처리됩니다.

즉,
“그 클래스가 val set에 사실상 없어서 예측도 안 했음”
같은 경우도 0점처럼 보일 수 있습니다.

희소 클래스 모니터링에는 이 방식이 꽤 불리합니다.
present-only 기준으로 skip할지, 현재 방식처럼 엄격히 볼지는 철학의 문제이지만, 지금 숫자는 체감보다 낮게 나올 가능성이 있습니다.

## 구조 자체에 대한 제 판단

4-head 구조 자체는 나쁘지 않습니다. 오히려 출발점은 괜찮습니다.

* Detection trunk 공유
* DA는 shallow
* RM/lane-subclass는 head feature 사용
* lane-subclass를 binary RM과 분리

이 방향은 충분히 설득력이 있습니다.

다만 지금 상태는 “좋은 방향의 v1”이지, 아직 “잘 다듬어진 v2”는 아닙니다.
특히 lane-subclass를 독립 head로 둔 만큼, 그에 맞는 loss/metric/hierarchy까지 같이 정교화되어야 설계 의도가 살아납니다.

## 제일 먼저 손볼 우선순위

1. `PV26MultiHeadYOLO26`의 hook + train-mode dry forward 정리
   (`pv26/multitask_model.py:241-259`)

2. lane-subclass loss를 lane 영역 중심으로 재설계
   (`pv26/criterion.py:444-474`)

3. best checkpoint 기준을 detection-only에서 멀티태스크 기준으로 수정
   (`tools/train/train_pv26.py:457-460`)

4. optimizer param group 분리
   (`tools/train/train_pv26.py:551-562`)

5. subset detection supervision 미지원이면 즉시 에러 처리
   (`pv26/criterion.py:164-170`)

한 줄로 정리하면,
**코드 완성도는 꽤 높지만, 실제 성능을 더 끌어올리려면 “Ultralytics 결합부”, “lane-subclass 학습 방식”, “멀티태스크에 맞는 선발/최적화 정책” 세 축을 먼저 손보는 것이 맞습니다.**
