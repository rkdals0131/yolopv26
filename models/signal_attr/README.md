# SignalAttr

신호등 crop을 받아 색상과 화살표 점등을 예측하는 분류기다. 모델은 [classifier.py](../../model/signal_attr/classifier.py), 학습은 [training.py](../../model/signal_attr/training.py)에 있다. 박스의 여백과 crop 생성은 [crop.py](../../model/signal_attr/crop.py)가 처리한다.

## 포함한 가중치

`best_signal_attr.pt`는 Plan B에서 사용하던 체크포인트를 2026-09-21에 복사한 파일이다. 크기는 561,409바이트이며, 입력은 RGB 128×128이다. 복사 직후 원본과 SHA-256이 같은 것을 확인했다.

```text
b7cdfb844479b0bb84606dd8d201d180659ca185b4067a682aeffea939b67709
```

코드와 가중치는 YOLOPV26 안에서 불러온다. 실행에 Plan B 패키지를 설치할 필요는 없다.

동봉한 가중치는 이전 상태 라벨로 학습했다. 보행자 상태를 제외하고 좌회전과 기타 화살표를 합쳤던 기준 모델이다. 제품용 추론에서는 이 가중치의 상태를 무효로 표시한다.

`competition_20260923.pt`는 `state_semantics=left_arrow`가 기록된 ROS 추론용 SignalAttr 체크포인트다. `models/pv26/competition_20260924.pt`와 함께 사용한다. 기존 `best_signal_attr.pt`의 학습 초기화 용도는 유지한다.

새 학습 경로는 차량과 보행자의 색상을 학습하고, 화살표는 차량의 좌회전만 사용한다. 차량용과 보행자용의 종류는 본체 검출기가 구분한다. 새 라벨로 crop을 만들고 재학습하는 명령은 [실행 안내](../../docs/7_RUN_GUIDE.md#signalattr)에 있다.

## 사용

저장소 루트에서 모델을 불러온다.

```python
from pathlib import Path
from model.signal_attr import load_signal_attr_classifier_checkpoint

loaded = load_signal_attr_classifier_checkpoint(
    Path("models/signal_attr/best_signal_attr.pt"), device="cpu"
)
model = loaded["model"]
```

`model.signal_attr.SignalAttrRuntime`은 원본 이미지와 검출 박스를 받아 crop 생성과 배치 판독을 수행한다. `from_checkpoint`로 가중치를 읽고 `predict(image, detections)`로 검출 ID에 연결된 상태를 얻는다.

TorchScript로 내보내려면 다음 명령을 실행한다.

```bash
python3 tools/export_signal_attr_torchscript.py
```

결과는 같은 디렉터리의 `best_signal_attr.torchscript.pt`와 `best_signal_attr.torchscript.meta.json`이다. 파일이 이미 있으면 `--overwrite`로 갱신한다. `--checkpoint`, `--output`, `--metadata`로 경로를 지정할 수도 있다.

TorchScript는 정규화된 `[N, 3, 128, 128]` 텐서를 받아 다음 순서로 반환한다.

| 출력 | 형태 | 의미 |
| --- | --- | --- |
| `base_color_logits` | `[N, 4]` | off, red, yellow, green 순서 |
| `arrow_logit` | `[N]` | 화살표 점등 logit. 제품 가중치는 차량 좌회전 의미 |

입력 정규화와 crop 설정, 상태 의미와 소등 정책은 모델에 포함한 `metadata.json`과 옆의 JSON 파일에 기록한다. 판독 정책은 내부 [classifier.py](../../model/signal_attr/classifier.py)에 있다. 제품용 [runtime.py](../../model/signal_attr/runtime.py)는 검출 ID에 상태를 연결하고, 보행자 좌회전은 판독 대상에서 제외한다.

## 확인한 결과

복사한 기준 모델은 실제 crop 5개에서 Plan B TorchScript와 logit 차이가 0이었다. 새 제품 경로는 소량의 실제 crop으로 GPU 학습과 재개, 상태별 평가와 내보내기를 확인했다. 전체 데이터 학습과 정확도 평가는 남아 있다. 자세한 시험 범위는 [현재 상태](../../docs/00A_CURRENT_STATUS.md)에 기록했다.
