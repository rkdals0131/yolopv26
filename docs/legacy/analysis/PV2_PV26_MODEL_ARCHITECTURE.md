# YOLOPv2 / YOLOPv26 Model Architecture Note

## Scope

이 문서는 현재 로컬 환경에서 확인 가능한 두 모델 구조를 정리한다.

- `YOLOPv2`
  - `spade` 패키지에서 실제로 사용하는 TorchScript 가중치 `yolopv2.pt` 기준
  - 원본 학습 코드가 저장소에 없어서, 가중치 내부 모듈 구조와 실행 결과를 기준으로 분석
- `YOLOPv26`
  - 현재 `YOLOPv26` 저장소 코드 기준
  - `PV26MultiHeadYOLO26`를 실제 instantiate 해서 확인한 구조 기준

핵심만 먼저 말하면:

1. `YOLOPv2`는 `detect + drivable segmentation + lane line segmentation` 3갈래 구조다.
2. `YOLOPv26`는 `yolo26n detect trunk` 위에 `DA`, `RM`, `lane-subclass` head를 PV26 전용으로 붙인 구조다.
3. 즉 `PV26`는 `yolo26n-seg`를 그대로 쓰는 게 아니라, `yolo26n detect backbone/neck + PV26 custom segmentation heads` 구조다.

---

## 1. YOLOPv2 Structure

## 1.1 Runtime Output Form

`spade/scripts/demo.py`와 `spade/scripts/yolopv2_semantic_node.py` 기준으로, 현재 `yolopv2.pt`의 출력은 아래 형식이다.

```python
([pred, anchor_grid], seg, ll)
```

- `pred`
  - detection feature outputs
- `anchor_grid`
  - detection anchor grid
- `seg`
  - drivable-area segmentation logits
- `ll`
  - lane-line segmentation logits

실제 사용처:

- `seg` -> drivable mask
- `ll` -> lane mask
- 두 결과를 합쳐 `spade`에서 쓰는 `semantic_id`로 변환

현재 `spade`에서 최종 semantic id는 다음처럼 쓴다.

- `0 = background`
- `1 = drivable`
- `2 = lane`

## 1.2 Actual Block Layout

현재 로컬 `yolopv2.pt`를 열어 확인한 결과, 상위는 `Sequential(0..125)` 한 덩어리다.

큰 구조는 아래처럼 읽는 것이 가장 자연스럽다.

### A. Shared trunk / backbone + early neck

- `0 ~ 50`
- 주 구성 요소:
  - `Conv`
  - `MP`
  - `Concat`
  - 반복되는 CSP 계열 블록성 패턴

이 구간은 전형적인 구형 YOLO 계열의 shared feature extractor 역할을 한다.

### B. High-level context block

- `51`
- `SPPCSPC`

큰 receptive field 문맥을 모으는 블록이다.

### C. Detection neck

- `52 ~ 101`
- 주 구성 요소:
  - `Conv`
  - `Upsample`
  - `Concat`
  - FPN/PAN 스타일의 상향/하향 feature fusion

### D. Detection head

- `102 ~ 104`
  - `RepConv` x 3
- `105`
  - `IDetect`

즉 detection은 3개 scale head를 사용한다.

### E. Drivable segmentation head

- `106 ~ 116`
- 흐름:

```text
Conv
-> Upsample
-> BottleneckCSP
-> Conv
-> Upsample
-> Conv
-> Upsample
-> Conv
-> BottleneckCSP
-> Upsample
-> out_Conv
```

최종 출력 채널은 `2`다.

### F. Lane line segmentation head

- `117 ~ 125`
- 흐름:

```text
SE_AT
-> ConvTran
-> BottleneckCSP
-> Conv
-> ConvTran
-> Conv
-> BottleneckCSP
-> ConvTran
-> out_Conv
```

최종 출력 채널은 `1`이다.

## 1.3 Actual Output Shapes

입력 `1 x 3 x 384 x 1408` 기준 실제 출력 shape:

- detection feature maps
  - `(1, 255, 48, 176)`
  - `(1, 255, 24, 88)`
  - `(1, 255, 12, 44)`
- `seg`
  - `(1, 2, 384, 1408)`
- `ll`
  - `(1, 1, 384, 1408)`

anchor는 3 scale에서 아래와 같다.

- scale 1
  - `(12,16)`, `(19,36)`, `(40,28)`
- scale 2
  - `(36,75)`, `(76,55)`, `(72,146)`
- scale 3
  - `(142,110)`, `(192,243)`, `(459,401)`

## 1.4 Practical Interpretation

현재 `spade` 관점에서 중요한 건 detection보다 segmentation 두 개다.

- `seg`
  - 주행 가능 영역
- `ll`
  - 차선

즉 `YOLOPv2`는 `spade`용으로 보면 사실상:

```text
camera image
-> shared vision trunk
-> drivable head + lane head
-> semantic_id
```

처럼 쓰이고 있다.

---

## 2. YOLOPv26 Structure

## 2.1 Overall Design

현재 `PV26`의 실제 모델은 `PV26MultiHeadYOLO26`이다.

구조 철학은 아래와 같다.

```text
yolo26n detect trunk
-> OD branch (Ultralytics Detect head 그대로 사용)
-> DA branch (PV26 custom head)
-> RM branch (PV26 custom decoder + head)
-> lane-subclass branch (PV26 custom head)
```

중요:

- `PV26`는 `yolo26n detect 모델`을 trunk로 사용한다.
- segmentation trunk를 통째로 가져오는 구조가 아니다.
- detection trunk에서 feature를 뽑아 PV26 전용 segmentation head를 붙인다.

## 2.2 YOLO26 Trunk Actual Layer Layout

`DetectionModel("yolo26n.yaml", ch=3, nc=7)`를 실제 생성했을 때의 상위 레이어는 아래 24개다.

| idx | module | 역할 |
|---|---|---|
| 0 | `Conv` | stem downsample |
| 1 | `Conv` | downsample |
| 2 | `C3k2` | early feature block |
| 3 | `Conv` | downsample |
| 4 | `C3k2` | shallow backbone feature (`P3 backbone`) |
| 5 | `Conv` | downsample |
| 6 | `C3k2` | mid feature block |
| 7 | `Conv` | downsample |
| 8 | `C3k2` | deep feature block |
| 9 | `SPPF` | spatial pyramid pooling |
| 10 | `C2PSA` | high-level attention/context block |
| 11 | `Upsample` | top-down fusion 시작 |
| 12 | `Concat` | with layer 6 |
| 13 | `C3k2` | fused neck block |
| 14 | `Upsample` | top-down fusion |
| 15 | `Concat` | with layer 4 |
| 16 | `C3k2` | smallest-scale detect feature |
| 17 | `Conv` | downsample |
| 18 | `Concat` | with layer 13 |
| 19 | `C3k2` | middle-scale detect feature |
| 20 | `Conv` | downsample |
| 21 | `Concat` | with layer 10 |
| 22 | `C3k2` | largest-scale detect feature |
| 23 | `Detect` | 3-scale detection head |

즉 `yolo26n` 내부만 놓고 보면:

```text
Backbone: 0 ~ 10
Neck:     11 ~ 22
Detect:   23
```

## 2.3 Where PV26 Branches

`PV26`는 detection trunk에서 feature를 두 군데서 가져온다.

### A. `p3_backbone`

- source: `det_model.model[4]`
- 의미:
  - 얕은 backbone feature
  - stride 8
  - DA head 입력

### B. `p3_head`

- source:
  - `Detect` head 입력 feature 중 첫 번째 scale
  - 코드상 `one2many.feats[0]`
- 의미:
  - neck을 거친 fused feature
  - stride 8
  - RM / lane-subclass 입력

즉 분기 구조는 아래와 같다.

```text
yolo26n backbone/neck
  ├─ shallow P3 backbone feature -> DA head
  └─ fused P3 head feature      -> RM decoder -> RM / lane-subclass head
```

## 2.4 Detection Head

Detection은 Ultralytics `Detect`를 그대로 쓴다.

현재 `nc=7` 기준:

- 3개 scale 사용
- 입력 feature source:
  - layer `16`
  - layer `19`
  - layer `22`

head 내부는 scale별로 크게 두 갈래다.

- `cv2`
  - box regression branch
- `cv3`
  - class prediction branch

또한 내부적으로:

- `one2many`
- `one2one`

두 종류 출력을 같이 유지한다.

즉 detection 쪽은 단순한 old-style anchor tensor가 아니라, 현재 Ultralytics detect head 계약을 그대로 따른다.

## 2.5 Drivable Area Head

`DA`는 `DrivableAreaHeadP3`로 구현돼 있다.

구조:

```text
input: p3_backbone
-> stage0 ConvBNAct
-> up8_to4
-> stage1 ConvBNAct
-> up4_to2
-> stage2 ConvBNAct
-> up2_to1   # seg_output_stride == 1 일 때
-> pred 1x1 conv
```

특징:

- 얕은 feature를 사용한다.
- detection neck을 거친 feature가 아니라 backbone의 P3에서 바로 분기한다.
- 출력 채널은 `1`

## 2.6 Road Marking Head

`RM`은 `RoadMarkingDecoderDeconv + RoadMarkingPredictionHead` 구조다.

decoder:

```text
input: p3_head
-> stem ConvBNAct
-> deconv1
-> deconv2
-> deconv3   # seg_output_stride == 1 일 때
```

prediction:

```text
decoded feature
-> 1x1 conv
-> 3 channels
```

채널 의미:

1. `lane_marker`
2. `road_marker_non_lane`
3. `stop_line`

## 2.7 Lane-Subclass Head

lane-subclass는 RM decoder 출력을 공유해서 쓴다.

구조:

```text
shared RM decoder output
-> 1x1 conv
-> 5 channels
```

5채널 의미:

1. background
2. white_solid
3. white_dashed
4. yellow_solid
5. yellow_dashed

즉 `RM`과 `lane-subclass`는 완전히 별도 decoder 두 개가 아니라, decoder trunk는 공유하고 마지막 prediction head만 따로 둔 구조다.

## 2.8 Actual Feature / Output Shapes

입력 `1 x 3 x 544 x 960`, `seg_output_stride=1` 기준 실제 확인 shape:

- `p3_backbone`
  - `(1, 128, 68, 120)`
- `p3_head`
  - `(1, 64, 68, 120)`
- `da`
  - `(1, 1, 544, 960)`
- `rm`
  - `(1, 3, 544, 960)`
- `rm_lane_subclass`
  - `(1, 5, 544, 960)`

즉 현재 기본 설정에서는 segmentation 출력이 입력 해상도와 같은 크기로 올라온다.

`seg_output_stride=2`로 바꾸면 segmentation 출력 해상도는 절반으로 내려간다.

---

## 3. YOLOPv2 vs YOLOPv26

| 항목 | YOLOPv2 | YOLOPv26 |
|---|---|---|
| trunk 계열 | 구형 YOLO 계열 TorchScript weight | `yolo26n` detect trunk |
| detection | 있음 | 있음 |
| drivable | 있음 | 있음 |
| lane line | 있음 | RM + lane-subclass로 더 세분화 |
| segmentation branch 수 | 2개 (`seg`, `ll`) | 3개 (`da`, `rm`, `rm_lane_subclass`) |
| 최종 semantic id | `spade`에서 단순 합성 | 별도 composition 필요 |
| 출력 richness | 비교적 단순 | 훨씬 풍부 |

핵심 차이:

- `YOLOPv2`
  - drivable와 lane을 바로 뽑아 `spade`가 바로 쓰기 쉬움
- `YOLOPv26`
  - detection + drivable + road marking + lane subclass까지 한 모델에서 처리
  - 대신 `spade`에 넣으려면 후처리/합성 어댑터가 필요

---

## 4. SPADE Integration Meaning

현재 `spade`가 직접 원하는 것은 사실상 이 한 장이다.

```text
sensor_msgs/Image (mono8 semantic_id)
```

따라서 `YOLOPv26`를 `spade`에 붙일 때 핵심은:

1. `DA + RM + lane-subclass`를 semantic id로 합성하고
2. 원본 카메라 해상도로 맞춘 뒤
3. `mono8` semantic image로 publish

하는 것이다.

즉 `PV26` 교체 작업의 본질은:

```text
YOLOPv2 semantic node 교체
```

이지,

```text
fusion 알고리즘 전체 재작성
```

은 아니다.

---

## 5. Evidence

### YOLOPv2 runtime usage

- [spade/scripts/demo.py](/home/user1/ROS2_Workspace/ros2_ws/src/spade/scripts/demo.py)
- [spade/scripts/yolopv2_semantic_node.py](/home/user1/ROS2_Workspace/ros2_ws/src/spade/scripts/yolopv2_semantic_node.py)

### PV26 architecture code

- [pv26/model/multitask_impl.py](/home/user1/Python_Workspace/YOLOPv26/pv26/model/multitask_impl.py)
- [pv26/model/outputs.py](/home/user1/Python_Workspace/YOLOPv26/pv26/model/outputs.py)
- [docs/PV26_PRD.md](/home/user1/Python_Workspace/YOLOPv26/docs/PV26_PRD.md)

### Note

`YOLOPv2` 부분은 저장소에 원본 정의 코드가 없어서 TorchScript 가중치 내부 모듈 트리와 실제 실행 shape를 기준으로 정리했다.
