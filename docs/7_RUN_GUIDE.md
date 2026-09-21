# 실행 안내

명령은 YOLOPV26 저장소 루트에서 실행한다. 본체 설정은 [pv26.yaml](../config/pv26.yaml), SignalAttr 설정은 [signal_attr.yaml](../config/signal_attr.yaml)에 있다. 실행에 사용하는 라이브러리는 [requirements.txt](../requirements.txt)에 정리했다.

## 상태 확인과 작업 TUI

```bash
python3 tools/check_env.py
```

현재 설정, GPU 가용 VRAM, 저장소 여유 공간, 데이터 경로와 최근 학습 실행을 표시한다. 작업을 고르면 사용할 데이터나 체크포인트와 출력 경로를 선택하고, 최종 명령에 `y`로 답했을 때 실행한다. 경로는 공백이 있어도 그대로 입력할 수 있다.

| 키 | 동작 |
| --- | --- |
| `1` | 신호등 원본에서 SignalAttr crop 생성. 소등 라벨 처리와 원본 수 제한 선택 |
| `C` / `A` | 각각 본체 / SignalAttr의 새 학습 |
| `D` | 본체 또는 SignalAttr를 지정한 업데이트 수만 실행하고 저장 |
| `E` | 기존 실행의 설정과 optimizer, 데이터 진행을 복구해 재개 |
| `K` | 선택한 가중치로 현재 설정의 새 학습 시작 |
| `F` / `G` | 각각 본체 / SignalAttr TorchScript 내보내기 |
| `P` | 영상의 관측 JSON과 overlay 생성 |
| `L` | 실행 당시 설정, 평가 지표와 체크포인트 경로 보기 |
| `S` | 사용할 YAML 파일의 경로 변경 |
| `H` / `R` / `Q` | 도움말 / 상태 새로고침 / 종료 |

선택 목록의 `M`은 경로 직접 입력, 질문 중 `B`는 이전 화면으로 복귀다. 숫자 파라미터는 YAML에서 편집한다. `E`는 해당 실행에 저장된 설정을 사용하고, `K`는 현재 설정으로 새 optimizer와 학습률 일정을 시작한다.

작업 로그는 같은 터미널에 출력한다. 학습 루프 실행 중 `Ctrl+C`를 누르면 중단을 전달하고, 학습기가 진행 중인 step을 마쳐 저장한 뒤 TUI로 돌아온다. 다시 `Ctrl+C`를 누르면 종료를 재요청하고, 응답하지 않는 작업은 강제 종료한다. 이후 `E`로 마지막 체크포인트에서 재개할 수 있다.

화면의 진행도는 요약·평가 JSON에 마지막으로 기록된 값이다. 실행 중 표시는 파일 락으로 확인하며 큰 체크포인트를 매번 불러오지 않는다. 원본의 학습/검증 표시는 디렉터리 존재 여부이고 crop 개수는 데이터 생성 기록의 값이다. 실제 학습 로그와 데이터 검사는 각 CLI에서 확인한다.

```bash
python3 tools/check_env.py --once
python3 tools/check_env.py --json
python3 tools/check_env.py --config /경로/pv26.yaml --signal-config /경로/signal_attr.yaml
```

`--once`는 화면을 한 번 출력하고, `--json`은 상태를 JSON으로 출력하고 종료한다. 터미널에 연결되지 않은 기본 실행도 JSON을 출력한다. 이 모드에서는 학습이나 내보내기를 실행하지 않는다.

## 본체 학습

```bash
python3 tools/run_pv26_train.py --config config/pv26.yaml
```

기본 단계는 joint다. 신호등만 학습할 때는 `--stage detector`, 본체를 고정하고 도로표식 디코더를 초기화할 때는 `--stage roadmark`를 사용한다. 다음 단계로 옮길 때는 이전 가중치를 `--initial-checkpoint`로 지정한다. 새 실행은 optimizer와 학습률 일정을 새로 시작한다.

초기 가중치는 저장소 루트의 `yolo26s.pt`에서 읽는다. 데이터 경로는 설정의 `data.sources`에 추가할 수 있으며, source별 `weight`가 노출 비율을 정한다. `kind: traffic`은 신호등 검출만, `kind: roadmark`는 차선과 정지선만 감독한다.

현재 기본값은 BF16, 논리 배치 32장, 물리 배치 16장, worker 4개다. 논리 배치는 한 번의 optimizer 업데이트에 사용하는 표본 수이며, 물리 배치는 한 번에 GPU에 올리는 표본 수다. OOM이 발생하면 물리 배치를 줄여 같은 논리 배치를 다시 처리한다.

짧은 연결 시험에는 다음 명령을 사용할 수 있다.

```bash
python3 tools/run_pv26_train.py --steps 2 --sample-limit 64
```

`--sample-limit`은 source별 원본 수를 제한한다. 본학습에서는 이 옵션을 생략한다. 검증 표본은 source마다 전체 목록에 걸쳐 고르게 선택하며, 물리 배치 크기를 바꿔도 같은 표본을 사용한다.

## 재개

```bash
python3 tools/run_pv26_train.py --resume-run /home/user1/Storage/ROS2_Workspace_offload/yolopv26/실행폴더
```

`--steps N`을 함께 주면 N번 더 업데이트한 뒤 저장하고 종료한다. 전체 학습률 일정은 처음 설정한 `max_steps`를 유지한다. 재개할 때는 저장된 데이터 목록과 학습 설정을 사용한다. worker 수와 물리 배치는 실행 환경에 맞춰 조정할 수 있다.

SIGINT 또는 SIGTERM을 받으면 진행 중인 step을 마친 뒤 저장한다. 강제 종료되면 마지막 완료된 체크포인트에서 재개한다. 같은 실행 폴더에 두 학습 프로세스가 동시에 접근하면 두 번째 실행은 중단된다.

## SignalAttr

`config/signal_attr.yaml`의 `prepare`는 원본 경로와 crop 설정을, `train`은 학습 기본값을 관리한다. 각 명령의 `--config`로 다른 설정 파일을 지정할 수 있다. 먼저 제품용 crop을 만든다.

```bash
python3 tools/train_signal_attr.py prepare \
  --output-dir /home/user1/Storage/ROS2_Workspace_offload/yolopv26/signal_crops \
  --all-off-policy exclude
```

예시는 모든 램프가 off인 표본을 상태 학습에서 제외한다. 실제 소등 표적으로 학습하려면 `--all-off-policy off`를 지정한다. 이 선택은 데이터와 체크포인트에 기록된다. 기타 화살표만 켜진 경우에는 좌회전 음성 표적으로 사용한다.

`--raw-root`로 원본 위치를, `--workers`로 crop 생성 병렬도를 지정한다. `--sample-limit N`은 각 split에서 처리할 원본 이미지 수를 제한한다. 원본 파일 목록은 먼저 탐색한다.

```bash
python3 tools/train_signal_attr.py train \
  --dataset /home/user1/Storage/ROS2_Workspace_offload/yolopv26/signal_crops
```

새 실행은 상태별 균등 표본 선택을 기본으로 한다. 차량의 색상과 좌회전 조합, 보행자의 색상별 그룹을 균등하게 뽑는다. `--sampling natural`을 주면 원래 표본 비율을 사용한다. checkpoint 선택에는 상태별 오탐과 미탐을 반영한 `macro_state_f1`을 사용하고, 판독 유효 비율도 기록한다.

SignalAttr도 `--resume-run`과 `--steps`로 재개할 수 있다. 실행별로 저장한 crop 라벨과 설정을 읽으므로, 나중에 원본 라벨을 수정해도 그 실행의 표본은 유지된다. 배포용 체크포인트는 실행 폴더의 `best_signal_attr.pt`다.

## 모델 내보내기

```bash
python3 tools/export_pv26_torchscript.py \
  --checkpoint /home/user1/Storage/ROS2_Workspace_offload/yolopv26/실행폴더/checkpoints/best.pt \
  --device cuda:0

python3 tools/export_signal_attr_torchscript.py \
  --checkpoint /home/user1/Storage/ROS2_Workspace_offload/yolopv26/SignalAttr실행폴더/best_signal_attr.pt
```

기본 출력은 체크포인트 옆의 TorchScript 파일이다. 출력 형식은 모델 안의 `metadata.json`에 포함하며, 같은 내용을 읽기 쉬운 JSON 파일로도 저장한다. 기존 출력 파일을 갱신할 때는 `--overwrite`를 지정한다.

본체는 고정된 입력 크기와 batch 2로 내보낸다. 출력은 네트워크 좌표의 검출 `[B,N,6]`과 도로표식 logit `[B,3,H/4,W/4]`이다. SignalAttr는 정규화한 crop 텐서를 받아 색상과 화살표 logit을 반환한다.

## 이미지 추론

```bash
python3 tools/predict_pv26.py \
  --checkpoint /home/user1/Storage/ROS2_Workspace_offload/yolopv26/실행폴더/checkpoints/best.pt \
  --signal-checkpoint /home/user1/Storage/ROS2_Workspace_offload/yolopv26/SignalAttr실행폴더/best_signal_attr.pt \
  --images /경로/left.jpg /경로/right.jpg \
  --device cuda:0 \
  --output /home/user1/Storage/ROS2_Workspace_offload/yolopv26/관측결과.json
```

`--overlay`에 디렉터리를 지정하면 박스와 점열을 그린 이미지도 저장한다. JSON에는 원본 영상 좌표의 박스, 차량용/보행자용 구분, 상태와 도로표식 점열이 담긴다.

이 명령은 PyTorch 체크포인트를 읽는다. 내보낸 TorchScript를 사용할 때는 포함된 metadata에 맞춰 입력을 만들고, 내부 후처리 함수로 박스와 점열을 복원한다. 동봉한 이전 SignalAttr 기준 가중치는 좌회전 전용 재학습 전 모델이므로, 제품용 추론에서는 상태를 무효로 표시한다.

## 저장 위치

큰 산출물은 외장 SSD의 `~/Storage/ROS2_Workspace_offload/yolopv26` 아래에 저장한다. 실행 폴더에는 설정, 고정한 표본 목록, 최근 평가 결과가 있으며 checkpoints에는 latest, previous, best가 유지된다. 코드와 소형 기준 가중치는 저장소에서 관리한다.
