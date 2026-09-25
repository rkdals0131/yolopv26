# YOLOPV26

카메라 영상에서 차량용 신호등과 보행자용 신호등을 구분하고 점등 상태를 읽는다. 흰 차선, 노란 차선, 정지선은 원본 영상 좌표의 점열로 출력한다.

공유 YOLO26-s 본체와 도로표식 디코더, SignalAttr를 이 저장소에서 학습하고 실행한다. AIHub 원본을 읽는 피더, mixed precision, OOM 재시도, 체크포인트 재개를 구현했다. 첫 PV26 joint 80,000 step 학습과 AIHub 전체 검증을 완료했으며, 제품용 SignalAttr 장기 학습과 실차 최종 평가는 남아 있다.

## 실행

저장소 루트에서 실행한다. 원본 경로와 학습 설정은 [pv26.yaml](config/pv26.yaml)에 있다.

```bash
python3 tools/check_env.py
```

TUI에서 GPU와 저장소, 데이터 경로, 최근 학습 상태를 보고 작업을 고른다. 실행할 명령을 확인한 뒤 학습, 재개, 내보내기와 이미지 추론을 시작한다. 상태만 읽을 때는 `--once` 또는 `--json`을 사용한다. 메뉴와 직접 CLI 명령은 [실행 안내](docs/7_RUN_GUIDE.md)에 있다.

본체에는 신호등 검출 헤드와 도로표식 헤드가 있다. 도로표식 헤드가 흰 차선, 노란 차선, 정지선 세 채널을 출력하고, 검출된 신호등의 상태는 SignalAttr가 읽는다.

SignalAttr의 crop 생성과 학습 설정은 [signal_attr.yaml](config/signal_attr.yaml), 학습과 내보내기 명령은 [실행 안내](docs/7_RUN_GUIDE.md)에 있다.

차선 원본은 도로표식만, 신호등 원본은 신호등만 감독한다. `data.sources`의 `kind`가 감독 태스크를, `weight`가 학습 노출 비율을 정한다.

## 문서

- [목표와 실행 조건](docs/0_PRD.md)
- [현재 구현과 확인 결과](docs/00A_CURRENT_STATUS.md)
- [모델 구조](docs/2_SYSTEM_ARCHITECTURE.md)
- [학습과 자원 관리](docs/6_TRAINING_AND_EVALUATION.md)
- [다음 작업](docs/00C_NEXT_GATES.md)
- [80k 이후 연구·실험 설계와 인수인계](docs/20260923_PV26_STAGE2_RESEARCH.md)
- [수동주행 MCAP 최종 평가 설계](docs/20260923_MCAP_FINAL_EVALUATION_PROTOCOL.md)

[SignalAttr](models/signal_attr/README.md)의 기준 가중치는 저장소에 포함한다. ROS 추론에 사용할 PV26 본체와 SignalAttr 체크포인트는 각각 `models/pv26/competition_20260924.pt`, `models/signal_attr/competition_20260923.pt`에 둔다. `pv26_ros`는 이 경로를 저장소 루트 기준 상대경로로 읽는다. 본체 초기화에는 저장소 루트의 `yolo26s.pt`를 사용한다. 학습한 체크포인트는 초기 가중치 파일 없이 다시 불러올 수 있다.

이전 코드와 설정은 [legacy](legacy/README.md), 과거 설계는 [보관 문서](docs/legacy/README.md), 실험 결과는 [history](docs/history/README.md)에 있다.
