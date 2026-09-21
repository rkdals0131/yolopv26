# 이전 구현 보관

2026-09-21에 현재 신호등·도로표식 모델에서 분리한 코드다. 범용 OD bootstrap, teacher와 pseudo-label 생성, 구형 헤드와 학습기, Modal 실행 도구, 이전 설정과 해당 테스트를 보관한다.

현재 학습은 저장소 루트의 `tools/run_pv26_train.py`와 `tools/train_signal_attr.py`를 사용한다. 이전 실행을 조사할 때는 이 디렉터리에서 별도 Python 프로세스로 연다. 두 구현이 `model`과 `common`이라는 패키지 이름을 공유한다.

## 기존 ROS 런타임

`pv26_ros_runtime`이 구형 named-output TorchScript를 사용할 때는 `pv26_repo_root`를 이 `legacy` 디렉터리로 지정한다.

```text
pv26_repo_root:=/home/user1/ROS2_Workspace/ros2_ws/src/yolopv26/legacy
```

구형 `model.engine.postprocess`와 의존 코드를 함께 보관했으며, 별도 프로세스에서 import를 확인했다. 현재 두 헤드 모델의 출력은 이 ROS 런타임의 구형 출력 계약과 다르므로 별도 연결 작업이 필요하다.

데이터와 실행 산출물, 기존 대형 가중치는 옮기지 않았다. 이전 학습을 다시 실행하려면 보관한 설정의 데이터·가중치 경로와 당시 환경을 확인해야 한다. 과거 설계와 명령 설명은 [보관 문서](../docs/legacy/README.md)에 있다.
