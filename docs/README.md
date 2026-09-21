# 문서 안내

2026-09-21 기준. 신호등 판독과 흰 차선, 노란 차선, 정지선 인지의 첫 구현을 마쳤다. 실제 원본을 이용한 짧은 GPU 학습과 재개, OOM 재시도, 모델 내보내기를 확인했다. 정확도와 실차 처리시간의 합격 여부는 아직 평가하지 않았다.

| 문서 | 내용 |
| --- | --- |
| [PRD](0_PRD.md) | 관측 대상, 출력, 실행 조건 |
| [현재 상태](00A_CURRENT_STATUS.md) | 구현한 기능과 실제 확인 결과 |
| [모델 구조](2_SYSTEM_ARCHITECTURE.md) | 공유 본체, 도로표식 디코더, SignalAttr |
| [학습과 평가](6_TRAINING_AND_EVALUATION.md) | 데이터 공급, 정밀도, OOM 대응, 재개와 저장 |
| [실행 안내](7_RUN_GUIDE.md) | check_env TUI, 학습, 재개, SignalAttr, 내보내기와 추론 명령 |
| [추론 처리시간](8_INFERENCE_PERFORMANCE.md) | 첫 호출과 반복 실행의 지연시간, 후처리 병목과 최적화 순서 |
| [다음 작업](00C_NEXT_GATES.md) | 본학습과 성능 평가에서 할 일 |

[코드 보관](../legacy/README.md)은 이전 실행 경로와 ROS 연동 시 사용할 위치를 설명한다. [보관 문서](legacy/README.md)는 과거 설계와 실행 안내를, [history](history/README.md)는 실험 결과를 담고 있다.
