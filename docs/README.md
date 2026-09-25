# 문서 안내

2026-09-23 기준. 첫 PV26 joint 80,000 step 학습과 AIHub validation 57,700장 전체 검증을 완료했다. 후속 파인튜닝 연구와 실험 설계를 정리했으며, 사용자가 지정한 수동주행 MCAP의 최종 정량 평가와 실차 처리시간 검증은 남아 있다. 개별 문서의 이전 측정 결과는 해당 문서의 기준 날짜를 따른다.

| 문서 | 내용 |
| --- | --- |
| [PRD](0_PRD.md) | 관측 대상, 출력, 실행 조건 |
| [현재 상태](00A_CURRENT_STATUS.md) | 구현한 기능과 실제 확인 결과 |
| [모델 구조](2_SYSTEM_ARCHITECTURE.md) | 공유 본체, 도로표식 디코더, SignalAttr |
| [학습과 평가](6_TRAINING_AND_EVALUATION.md) | 데이터 공급, 정밀도, OOM 대응, 재개와 저장 |
| [실행 안내](7_RUN_GUIDE.md) | check_env TUI, 학습, 재개, SignalAttr, 내보내기와 추론 명령 |
| [추론 처리시간](8_INFERENCE_PERFORMANCE.md) | 첫 호출과 반복 실행의 지연시간, 후처리 병목과 최적화 순서 |
| [80k 이후 연구·실험 설계](20260923_PV26_STAGE2_RESEARCH.md) | 문헌 근거, 파인튜닝·데이터·손실·표현 비교, 예산과 다음 에이전트 인수인계 |
| [후속 파인튜닝 정식 DoE](20260923_PV26_FORMAL_DOE.md) | 6개 pilot의 한계, 87회 후보 행렬, 3-seed 반복·교차 효과 분석과 자동 실행 |
| [수동주행 MCAP 최종 평가](20260923_MCAP_FINAL_EVALUATION_PROTOCOL.md) | 260920 14:07:38 기록, 최종 평가 분리, 라벨·신호등 상태·시간 안정성 지표 |
| [다음 작업](00C_NEXT_GATES.md) | 본학습과 성능 평가에서 할 일 |

[코드 보관](../legacy/README.md)은 이전 실행 경로와 ROS 연동 시 사용할 위치를 설명한다. [보관 문서](legacy/README.md)는 과거 설계와 실행 안내를, [history](history/README.md)는 실험 결과를 담고 있다.
