# Status History

> 이 디렉터리는 실패/수정/결과 오답노트다.
> 원래 상황, 바꾼 것, 실제 결과, 다음에 하지 말 것을 압축해서 남긴다.

긴 history 본문은 번호 범위별 파일에 보존한다. AI나 리뷰어에게 전체 history를 넣지 말고, 먼저 이 표에서 관련 chunk를 고른 뒤 필요한 범위만 읽는다.

## Split Index

| Entries | Date span | Summary |
| --- | --- | --- |
| [`1-25`](00B_001-025.md) | `2026-04-06 to 2026-05-11` | 초기 metric collapse, strict runtime cleanup, AMP 실패, 2026-05-05 baseline, lane60 probe, Gate 1/2/3 초반 진단. |
| [`26-50`](00B_026-050.md) | `2026-05-11` | stop-line component/readout audit와 lane row-scan 계열 초반 probe가 왜 production path가 아니었는지 보존. |
| [`51-75`](00B_051-075.md) | `2026-05-11` | lane/stop-line loss, sampler, validator, task-head merge 실험의 2026-05-11 negative ledger. |
| [`76-100`](00B_076-100.md) | `2026-05-11 to 2026-05-12` | crosswalk retention, stop-line/lane readout, threshold/oracle, soft-instance 계열이 닫힌 이유 정리. |
| [`101-125`](00B_101-125.md) | `2026-05-12` | segment-MIL, flip-TTA, validator/manifest audit, hard-negative sampler와 center-rank 계열의 재현 증거. |
| [`126-150`](00B_126-150.md) | `2026-05-12 to 2026-05-13` | center-rank replay, fragment/union stop-line readout, lane FN recovery, selector/photometric audit 정리. |
| [`151-175`](00B_151-175.md) | `2026-05-13` | lane FN/repair bucket, stop-line no-oracle center/axis 진단, area-rescue와 geometry guard 실패 기록. |
| [`176-200`](00B_176-200.md) | `2026-05-13 to 2026-05-14` | lane duplicate/semantic/ensemble/repairability와 stop-line geometry-regression 계열이 막힌 근거. |
| [`201-225`](00B_201-225.md) | `2026-05-14` | P4/context/task-mask/TTA/polyfit 실험과 projection/composite tooling 복구 기록. |
| [`226-249`](00B_226-249.md) | `2026-05-14 to 2026-05-29` | scale/TTA, live distill, reproducibility restore, temporal-neighbor, PCGrad 초반 2026-05-29 기록. |
| [`250-275`](00B_250-275.md) | `2026-05-29` | PCGrad/adapters, stop-line segment/verifier, task routing, specialist train, retention distill 축 정리. |
| [`276-300`](00B_276-300.md) | `2026-05-29 to 2026-05-30` | stop-line/lane specialist, dense/conditional decoder, static-trunk, focus-crop, mask/endpoint probe 결과. |
| [`301-326`](00B_301-326.md) | `2026-05-30` | context/query/distance/augmentation/current-family vector decoder와 ROI/verifier 실험의 2026-05-30 ledger. |
| [`327-351`](00B_327-351.md) | `2026-05-30` | raw candidate, area-ROI verifier, source-router, dense-seed geometry, interpolation/repair 계열 기록. |
| [`352-376`](00B_352-376.md) | `2026-05-30 to 2026-05-31` | denoise/hard-negative, temporal/source-router, dense-map/conditional/affine probe가 fixed gate를 넘지 못한 근거. |
| [`377-401`](00B_377-401.md) | `2026-05-31` | distill, current-family decoder, temporal/ridge/area-ROI/copy-paste 실험과 area-ROI frontier 기록. |
| [`402-416`](00B_402-416.md) | `2026-05-31 to 2026-06-02` | late temporal stop-line/lane probes, artifact sweep, main/develop/branch/tag hygiene 최종 checkpoint. |

## Update Rule

- 새 history entry는 마지막 chunk에 이어 쓰거나, chunk가 다시 길어지면 다음 번호 범위 파일을 추가한다.
- 이 index에는 chunk 링크, date span, 한 줄 summary만 갱신한다.
- 상세 실패/수정/결과 본문은 `docs/history/`에 둔다.
