# Git Branch Workflow

이 저장소의 브랜치 운영은 `omx -> develop -> main`의 역할 분리를 전제로 한다.

## Branch Roles
- `omx`는 `omx`, `ulw`, `team`, `swarm` 같은 OMX 기반 세션이 raw하게 진행되는 작업 브랜치다.
- `omx`에서는 auto-checkpoint, 중간 merge, 실험성 커밋이 자주 생길 수 있다. 이 히스토리는 작업 히스토리로 보고, 그대로 GitHub 게시용 히스토리로 간주하지 않는다.
- `develop`은 통합 브랜치다. 일반 개발이나 기능 추가도 여기서 진행할 수 있다.
- `main`은 릴리스 브랜치다. `develop`에서 기능 테스트까지 끝난 뒤에만 올린다.

## Merge Policy
- `develop`을 cherry-pick만 모아둔 브랜치처럼 운영하지 않는다.
- 대신 `omx`에서 작업을 진행하다가 기능 단위로 의미 있는 경계가 생기면 `develop`으로 주기적으로 merge한다.
- 필요하면 `develop`도 다시 `omx`로 merge해서 두 브랜치가 실제 merge history로 계속 이어지게 유지한다.
- 핵심은 `omx`와 `develop`이 서로 단절되지 않고, 반복적인 merge로 함께 진화하는 것이다.

## Merge Commit Messages
- merge commit 제목은 그냥 `Merged`처럼 쓰지 않는다.
- 그 merge 시점에 어떤 기능 묶음이 들어왔는지를 제목에 직접 쓴다.
- 예시: `Merge omx into develop: split PV26 train runtime and stabilize facade`
- 예시: `Merge develop into omx: bring integrated OD bootstrap cleanup forward`

## Practical Rule
- 현재 시점의 최신 구현 상태를 볼 때 `main`만 기준으로 판단하지 않는다.
- `main`은 의도적으로 과거 상태일 수 있다. 최신 통합 상태는 `develop`, 최신 raw 작업 상태는 `omx`에서 확인한다.
- 최종 `main` 반영과 GitHub push는 테스트가 끝난 뒤 사용자 판단으로 진행한다.

## Lane-Family F1 Experiment Worktrees

현재 lane-family 목표는 broader validation에서 lane / stop-line / crosswalk F1을 모두 `0.60` 이상으로 만드는 것이다. `phase_objective` 0.6 통과나 exact subset 통과만으로는 이 목표를 달성한 것으로 보지 않는다.

F1 0.6+ 실험은 더 이상 "실험 하나당 branch 하나"로 운영하지 않는다. 브랜치는 연구 방향이 실제로 갈라질 때만 만든다. 단일 방향 안의 반복 실험, negative result, threshold/decoder/loss 세부 시도는 같은 branch에서 커밋과 문서 이력으로 직렬화한다.

Branch creation rule:

- 새 branch는 병렬 비교가 필요한 독립 가설에만 만든다. 예: "current dense/postprocess line"과 "새로운 BEV/temporal line"처럼 merge 방향이 실제로 다른 경우.
- 같은 가설의 train-batch scale-up, threshold adjustment, verifier feature 추가, loss-weight 변경, cleanup/docs update마다 새 branch를 만들지 않는다.
- worktree를 병렬로 써야 하면 먼저 기존 방향 branch를 재사용한다. 임시 worktree 이름이 필요해도 Git branch를 새로 늘리지 않는다.
- 고정 best, router anchor, old frontier는 live branch가 아니라 tag로 보존한다.
- `develop`은 현재 공식 통합 branch다. best/status/docs는 `develop`에 올리고, 장기 연구 방향 branch는 develop에 통합되면 삭제하거나 tag로 치환한다.

If a branch is truly needed:

- branch 이름은 `exp/lane-family-f1/<direction>`처럼 장기 방향 단위로 짧게 쓴다. 예: `exp/lane-family-f1/bev-temporal-stopline`.
- worktree 경로는 repo 밖 sibling 경로를 쓴다. 예: `/home/kai/yolopv26-exp-bev-temporal-stopline`.
- 한 worktree는 한 축만 바꾼다. 예: postprocess, stop-line head, lane head, sampler/feeder 중 하나.
- 같은 worktree에서 architecture와 sampler를 동시에 바꾸지 않는다.
- negative result도 커밋이나 문서로 남긴다. 반복 금지 근거는 `docs/00B_STATUS_HISTORY.md`에 축약한다.
- develop 승격 전에는 exact replay와 broader validation replay를 모두 확인한다.

Current archived anchors:

- `archive/lane-family-current-frontier-20260602`: develop에 편입된 current frontier를 보존한다.
- `archive/lane-family-router-best-20260529`: best numeric two-checkpoint/router stop-line tradeoff anchor를 보존한다.

초기 축은 아래 네 개다.

| Axis | Scope |
| --- | --- |
| `postprocess` | geometry filters, thresholds, component filtering, mask-to-vector decode |
| `stopline` | stop-line target/loss/head/decode contract |
| `lane` | centerline recall, gated refinement, vectorizer recovery |
| `sampler-feeder` | task-positive sampler, validation support, batch composition, encoded feeder behavior |

Cleanup boundary:

- `/tmp` 안의 PV26 임시 산출물은 필요 없으면 삭제해도 된다.
- `/tmp` 밖 경로는 삭제하지 않는다.
- repo 밖 worktree나 run folder를 치울 때는 먼저 삭제후보 폴더로 이동한다. 예: `/home/kai/yolopv26_deletion_candidates/<timestamp>-<name>`.
- `runs/`는 `.gitignore` 대상이므로 커밋에 포함되지 않는다. 문서에는 재현 가능한 command와 핵심 metric만 남긴다.
