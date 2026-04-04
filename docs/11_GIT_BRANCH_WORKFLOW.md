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
