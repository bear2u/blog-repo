---
layout: post
title: "Entire CLI 완벽 가이드 - 06. Strategy 개요"
date: 2026-02-11
categories: [AI, Development Tools]
tags: [Entire, CLI, AI Coding, Strategy, Git]
description: "Entire CLI의 Manual-commit과 Auto-commit 전략 비교 및 선택 가이드"
---

## 06. Strategy 개요

Entire CLI는 AI와의 코딩 세션을 Git 저장소에 저장하는 두 가지 핵심 전략을 제공합니다. 각 전략은 서로 다른 워크플로우와 사용 사례에 최적화되어 있습니다.

### 전략의 필요성

AI 코딩 어시스턴트를 사용할 때 발생하는 주요 과제는 다음과 같습니다.

```
문제 1: 변경 사항 추적
┌─────────────────────────────────────┐
│ AI가 여러 파일을 동시에 수정       │
│ 어떤 변경이 어느 시점에 발생했는지?│
│ 잘못된 변경을 어떻게 되돌리는가?   │
└─────────────────────────────────────┘

문제 2: 실험과 본 작업 분리
┌─────────────────────────────────────┐
│ AI와의 실험적 대화                  │
│ 프로덕션 코드와 섞이면 안 됨       │
│ 좋은 결과만 메인 브랜치로          │
└─────────────────────────────────────┘

문제 3: 세션 이력 관리
┌─────────────────────────────────────┐
│ 과거 AI 대화 내용 찾기             │
│ "그때 AI가 뭐라고 했더라?"         │
│ 이전 솔루션 재사용                 │
└─────────────────────────────────────┘
```

Entire CLI의 전략 시스템은 이러한 문제를 해결합니다.

### 두 가지 전략 개요

#### 1. Manual-Commit Strategy (수동 커밋)

**철학**: "AI 변경사항을 검토 후 명시적으로 승인"

```
작업 흐름:
┌──────────────────────────────────────────────┐
│ 1. AI가 shadow branch에서 작업              │
│    main: ───o                                │
│    shadow:  └──o──o──o (AI 실험)           │
│                                              │
│ 2. 개발자가 변경 검토                        │
│    git diff main...shadow                    │
│                                              │
│ 3. 마음에 들면 checkpoint 생성               │
│    entire checkpoint create                  │
│                                              │
│ 4. 준비되면 main으로 병합                    │
│    git merge shadow                          │
└──────────────────────────────────────────────┘
```

**특징**:
- Shadow branch에서 격리된 작업
- 명시적인 checkpoint 생성
- 개발자가 병합 시점 제어
- Git 기록이 깔끔하게 유지

#### 2. Auto-Commit Strategy (자동 커밋)

**철학**: "AI 변경사항을 자동으로 기록하고 추적"

```
작업 흐름:
┌──────────────────────────────────────────────┐
│ 1. AI가 변경할 때마다 자동 커밋              │
│    main: ───o──o──o──o                      │
│         (자동) (자동) (자동)                │
│                                              │
│ 2. 각 변경이 즉시 기록됨                     │
│    - 프롬프트와 응답 저장                    │
│    - 코드 변경 커밋                          │
│                                              │
│ 3. 필요시 되돌리기 쉬움                      │
│    git reset HEAD~3                          │
│    또는                                      │
│    entire checkpoint restore <id>            │
└──────────────────────────────────────────────┘
```

**특징**:
- 현재 브랜치에서 직접 작업
- 자동 checkpoint 생성
- 즉각적인 변경 기록
- 빠른 반복 개발

### 전략 비교표

| 측면 | Manual-Commit | Auto-Commit |
|------|---------------|-------------|
| **작업 위치** | Shadow branch | 현재 branch |
| **커밋 방식** | 수동 (명시적) | 자동 (즉시) |
| **Git 기록** | 깔끔, 큐레이션됨 | 상세, 모든 단계 |
| **안전성** | 높음 (격리) | 중간 (직접 수정) |
| **되돌리기** | Branch 삭제 | git reset |
| **검토 흐름** | 병합 전 검토 | 사후 검토 |
| **사용 난이도** | 중간 | 쉬움 |

### 워크플로우별 권장 전략

#### Manual-Commit 추천 상황

```yaml
프로덕션 코드베이스:
  - 팀 프로젝트
  - 엄격한 코드 리뷰
  - 안정성 우선

실험적 탐색:
  - 여러 접근 방식 시도
  - 큰 리팩토링
  - 불확실한 변경

오픈소스 기여:
  - 깔끔한 커밋 기록 필요
  - PR 준비 작업
```

**예시**:
```bash
# 새 기능 실험
entire session start --strategy manual-commit \
  --message "새 인증 시스템 프로토타입"

# AI와 대화하며 개발
$ AI: 사용자 인증 로직을 추가해줘
# ... AI가 여러 파일 수정 ...

# 결과 검토
git diff main...entire/shadow/session-abc123

# 마음에 들면 checkpoint
entire checkpoint create --message "기본 인증 완료"

# 더 진행
$ AI: OAuth 지원 추가해줘
# ... 추가 작업 ...

# 최종 검토 후 병합
git checkout main
git merge entire/shadow/session-abc123
```

#### Auto-Commit 추천 상황

```yaml
개인 프로젝트:
  - 빠른 프로토타이핑
  - 학습 목적
  - 실험적 코드

버그 수정:
  - 빠른 수정 필요
  - 간단한 변경
  - 즉시 테스트

문서 작업:
  - README 업데이트
  - 주석 추가
  - 예제 코드
```

**예시**:
```bash
# 빠른 버그 수정
entire session start --strategy auto-commit \
  --message "로그인 버그 수정"

# AI와 대화
$ AI: 로그인 시 토큰이 저장 안 되는 버그 고쳐줘
# ... AI가 수정 ...
# 자동으로 커밋됨!

# 즉시 확인
git log -1
# commit abc123
# AI Assistant: Fix token storage bug

# 추가 수정
$ AI: 에러 메시지도 개선해줘
# ... 자동 커밋 ...

# 기록 확인
git log --oneline -5
```

### 전략 전환

두 전략 사이를 자유롭게 전환할 수 있습니다.

```bash
# 프로젝트 기본 전략 설정
entire config set strategy manual-commit

# 특정 세션만 다른 전략 사용
entire session start --strategy auto-commit

# 전략 확인
entire config get strategy
# manual-commit

# 현재 세션의 전략 확인
entire session info
# Strategy: auto-commit (session override)
```

### 전략 마이그레이션

기존 세션을 다른 전략으로 전환:

```bash
# Manual-commit → Auto-commit
# (shadow branch 내용을 main으로 병합)
git checkout main
git merge entire/shadow/session-abc123
entire session convert --to auto-commit

# Auto-commit → Manual-commit
# (현재 작업을 shadow branch로 이동)
entire session convert --to manual-commit
# 새 shadow branch 생성됨
# 기존 커밋들이 shadow로 이동
```

### 하이브리드 워크플로우

실무에서는 두 전략을 혼합하여 사용할 수 있습니다.

```bash
# 시나리오: 큰 기능 개발

# 1단계: Manual-commit으로 실험
entire session start --strategy manual-commit \
  --message "결제 시스템 설계"

# AI와 여러 접근 방식 탐색
# Shadow branch에서 안전하게 실험

# 2단계: 좋은 방향 확정 후 Auto-commit으로 전환
entire checkpoint create --message "기본 구조 확정"
entire session convert --to auto-commit

# 3단계: 빠른 반복 개발
# 자동 커밋으로 빠르게 진행

# 4단계: 마무리는 다시 Manual-commit
entire session convert --to manual-commit
# 최종 정리 및 리팩토링
```

### 전략별 내부 동작

#### Manual-Commit 내부 구조

```
.git/
├── refs/
│   ├── heads/
│   │   ├── main                    (프로덕션)
│   │   └── entire/
│   │       └── shadow/
│   │           └── session-abc123  (AI 작업)
│   └── entire/
│       └── metadata/
│           └── session-abc123      (메타데이터)
└── objects/
    └── (변경 내용 저장)

작업 영역 상태:
- Working directory: main branch 체크아웃
- Shadow branch: 백그라운드에서 업데이트
- 개발자가 명시적으로 병합하기 전까지 main 불변
```

#### Auto-Commit 내부 구조

```
.git/
├── refs/
│   ├── heads/
│   │   └── main                    (직접 업데이트)
│   └── entire/
│       └── metadata/
│           └── session-abc123      (메타데이터)
└── objects/
    └── (변경 내용 즉시 커밋)

작업 영역 상태:
- Working directory: main branch 체크아웃
- 모든 변경이 main에 즉시 커밋
- Shadow branch 없음
```

### 메타데이터 저장소

두 전략 모두 세션 메타데이터를 별도로 관리합니다.

```bash
# 메타데이터 조회
git ls-tree entire/metadata/session-abc123

# 예시 출력:
100644 blob sha1  checkpoints.json
100644 blob sha2  session.json
100644 blob sha3  prompts/001.txt
100644 blob sha4  prompts/002.txt
100644 blob sha5  responses/001.txt
100644 blob sha6  responses/002.txt
```

**메타데이터 구조**:

```json
// session.json
{
  "id": "abc123",
  "strategy": "manual-commit",
  "created": "2026-02-11T10:00:00Z",
  "updated": "2026-02-11T12:30:00Z",
  "message": "새 인증 시스템 프로토타입",
  "branch": "main",
  "shadowBranch": "entire/shadow/session-abc123"
}

// checkpoints.json
{
  "checkpoints": [
    {
      "id": "4a7b3c",
      "type": "manual",
      "timestamp": "2026-02-11T11:00:00Z",
      "message": "기본 인증 완료",
      "commit": "sha1-of-commit",
      "prompt": "prompts/001.txt",
      "response": "responses/001.txt"
    },
    {
      "id": "9e2f1d",
      "type": "manual",
      "timestamp": "2026-02-11T12:00:00Z",
      "message": "OAuth 지원 추가",
      "commit": "sha2-of-commit",
      "prompt": "prompts/002.txt",
      "response": "responses/002.txt"
    }
  ]
}
```

### 전략 선택 의사결정 트리

```
시작: 새 Entire 세션 시작
  │
  ├─ Q1: 팀 프로젝트인가?
  │   ├─ Yes → Q2: 코드 리뷰 필요?
  │   │         ├─ Yes → Manual-Commit
  │   │         └─ No → Q3으로
  │   └─ No → Q3으로
  │
  ├─ Q3: 실험적 작업인가?
  │   ├─ Yes → Q4: 여러 접근 시도?
  │   │         ├─ Yes → Manual-Commit
  │   │         └─ No → Auto-Commit
  │   └─ No → Q5로
  │
  └─ Q5: 빠른 반복 필요?
      ├─ Yes → Auto-Commit
      └─ No → Manual-Commit (기본)
```

### 실전 예시

#### 예시 1: 오픈소스 기여 (Manual-Commit)

```bash
# 1. Fork한 저장소에서 작업
cd my-forked-project
git checkout -b feature/new-api

# 2. Entire 세션 시작
entire session start --strategy manual-commit \
  --message "새 REST API 엔드포인트 추가"

# 3. AI와 개발
$ AI: /users/{id}/profile GET 엔드포인트 만들어줘
# ... shadow branch에서 작업 ...

# 4. 검토
git diff feature/new-api...entire/shadow/session-abc123

# 5. 좋으면 checkpoint
entire checkpoint create --message "프로필 API 완료"

# 6. 테스트 추가
$ AI: 이 API에 대한 단위 테스트 작성해줘
# ... 테스트 추가 ...

entire checkpoint create --message "테스트 추가"

# 7. 문서화
$ AI: README에 이 API 사용법 추가해줘
# ... 문서 업데이트 ...

entire checkpoint create --message "문서화 완료"

# 8. 최종 병합
git merge entire/shadow/session-abc123

# 9. 깔끔한 커밋 기록으로 PR 생성
git log --oneline
# abc123 문서화 완료
# def456 테스트 추가
# ghi789 프로필 API 완료
```

#### 예시 2: 긴급 버그 수정 (Auto-Commit)

```bash
# 1. 버그 리포트 받음
# "프로덕션에서 로그인 안 됨!"

# 2. 빠르게 세션 시작
entire session start --strategy auto-commit \
  --message "긴급: 로그인 버그 수정"

# 3. AI에게 즉시 요청
$ AI: 로그인 버튼 클릭 시 아무 반응 없는 버그 찾아서 고쳐줘
# ... AI가 분석하고 수정 ...
# 자동 커밋됨!

# 4. 즉시 테스트
npm test

# 5. 추가 문제 발견
$ AI: 에러 핸들링도 추가해줘
# ... 자동 커밋 ...

# 6. 배포
git push origin main

# 7. 나중에 이력 확인
entire session info
# 모든 변경 단계가 자동으로 기록됨
```

#### 예시 3: 학습 프로젝트 (Auto-Commit)

```bash
# React 학습 중
entire session start --strategy auto-commit \
  --message "React Hooks 연습"

# 단계별 학습
$ AI: useState 예제 만들어줘
$ AI: 이제 useEffect 추가해줘
$ AI: custom hook으로 리팩토링해줘
$ AI: 에러 바운더리 추가해줘

# 각 단계가 자동으로 커밋되어 학습 과정 기록
git log --oneline
# 각 학습 단계가 명확하게 기록됨
```

### 베스트 프랙티스

#### Manual-Commit 사용 시

```bash
# 1. 의미있는 checkpoint 메시지
entire checkpoint create --message "사용자 인증 로직 완료"
# 나쁜 예: entire checkpoint create --message "작업"

# 2. 정기적인 검토
git diff main...entire/shadow/session-abc123 | less

# 3. 병합 전 테스트
git checkout entire/shadow/session-abc123
npm test
git checkout main

# 4. 선택적 병합
git cherry-pick <commit-sha>  # 특정 커밋만
```

#### Auto-Commit 사용 시

```bash
# 1. 작은 단위로 작업
$ AI: 먼저 타입 정의만 추가해줘
$ AI: 이제 구현부 추가해줘
$ AI: 마지막으로 테스트 추가해줘

# 2. 정기적인 백업 checkpoint
entire checkpoint create --message "작업 중간 저장"

# 3. 실수 시 빠른 되돌리기
git reset HEAD~1  # 마지막 커밋 취소

# 4. 주기적인 정리
git rebase -i HEAD~10  # 커밋 정리
```

### 다음 장 예고

다음 장에서는 **Manual-Commit Strategy**의 핵심인 Shadow Branch 메커니즘을 상세히 다룹니다.

- Shadow branch 생성 및 관리
- Worktree 기반 격리
- 병합 전략
- Conflict 해결

---

**관련 문서**:
- [07. Manual-Commit Strategy](/2026/02/11/entire-cli-guide-07-manual-commit/)
- [08. Auto-Commit Strategy](/2026/02/11/entire-cli-guide-08-auto-commit/)
- [09. Session 관리](/2026/02/11/entire-cli-guide-09-session-management/)
