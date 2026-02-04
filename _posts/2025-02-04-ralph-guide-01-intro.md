---
layout: post
title: "Ralph 가이드 01 - 소개"
date: 2025-02-04
categories: [AI, Claude Code, Ralph]
tags: [ralph, claude-code, autonomous-development, ai-loop]
series: "ralph-guide"
permalink: /ralph-guide-01-intro/
---

# Ralph 소개

## Ralph란?

**Ralph**는 Geoffrey Huntley의 "Ralph 기법"을 구현한 오픈소스 도구로, Claude Code를 활용한 자율적인 연속 개발 사이클을 가능하게 합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      Ralph 개발 루프                         │
│                                                              │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│    │  Read    │ -> │ Implement│ -> │   Test   │            │
│    │ PROMPT   │    │   Task   │    │  & Fix   │            │
│    └──────────┘    └──────────┘    └──────────┘            │
│         ▲                               │                   │
│         │         ┌──────────┐          │                   │
│         └─────────│  Update  │<─────────┘                   │
│                   │ fix_plan │                              │
│                   └──────────┘                              │
│                        │                                     │
│                   (반복 또는 종료)                            │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 특징

| 특징 | 설명 |
|------|------|
| **자율 개발** | 프로젝트 완료까지 자동으로 반복 실행 |
| **지능형 종료** | Dual-condition 체크로 정확한 완료 감지 |
| **안전장치** | 무한 루프와 API 남용 방지 내장 |
| **모니터링** | 실시간 진행 상황 대시보드 |

## Geoffrey Huntley의 Ralph 기법

Ralph 기법은 Geoffrey Huntley가 고안한 자율적 AI 개발 방법론입니다. 핵심 아이디어는 AI가 스스로 작업을 계획하고, 실행하고, 검증하는 사이클을 반복하도록 하는 것입니다.

### 기존 개발 방식과의 비교

**기존 AI 보조 개발:**
```
개발자 → AI에게 요청 → AI 응답 → 개발자 검토 → 다시 요청 → ...
(반복적인 수동 개입 필요)
```

**Ralph 기법:**
```
개발자 → 요구사항 정의 → Ralph 실행 → (자율적 개발) → 완료된 프로젝트
(한 번 설정 후 자율 실행)
```

### 원칙

1. **명확한 목표 정의**: PROMPT.md에 프로젝트 비전 기술
2. **구체적인 작업 목록**: fix_plan.md에 체크리스트 작성
3. **자율적 실행**: AI가 스스로 판단하며 작업 수행
4. **지능적 종료**: 모든 작업 완료 시 자동 종료

## 왜 Ralph인가?

### 문제점: 기존 AI 개발의 한계

1. **컨텍스트 손실**: 세션이 끊기면 처음부터 다시 설명
2. **반복 작업**: 같은 지시를 계속 반복
3. **일관성 부족**: 매번 다른 방식으로 구현
4. **감독 필요**: 지속적인 개발자 개입

### 해결책: Ralph의 접근

```javascript
// Ralph의 자율 루프 개념
while (!projectComplete) {
  const context = readPromptAndFixPlan();
  const nextTask = findNextUncheckedTask();

  if (!nextTask) {
    if (allTasksComplete() && testsPass()) {
      setExitSignal();  // 프로젝트 완료
    }
    continue;
  }

  implementTask(nextTask);
  runTests();
  updateFixPlan(nextTask, 'completed');
}
```

### Ralph의 장점

| 장점 | 설명 |
|------|------|
| **일관성** | 동일한 컨텍스트로 일관된 개발 |
| **효율성** | 개발자 개입 없이 연속 작업 |
| **추적성** | 모든 작업 이력 기록 |
| **안전성** | 서킷 브레이커로 이상 상황 감지 |

## 어떤 프로젝트에 적합한가?

### 적합한 경우

- **새 프로젝트 시작**: 처음부터 구조를 잡아가는 프로젝트
- **명확한 요구사항**: 구체적으로 정의된 기능 목록
- **반복적인 작업**: CRUD, API 엔드포인트 구현 등
- **프로토타이핑**: 빠른 MVP 개발

### 부적합한 경우

- **복잡한 아키텍처 결정**: 인간의 판단이 필요한 설계
- **레거시 시스템 통합**: 깊은 도메인 지식 필요
- **실시간 협업**: 다른 개발자와 동시 작업

## 빠른 시작

```bash
# 1. Ralph 설치 (한 번만)
git clone https://github.com/frankbria/ralph-claude-code.git
cd ralph-claude-code
./install.sh

# 2. 프로젝트에서 활성화
cd my-project
ralph-enable

# 3. 요구사항 편집
vim .ralph/PROMPT.md
vim .ralph/fix_plan.md

# 4. 자율 개발 시작
ralph --monitor
```

## 다음 단계

이제 Ralph의 기본 개념을 이해했으니, 다음 장에서 실제 설치와 설정 방법을 알아봅니다.

---

**다음 장:** [설치 및 시작](/ralph-guide-02-installation/)
