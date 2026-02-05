---
layout: post
title: "Context Engineering 완벽 가이드 (10) - 실전 적용"
date: 2026-02-05
permalink: /context-engineering-guide-10-practical/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, CLAUDE.md, Practical, Best Practices, Implementation]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "CLAUDE.md 작성법과 실전 적용 방법을 알아봅니다."
---

## CLAUDE.md - Cognitive Operating System

CLAUDE.md는 Claude Code를 위한 **인지 운영 체제**입니다. 프로젝트 루트에 배치하여 AI의 동작을 정의합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    CLAUDE.md Structure                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. Core Meta-Cognitive Framework                   │    │
│  │     • Context Schemas                               │    │
│  │     • Reasoning Protocols                           │    │
│  │     • Self-Improvement Protocol                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  2. Workflow Protocols                               │    │
│  │     • Explore-Plan-Code-Commit                      │    │
│  │     • Test-Driven Development                       │    │
│  │     • UI Iteration                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  3. Code Analysis & Generation                       │    │
│  │     • Analysis Protocol                              │    │
│  │     • Generation Protocol                            │    │
│  │     • Refactoring Protocol                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  4. Project-Specific Conventions                     │    │
│  │     • Bash Commands                                  │    │
│  │     • Code Style                                     │    │
│  │     • Git Workflow                                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## CLAUDE.md 작성 예시

### 기본 템플릿

```markdown
# CLAUDE.md

## Project Overview
Brief description of what this project does.

## Tech Stack
- Language: Python 3.11+
- Framework: FastAPI
- Database: PostgreSQL
- Testing: pytest

## Development Commands
- `make dev`: Start development server
- `make test`: Run all tests
- `make lint`: Run linter

## Code Style
- Use type hints for all functions
- Follow PEP 8
- Maximum line length: 88 (Black)
- Docstrings: Google style

## Git Workflow
- Branch naming: `feature/`, `fix/`, `refactor/`
- Commit messages: Conventional Commits
- Always squash merge to main

## Architecture
```
src/
├── api/          # FastAPI routes
├── core/         # Business logic
├── models/       # Database models
├── services/     # External services
└── utils/        # Helpers
```

## Important Notes
- Never commit secrets to git
- Run tests before committing
- Update docs when changing APIs
```

### 고급 템플릿 (Cognitive Tools 포함)

```markdown
# CLAUDE.md - Cognitive Operating System

## 1. Core Meta-Cognitive Framework

### Reasoning Protocols

```
/reasoning.systematic{
    intent="Break down complex problems",
    process=[
        /understand{action="Restate problem"},
        /analyze{action="Break into components"},
        /plan{action="Design approach"},
        /execute{action="Implement"},
        /verify{action="Validate"}
    ]
}
```

### Self-Improvement Protocol

```
/self.reflect{
    intent="Continuously improve",
    process=[
        /assess{completeness, correctness, clarity},
        /identify{strengths, weaknesses},
        /improve{strategy, implementation}
    ]
}
```

## 2. Workflow Protocols

### Default Workflow: Explore-Plan-Code-Commit

1. **Explore**: Read relevant files, understand context
2. **Plan**: Create detailed plan with extended thinking
3. **Code**: Implement following the plan
4. **Commit**: Clean commits with descriptive messages

### When to use TDD

Use Test-Driven Development when:
- Adding new features with clear requirements
- Fixing bugs (write failing test first)
- Refactoring critical code

## 3. Project-Specific Conventions

### Commands
```bash
npm run build    # Build project
npm run test     # Run tests
npm run lint     # Lint code
npm run dev      # Development server
```

### Code Style
- Indentation: 2 spaces
- Semicolons: No
- Quotes: Single
- Trailing commas: ES5

### Git
- Feature branches from `main`
- Squash merge PRs
- Sign commits
```

---

## 컨텍스트 최적화 전략

### 1. 토큰 예산 관리

```
┌─────────────────────────────────────────────────────────────┐
│                    Token Budget Allocation                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Total Budget: 128K tokens                                  │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ System Prompt      ████░░░░░░░░░░░░░░░░░░  10%     │   │
│  │ CLAUDE.md          ██████░░░░░░░░░░░░░░░░  15%     │   │
│  │ Context (files)    ████████████░░░░░░░░░░  35%     │   │
│  │ Conversation       ██████████░░░░░░░░░░░░  25%     │   │
│  │ Tool Results       ████░░░░░░░░░░░░░░░░░░  10%     │   │
│  │ Reserved           ██░░░░░░░░░░░░░░░░░░░░  5%      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  Tips:                                                       │
│  • CLAUDE.md는 간결하게                                     │
│  • 필요한 파일만 컨텍스트에 포함                            │
│  • 긴 대화는 요약                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. 컨텍스트 계층화

```markdown
## Context Hierarchy

### Level 1: Always Present (CLAUDE.md)
- Project basics
- Core conventions
- Key commands

### Level 2: Task-Specific (Dynamic)
- Relevant source files
- Related tests
- Documentation

### Level 3: On-Demand (Retrieved)
- Deep knowledge base
- Historical context
- External docs
```

### 3. 프라이밍 기법

```markdown
## Priming Examples

Before asking for code, prime with examples:

### Example 1
Input: Create a user service
Output:
```python
class UserService:
    def __init__(self, db: Database):
        self.db = db

    async def get_user(self, user_id: int) -> User:
        return await self.db.users.get(user_id)
```

### Example 2
Input: Add validation
Output:
```python
from pydantic import validator

class CreateUserRequest(BaseModel):
    email: EmailStr
    name: str

    @validator('name')
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v
```

### Now Your Task
...
```

---

## 실전 적용 체크리스트

### 프로젝트 설정

- [ ] CLAUDE.md 생성
- [ ] 프로젝트 개요 작성
- [ ] 기술 스택 명시
- [ ] 개발 명령어 정리
- [ ] 코드 스타일 가이드
- [ ] Git 워크플로우

### 컨텍스트 최적화

- [ ] 필수 정보만 포함
- [ ] 중복 제거
- [ ] 계층 구조화
- [ ] 예제 포함
- [ ] 프로토콜 정의

### 워크플로우 정의

- [ ] 기본 워크플로우 선택
- [ ] 예외 케이스 처리
- [ ] 검증 단계 포함
- [ ] 롤백 절차

---

## 일반적인 실수와 해결책

| 실수 | 해결책 |
|------|--------|
| CLAUDE.md가 너무 김 | 핵심만 남기고 상세는 별도 파일 |
| 모순되는 지시문 | 우선순위 명시 |
| 모호한 요구사항 | 구체적 예시 추가 |
| 컨텍스트 부족 | 관련 파일 명시적 포함 |
| 컨텍스트 과다 | 필요한 것만 동적 로드 |

---

## 마무리

### Context Engineering의 핵심

```
┌─────────────────────────────────────────────────────────────┐
│              Context Engineering Mastery                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. First Principles                                         │
│     "프롬프트가 아니라 전체 컨텍스트를 설계한다"             │
│                                                              │
│  2. Measure Everything                                       │
│     "토큰, 지연, 품질을 측정한다"                            │
│                                                              │
│  3. Delete Ruthlessly                                        │
│     "불필요한 것은 제거한다"                                 │
│                                                              │
│  4. Iterate Continuously                                     │
│     "계속 개선한다"                                          │
│                                                              │
│  5. Leverage Structure                                       │
│     "구조화된 형식을 활용한다"                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 다음 단계

1. **[GitHub 레포지토리](https://github.com/davidkimai/Context-Engineering)** 탐색
2. **00_foundations/** 이론 학습
3. **20_templates/** 템플릿 활용
4. **cognitive-tools/** 인지 도구 실험
5. 자신만의 CLAUDE.md 작성

---

## 참고 자료

- [Context Engineering Survey (1400+ papers)](https://arxiv.org/pdf/2507.13334)
- [IBM Zurich Cognitive Tools](https://arxiv.org/pdf/2506.12115)
- [MEM1: Memory Systems](https://arxiv.org/pdf/2506.15841)
- [Emergent Symbolic Mechanisms](https://openreview.net/forum?id=y1SnRPDWx4)
- [Quantum Semantics](https://arxiv.org/pdf/2506.10077)

---

*이 가이드가 Context Engineering 여정에 도움이 되길 바랍니다!*
