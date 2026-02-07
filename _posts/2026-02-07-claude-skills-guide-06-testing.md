---
layout: post
title: "Claude Skills 완벽 가이드 (06) - 테스트 및 반복 개선"
date: 2026-02-07
permalink: /claude-skills-guide-06-testing/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Testing, Iteration, Quality Assurance]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "스킬을 효과적으로 테스트하고 개선하는 방법"
---

## 테스트 수준

스킬은 요구사항에 따라 다양한 수준으로 테스트할 수 있습니다:

### 1. 수동 테스트 (Claude.ai)
- ✅ 빠른 반복
- ✅ 설정 불필요
- ✅ 직접 쿼리 실행 및 동작 관찰

### 2. 스크립트 테스트 (Claude Code)
- ✅ 반복 가능한 검증
- ✅ 변경 사항에 대한 자동화된 테스트
- ✅ 여러 테스트 케이스 관리

### 3. 프로그래매틱 테스트 (Skills API)
- ✅ 체계적인 평가 스위트
- ✅ 정의된 테스트 세트에 대한 실행
- ✅ CI/CD 파이프라인 통합

---

## 권장 접근법

> **Pro Tip:** 확장하기 전에 하나의 작업에 집중하세요.
>
> 가장 효과적인 스킬 제작자들은 **어려운 단일 작업에서 Claude가 성공할 때까지 반복**합니다. 그런 다음 성공한 접근 방식을 스킬로 추출합니다.
>
> 이는 Claude의 컨텍스트 내 학습을 활용하며 광범위한 테스트보다 빠른 피드백을 제공합니다. 작동하는 기초가 생기면 여러 테스트 케이스로 확장합니다.

---

## 1. 트리거 테스트

**목표:** 스킬이 적절한 시기에 로드되는지 확인

### 테스트 케이스

```
✅ 명확한 작업에 트리거되어야 함:
- "Help me set up a new ProjectHub workspace"
- "I need to create a project in ProjectHub"
- "Initialize a ProjectHub project for Q4 planning"

✅ 변형된 표현에도 트리거되어야 함:
- "Can you make a new workspace in ProjectHub?"
- "Start a Q4 project using ProjectHub"
- "Set up ProjectHub for our team"

❌ 관련 없는 주제에는 트리거되지 말아야 함:
- "What's the weather in San Francisco?"
- "Help me write Python code"
- "Create a spreadsheet" (ProjectHub가 스프레드시트를 다루지 않는다면)
```

---

### 테스트 스크립트 예시

```python
# test_triggering.py
test_cases = [
    # Should trigger
    ("Help me set up a new ProjectHub workspace", True),
    ("I need to create a project in ProjectHub", True),
    ("Initialize a ProjectHub project for Q4 planning", True),

    # Should NOT trigger
    ("What's the weather?", False),
    ("Help me write Python code", False),
    ("Create a spreadsheet", False),
]

for query, should_trigger in test_cases:
    result = test_skill_trigger("projecthub-setup", query)
    assert result == should_trigger, f"Failed for: {query}"
```

---

## 2. 기능 테스트

**목표:** 스킬이 올바른 출력을 생성하는지 확인

### 테스트 영역

#### A) 유효한 출력 생성
```python
def test_project_creation():
    """Test: Create project with 5 tasks"""

    # Given
    project_name = "Q4 Planning"
    tasks = [
        "Define goals",
        "Assign team",
        "Set milestones",
        "Create timeline",
        "Review resources"
    ]

    # When
    result = run_skill("projecthub-setup", {
        "project_name": project_name,
        "tasks": tasks
    })

    # Then
    assert result.project_created == True
    assert len(result.tasks) == 5
    assert all(task.linked_to_project for task in result.tasks)
    assert result.errors == []
```

---

#### B) API 호출 성공
```python
def test_api_integration():
    """Test: All MCP calls succeed"""

    result = run_skill("linear-sprint-planner", {
        "sprint_name": "Sprint 23"
    })

    # Verify no API errors
    assert result.api_calls_failed == 0
    assert result.sprint_created == True
    assert result.tasks_created > 0
```

---

#### C) 에러 처리
```python
def test_error_handling():
    """Test: Graceful error handling"""

    # Test missing required field
    result = run_skill("projecthub-setup", {
        "project_name": None  # Missing required field
    })

    assert result.success == False
    assert "project_name is required" in result.error_message

    # Test invalid input
    result = run_skill("projecthub-setup", {
        "project_name": "",  # Empty string
        "tasks": []          # No tasks
    })

    assert result.success == False
    assert "at least one task" in result.error_message
```

---

#### D) 엣지 케이스
```python
def test_edge_cases():
    """Test: Handle edge cases"""

    # Very long project name
    result = run_skill("projecthub-setup", {
        "project_name": "A" * 500  # 500 characters
    })
    assert result.project_name_truncated == True

    # Maximum tasks
    result = run_skill("projecthub-setup", {
        "tasks": ["Task " + str(i) for i in range(100)]
    })
    assert len(result.tasks) <= 100

    # Special characters
    result = run_skill("projecthub-setup", {
        "project_name": "Q4 Planning (2025) — Main 🚀"
    })
    assert result.project_created == True
```

---

## 3. 성능 비교

**목표:** 스킬이 기준선 대비 결과를 개선하는지 증명

### 스킬 없이 (Baseline)

```
사용자 경험:
- 매번 처음부터 설명 필요
- 15번의 대화 왕복
- 3번의 실패한 API 호출 (재시도 필요)
- 12,000 토큰 소비
- 소요 시간: 5분

결과 품질:
- 일관성 없음
- 단계 누락 발생
- 베스트 프랙티스 미적용
```

---

### 스킬 사용

```
사용자 경험:
- 자동 워크플로우 실행
- 2번의 확인 질문만
- 0번의 실패한 API 호출
- 6,000 토큰 소비
- 소요 시간: 2분

결과 품질:
- 일관된 품질
- 모든 단계 완료
- 베스트 프랙티스 자동 적용
```

---

### 비교 메트릭

| 메트릭 | 스킬 없이 | 스킬 사용 | 개선 |
|--------|----------|---------|------|
| 대화 왕복 | 15회 | 2회 | **87% 감소** |
| API 실패 | 3회 | 0회 | **100% 감소** |
| 토큰 사용 | 12,000 | 6,000 | **50% 감소** |
| 소요 시간 | 5분 | 2분 | **60% 단축** |
| 일관성 | 낮음 | 높음 | **향상** |

---

## skill-creator 스킬 사용하기

**skill-creator** 스킬은 스킬 개발을 도와주는 메타 스킬입니다.

### 사용 가능 위치
- Claude.ai (플러그인 디렉토리에서)
- Claude Code (다운로드 후 사용)

---

### 스킬 생성

```
사용자: "Use the skill-creator skill to help me build a skill for
automating Linear sprint planning"

skill-creator:
- 자연어 설명에서 스킬 생성
- 적절한 형식의 SKILL.md와 frontmatter 생성
- 트리거 문구와 구조 제안
```

---

### 스킬 리뷰

```
사용자: "Review my skill and suggest improvements"

skill-creator:
- 일반적인 문제 플래깅:
  • 모호한 description
  • 트리거 조건 누락
  • 구조적 문제
- 과도/과소 트리거 위험 식별
- 스킬의 목적에 맞는 테스트 케이스 제안
```

---

### 반복 개선

```
사용자: "Use the issues & solution identified in this chat to
improve how the skill handles rate limiting errors"

skill-creator:
- 엣지 케이스나 실패 사례를 기반으로 개선
- 에러 처리 추가
- 명령어 명확화
```

---

### 제한사항

**skill-creator는:**
- ✅ 스킬 설계 및 개선 지원
- ❌ 자동화된 테스트 스위트 실행 불가
- ❌ 정량적 평가 결과 생성 불가

---

## 피드백 기반 반복

스킬은 **살아있는 문서**입니다. 다음을 기반으로 반복 개선하세요:

---

### A) Under-Triggering 신호

**증상:**
- 사용되어야 할 때 스킬이 로드되지 않음
- 사용자가 수동으로 활성화함
- "언제 사용하나요?" 지원 질문 증가

**해결책:** Description에 더 많은 세부사항과 뉘앙스 추가

```yaml
# Before
description: Helps with Linear projects

# After
description: End-to-end Linear sprint planning including task creation, team
  assignment, milestone setup, and notifications. Use when user says "plan sprint",
  "create Linear sprint", "set up iteration", "organize Linear tasks", or
  "start new sprint".
```

**추가 키워드:**
- 기술 용어 포함
- 동의어 추가
- 파일 확장자 명시 (해당되는 경우)

---

### B) Over-Triggering 신호

**증상:**
- 관련 없는 쿼리에도 스킬이 로드됨
- 사용자가 스킬을 비활성화함
- 목적에 대한 혼란

**해결책:** 더 구체적으로, 부정 트리거 추가

```yaml
# Before (너무 광범위)
description: Manages projects and tasks

# After (구체적)
description: Manages Linear sprint planning specifically for engineering teams.
  Use when user mentions "Linear sprint", "sprint planning", or "create sprint".
  Do NOT use for general task management, calendar events, or non-Linear tools.
```

---

### C) 실행 문제

**증상:**
- 일관성 없는 결과
- API 호출 실패
- 사용자 수정 필요

**해결책:** 명령어 개선, 에러 처리 추가

```markdown
# Before
## Step 3: Create tasks
Create the tasks in Linear.

# After
## Step 3: Create tasks

For each task:
1. Validate required fields (title, description, estimate)
2. Call Linear API:
   ```bash
   mcp-tool call linear create_issue \
     --title "${task.title}" \
     --description "${task.description}" \
     --estimate ${task.estimate}
   ```
3. If API fails with 429 (rate limit):
   - Wait 60 seconds
   - Retry once
   - If still fails, queue for later
4. If API fails with 401 (auth):
   - Check API key validity
   - Prompt user to reconnect Linear MCP
```

---

## 테스트 체크리스트

### 트리거 테스트
- [ ] 명확한 작업에 트리거되는가?
- [ ] 변형된 표현에도 트리거되는가?
- [ ] 관련 없는 주제에는 트리거되지 않는가?

### 기능 테스트
- [ ] 유효한 출력을 생성하는가?
- [ ] API 호출이 성공하는가?
- [ ] 에러 처리가 작동하는가?
- [ ] 엣지 케이스를 처리하는가?

### 성능 테스트
- [ ] 기준선 대비 개선되었는가?
- [ ] 토큰 사용량이 줄었는가?
- [ ] 사용자 경험이 향상되었는가?
- [ ] 결과가 일관적인가?

### 품질 테스트
- [ ] 명령어가 명확한가?
- [ ] 예시가 충분한가?
- [ ] 문서가 완전한가?
- [ ] 에러 메시지가 도움이 되는가?

---

## 다음 단계

테스트가 완료되었다면:

1. 스킬 배포 준비
2. 배포 채널 선택
3. 사용자 피드백 수집
4. 지속적 개선

---

*다음 글에서는 스킬 배포 및 공유 방법을 다룹니다.*
