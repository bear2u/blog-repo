---
layout: post
title: "Context Engineering 완벽 가이드 (3) - Neural Systems & Cognitive Tools"
date: 2026-02-05
permalink: /context-engineering-guide-03-cognitive-tools/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, Cognitive Tools, IBM Zurich, Reasoning, Mental Models]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "IBM Zurich 연구 기반의 인지 도구와 추론 프레임워크를 알아봅니다."
---

## Neural Systems 개요

**Neural Systems**는 Organs 위에 구축되어 인지 도구와 정신 모델을 제공합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Neural Systems                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Cognitive Tools Layer                   │    │
│  │                                                      │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │Understand│ │ Recall  │ │ Examine │ │Backtrack│   │    │
│  │  │Question │ │Related  │ │ Answer  │ │         │   │    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Reasoning Frameworks                    │    │
│  │                                                      │    │
│  │  Chain-of-Thought │ ReAct │ Self-Reflection         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## IBM Zurich 연구: Cognitive Tools

> **"Providing 'cognitive tools' to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%"**

IBM Zurich의 2025년 연구는 **인지 도구(Cognitive Tools)**가 LLM의 추론 능력을 획기적으로 향상시킴을 증명했습니다.

### 핵심 발견

1. **모듈러 추론**: 복잡한 작업을 모듈화된 "인지 도구"로 분해
2. **프롬프트 프로그램**: 구조화된 프롬프트 템플릿을 도구 호출처럼 사용
3. **휴리스틱 스캐폴딩**: 인간의 정신적 지름길을 모방

```
┌─────────────────────────────────────────────────────────────┐
│                 Cognitive Tools 아키텍처                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  문제 입력                                                   │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  understand_question                                  │  │
│  │  • 주요 개념 식별                                     │  │
│  │  • 관련 정보 추출                                     │  │
│  │  • 속성/정리/기법 하이라이트                          │  │
│  └───────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  recall_related                                       │  │
│  │  • 관련 개념 회상                                     │  │
│  │  • 유사 문제 패턴 인식                                │  │
│  │  • 적용 가능한 기법 목록화                            │  │
│  └───────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  solve_step_by_step                                   │  │
│  │  • 단계별 추론 수행                                   │  │
│  │  • 중간 결과 검증                                     │  │
│  └───────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  examine_answer                                       │  │
│  │  • 답변 검증                                          │  │
│  │  • 에지 케이스 확인                                   │  │
│  │  • 오류 감지                                          │  │
│  └───────────────────────────────────────────────────────┘  │
│      │                                                       │
│      ▼ (오류 발견 시)                                        │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  backtrack                                            │  │
│  │  • 이전 단계로 복귀                                   │  │
│  │  • 대안 접근법 시도                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Cognitive Tools 구현

### 1. understand_question

```python
UNDERSTAND_QUESTION_TEMPLATE = """
## Cognitive Tool: understand_question

Given the problem below, perform deep analysis:

### Problem:
{problem}

### Analysis Tasks:
1. **Identify Main Concepts**: What are the key mathematical/logical concepts?
2. **Extract Relevant Information**: What data is given? What is asked?
3. **Highlight Properties**: What theorems, properties, or techniques might apply?
4. **Clarify Constraints**: What are the boundaries or limitations?

### Output Format:
- Concepts: [list]
- Given: [structured data]
- Find: [what to solve]
- Applicable Techniques: [list]
- Constraints: [list]
"""

def understand_question(problem: str) -> dict:
    prompt = UNDERSTAND_QUESTION_TEMPLATE.format(problem=problem)
    response = llm.generate(prompt)
    return parse_understanding(response)
```

### 2. recall_related

```python
RECALL_RELATED_TEMPLATE = """
## Cognitive Tool: recall_related

Based on the problem analysis, recall related knowledge:

### Problem Analysis:
{understanding}

### Recall Tasks:
1. **Similar Problems**: What similar problems have you seen?
2. **Relevant Theorems**: What mathematical theorems apply?
3. **Solution Patterns**: What solution approaches are commonly used?
4. **Common Pitfalls**: What mistakes are commonly made?

### Output Format:
- Similar Problems: [examples]
- Key Theorems: [list with brief descriptions]
- Solution Strategies: [ordered by likelihood of success]
- Pitfalls to Avoid: [list]
"""
```

### 3. examine_answer

```python
EXAMINE_ANSWER_TEMPLATE = """
## Cognitive Tool: examine_answer

Critically examine the proposed solution:

### Solution:
{solution}

### Examination Tasks:
1. **Verify Logic**: Is each step logically sound?
2. **Check Calculations**: Are all calculations correct?
3. **Test Edge Cases**: Does the solution handle edge cases?
4. **Validate Format**: Is the answer in the required format?

### Output Format:
- Logic Check: [pass/fail with notes]
- Calculation Check: [pass/fail with notes]
- Edge Cases: [list with results]
- Format Check: [pass/fail]
- Confidence: [0-100%]
- Suggested Fixes: [if any]
"""
```

### 4. backtrack

```python
BACKTRACK_TEMPLATE = """
## Cognitive Tool: backtrack

The previous approach failed. Let's backtrack and try again.

### Failed Approach:
{failed_approach}

### Failure Reason:
{failure_reason}

### Backtrack Tasks:
1. **Identify Error Point**: Where did the reasoning go wrong?
2. **Alternative Approaches**: What other methods could work?
3. **Lessons Learned**: What should we avoid this time?
4. **New Strategy**: What's the next best approach?

### Output Format:
- Error Point: [step number and description]
- Root Cause: [analysis]
- Alternative Strategies: [ranked list]
- Selected Strategy: [choice with rationale]
- Implementation Plan: [steps]
"""
```

---

## 추론 프레임워크

### Chain-of-Thought (CoT)

```
┌─────────────────────────────────────────────────────────────┐
│                  Chain-of-Thought                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  문제: 사과가 5개 있고 3개를 먹었다면 남은 것은?              │
│                                                              │
│  추론 과정:                                                  │
│  Step 1: 처음 사과 개수 = 5개                                │
│  Step 2: 먹은 사과 개수 = 3개                                │
│  Step 3: 남은 사과 = 처음 - 먹은 = 5 - 3 = 2개               │
│                                                              │
│  답: 2개                                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Tree-of-Thought (ToT)

```
                      문제
                        │
          ┌─────────────┼─────────────┐
          │             │             │
       접근법 A      접근법 B      접근법 C
          │             │             │
       ┌──┴──┐       ┌──┴──┐       ┌──┴──┐
       │     │       │     │       │     │
      A1    A2      B1    B2      C1    C2
       │             │                   │
      ✗            ✓ ←────────── 최적 경로
```

### Self-Reflection

```python
SELF_REFLECT_TEMPLATE = """
## Self-Reflection Protocol

### Previous Output:
{output}

### Reflection Questions:
1. Is this answer complete?
2. Are there any logical errors?
3. Did I miss any edge cases?
4. Could the explanation be clearer?
5. What would I change if I did this again?

### Self-Assessment:
- Completeness: [1-5]
- Correctness: [1-5]
- Clarity: [1-5]

### Improvements:
[list specific improvements]

### Revised Output:
[if needed]
"""
```

---

## Prompt Programming

프롬프트를 코드처럼 구조화하는 기법입니다.

### 프로토콜 문법

```
/reasoning.systematic{
    intent="Break down complex problems into logical steps",
    input={
        problem="<problem_statement>",
        constraints="<constraints>",
        context="<context>"
    },
    process=[
        /understand{action="Restate problem and clarify goals"},
        /analyze{action="Break down into components"},
        /plan{action="Design step-by-step approach"},
        /execute{action="Implement solution methodically"},
        /verify{action="Validate against requirements"},
        /refine{action="Improve based on verification"}
    ],
    output={
        solution="Implemented solution",
        reasoning="Complete reasoning trace",
        verification="Validation evidence"
    }
}
```

### 확장된 사고 프로토콜

```
/thinking.extended{
    intent="Engage deep reasoning for complex problems",
    input={
        problem="<problem>",
        level="<basic|deep|deeper|ultra>"
    },
    process=[
        /explore{action="Consider multiple perspectives"},
        /evaluate{action="Assess trade-offs"},
        /simulate{action="Test mental models"},
        /synthesize{action="Integrate insights"},
        /articulate{action="Express reasoning clearly"}
    ],
    output={
        conclusion="Well-reasoned solution",
        rationale="Complete thinking process",
        alternatives="Other considered approaches"
    }
}
```

---

## 실전 예시: 수학 문제 해결

```python
async def solve_math_problem(problem: str):
    # 1. 문제 이해
    understanding = await cognitive_tool("understand_question", problem)

    # 2. 관련 지식 회상
    knowledge = await cognitive_tool("recall_related", understanding)

    # 3. 단계별 풀이
    solution = await cognitive_tool("solve_step_by_step", {
        "problem": problem,
        "understanding": understanding,
        "knowledge": knowledge
    })

    # 4. 답안 검증
    examination = await cognitive_tool("examine_answer", solution)

    # 5. 필요시 백트래킹
    if examination["confidence"] < 80:
        return await cognitive_tool("backtrack", {
            "failed_approach": solution,
            "failure_reason": examination["issues"]
        })

    return solution
```

---

## 성능 향상 결과

| 모델 | 기본 성능 | + Cognitive Tools | 향상률 |
|------|-----------|-------------------|--------|
| GPT-4.1 | 26.7% | 43.3% | +62% |
| Claude | 31.2% | 48.7% | +56% |
| Gemini | 28.5% | 44.1% | +55% |

> **핵심 인사이트**: 강력한 추론의 씨앗은 이미 LLM 안에 있습니다. Cognitive Tools는 이러한 능력을 잠금 해제하고 오케스트레이션합니다.

---

*다음 글에서는 Neural Field Theory를 살펴봅니다.*
