---
layout: post
title: "Context Engineering 완벽 가이드 (9) - Protocols & Templates"
date: 2026-02-05
permalink: /context-engineering-guide-09-protocols/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, Protocols, Templates, YAML, Schemas]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "재사용 가능한 프로토콜과 템플릿 설계 방법을 알아봅니다."
---

## Protocols & Templates 개요

프로토콜과 템플릿은 Context Engineering의 **재사용 가능한 빌딩 블록**입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Protocol System                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Protocol Shell                      │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Intent: What this protocol achieves         │    │    │
│  │  │ Input:  What it needs                       │    │    │
│  │  │ Process: Steps to execute                   │    │    │
│  │  │ Output: What it produces                    │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                    Templates                         │    │
│  │  • Context Templates                                │    │
│  │  • Prompt Templates                                 │    │
│  │  • Output Templates                                 │    │
│  │  • Workflow Templates                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Protocol Shell 구조

### 기본 문법

```
/protocol.name{
    intent="Protocol purpose",
    input={
        param1="description",
        param2="description"
    },
    process=[
        /step1{action="..."},
        /step2{action="..."},
        /step3{action="..."}
    ],
    output={
        result="description",
        metadata="description"
    }
}
```

### 예시: 체계적 추론 프로토콜

```
/reasoning.systematic{
    intent="Break down complex problems into logical steps",
    input={
        problem="<problem_statement>",
        constraints="<constraints>",
        context="<context>"
    },
    process=[
        /understand{
            action="Restate problem and clarify goals",
            output="problem_restatement"
        },
        /analyze{
            action="Break down into components",
            output="component_list"
        },
        /plan{
            action="Design step-by-step approach",
            output="execution_plan"
        },
        /execute{
            action="Implement solution methodically",
            output="solution"
        },
        /verify{
            action="Validate against requirements",
            output="verification_result"
        },
        /refine{
            action="Improve based on verification",
            condition="if verification fails",
            output="refined_solution"
        }
    ],
    output={
        solution="Implemented solution",
        reasoning="Complete reasoning trace",
        verification="Validation evidence"
    }
}
```

---

## 내장 프로토콜 라이브러리

### 1. 코드 분석 프로토콜

```
/code.analyze{
    intent="Deeply understand code structure and quality",
    input={
        code="<code_to_analyze>",
        focus="<specific_aspects>"
    },
    process=[
        /parse{
            structure="Identify main components",
            patterns="Recognize design patterns",
            flow="Trace execution paths"
        },
        /evaluate{
            quality="Assess code quality",
            performance="Identify bottlenecks",
            security="Spot vulnerabilities",
            maintainability="Evaluate long-term"
        },
        /summarize{
            purpose="Describe functionality",
            architecture="Outline approach",
            interfaces="Document contracts"
        }
    ],
    output={
        overview="High-level summary",
        details="Component breakdown",
        recommendations="Suggested improvements"
    }
}
```

### 2. 버그 진단 프로토콜

```
/bug.diagnose{
    intent="Systematically identify root causes",
    input={
        symptoms="<observed_problem>",
        context="<environment_and_conditions>"
    },
    process=[
        /reproduce{
            steps="Establish reproduction steps",
            environment="Identify env factors",
            consistency="Determine reproducibility"
        },
        /isolate{
            scope="Narrow down components",
            triggers="Identify specific triggers",
            patterns="Recognize symptom patterns"
        },
        /analyze{
            trace="Follow execution path",
            state="Examine relevant state",
            interactions="Study component interactions"
        },
        /hypothesize{
            causes="Formulate potential causes",
            tests="Design tests for each",
            verification="Plan verification approach"
        }
    ],
    output={
        diagnosis="Identified root cause",
        evidence="Supporting evidence",
        fix_strategy="Recommended solution"
    }
}
```

### 3. 자기 성찰 프로토콜

```
/self.reflect{
    intent="Continuously improve through evaluation",
    input={
        previous_output="<output_to_evaluate>",
        criteria="<evaluation_criteria>"
    },
    process=[
        /assess{
            completeness="Identify missing info",
            correctness="Verify accuracy",
            clarity="Evaluate understandability",
            effectiveness="Determine if meets needs"
        },
        /identify{
            strengths="Note what was done well",
            weaknesses="Recognize limitations",
            assumptions="Surface implicit assumptions"
        },
        /improve{
            strategy="Plan improvements",
            implementation="Apply methodically"
        }
    ],
    output={
        evaluation="Assessment of original",
        improved_output="Enhanced version",
        learning="Insights for future"
    }
}
```

---

## Context Templates

### 최소 컨텍스트 템플릿

```yaml
# minimal_context.yaml
version: "1.0"
name: "minimal_context"
description: "Bare minimum context for simple tasks"

context:
  system: |
    You are a helpful assistant.
    Be concise and accurate.

  constraints:
    - max_tokens: 500
    - format: plain_text
```

### 코딩 컨텍스트 템플릿

```yaml
# coding_context.yaml
version: "1.0"
name: "coding_context"
description: "Context for code-related tasks"

context:
  system: |
    You are an expert software engineer.
    Write clean, maintainable, well-documented code.

  languages:
    primary: python
    version: "3.11+"

  standards:
    - Use type hints
    - Follow PEP 8
    - Include docstrings
    - Handle errors gracefully

  tools:
    - read_file
    - write_file
    - exec
    - search

  output_format: |
    ```{language}
    # code here
    ```

    ## Explanation
    Brief explanation of the code.
```

### RAG 컨텍스트 템플릿

```yaml
# rag_context.yaml
version: "1.0"
name: "rag_context"
description: "Context for retrieval-augmented generation"

context:
  system: |
    Answer questions based on the provided documents.
    Always cite sources.
    If information is not in the documents, say so.

  retrieval:
    strategy: semantic_search
    top_k: 5
    rerank: true

  documents: |
    {retrieved_documents}

  format: |
    ## Answer
    {answer}

    ## Sources
    {citations}
```

---

## Workflow Templates

### Explore-Plan-Code-Commit

```yaml
# workflow_epcc.yaml
name: "explore_plan_code_commit"
description: "Systematic approach to coding tasks"

steps:
  - name: explore
    action: "Read and understand the codebase"
    tools: [read_file, search, list_dir]
    output: understanding

  - name: plan
    action: "Create detailed implementation plan"
    input: understanding
    thinking: extended  # Use deep reasoning
    output: plan

  - name: implement
    action: "Write code following the plan"
    input: plan
    tools: [write_file, edit_file]
    verify: true  # Verify each step
    output: code

  - name: finalize
    action: "Commit changes"
    input: code
    tools: [git_add, git_commit]
    output: commit
```

### Test-Driven Development

```yaml
# workflow_tdd.yaml
name: "test_driven_development"
description: "Test-first methodology"

steps:
  - name: write_tests
    action: "Create tests based on requirements"
    output: tests
    constraint: "Don't implement yet"

  - name: verify_tests_fail
    action: "Run tests to confirm they fail"
    input: tests
    expected: failure

  - name: implement
    action: "Write code to make tests pass"
    input: tests
    output: implementation

  - name: refactor
    action: "Clean up while maintaining tests"
    input: [tests, implementation]
    constraint: "Don't change behavior"

  - name: finalize
    action: "Commit both tests and implementation"
```

---

## Schema 정의

### Code Understanding Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Code Understanding Schema",
  "type": "object",
  "properties": {
    "codebase": {
      "type": "object",
      "properties": {
        "structure": {
          "type": "array",
          "description": "Key files and directories"
        },
        "architecture": {
          "type": "string",
          "description": "Overall architectural pattern"
        },
        "technologies": {
          "type": "array",
          "description": "Key technologies and frameworks"
        }
      }
    },
    "functionality": {
      "type": "object",
      "properties": {
        "entry_points": {"type": "array"},
        "core_workflows": {"type": "array"},
        "data_flow": {"type": "string"}
      }
    },
    "quality": {
      "type": "object",
      "properties": {
        "strengths": {"type": "array"},
        "concerns": {"type": "array"},
        "patterns": {"type": "array"}
      }
    }
  }
}
```

### Troubleshooting Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Troubleshooting Schema",
  "type": "object",
  "properties": {
    "problem": {
      "type": "object",
      "properties": {
        "symptoms": {"type": "array"},
        "context": {"type": "string"},
        "impact": {"type": "string"}
      }
    },
    "diagnosis": {
      "type": "object",
      "properties": {
        "potential_causes": {"type": "array"},
        "evidence": {"type": "array"},
        "verification_steps": {"type": "array"}
      }
    },
    "solution": {
      "type": "object",
      "properties": {
        "approach": {"type": "string"},
        "steps": {"type": "array"},
        "verification": {"type": "string"},
        "prevention": {"type": "string"}
      }
    }
  }
}
```

---

## 프로토콜 조합

```
┌─────────────────────────────────────────────────────────────┐
│                    Protocol Composition                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  /composite.code_review{                                     │
│      compose=[                                               │
│          /code.analyze,     # 먼저 분석                     │
│          /security.scan,    # 보안 검사                     │
│          /self.reflect,     # 자기 성찰                     │
│          /doc.generate      # 문서화                        │
│      ],                                                      │
│      flow="sequential",     # 순차 실행                     │
│      pass_context=true      # 컨텍스트 전달                 │
│  }                                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 베스트 프랙티스

| 원칙 | 설명 |
|------|------|
| **Single Purpose** | 각 프로토콜은 하나의 목적 |
| **Composable** | 다른 프로토콜과 조합 가능 |
| **Versioned** | 버전 관리로 변경 추적 |
| **Documented** | 목적과 사용법 명시 |
| **Tested** | 예상대로 작동하는지 검증 |

---

*다음 글에서는 실전 적용과 CLAUDE.md 작성법을 살펴봅니다.*
