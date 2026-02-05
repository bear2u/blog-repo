---
layout: post
title: "Beads 완벽 가이드 (8) - Molecules & Wisps"
date: 2025-02-04
permalink: /beads-guide-08-molecules/
author: Steve Yegge
categories: [AI 에이전트, Beads]
tags: [Beads, Molecules, Wisps, Templates, Workflow]
original_url: "https://github.com/steveyegge/beads"
excerpt: "Beads의 템플릿 워크플로우 시스템인 Molecules와 Wisps를 분석합니다."
---

## Molecules란?

**Molecules**는 Beads의 **재사용 가능한 작업 템플릿**입니다. 반복적인 워크플로우를 패턴화하여 일관된 작업 구조를 생성합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Molecules Overview                            │
│                                                                  │
│   Molecule Template          →         Instantiated Issues      │
│   ┌─────────────────┐                  ┌─────────────────┐      │
│   │ feature-dev     │    bd molecule   │ bd-a1b2 (Epic)  │      │
│   │   ├─ design     │  ────────────▶   │   ├─ bd-a1b2.1  │      │
│   │   ├─ implement  │                  │   ├─ bd-a1b2.2  │      │
│   │   ├─ test       │                  │   ├─ bd-a1b2.3  │      │
│   │   └─ deploy     │                  │   └─ bd-a1b2.4  │      │
│   └─────────────────┘                  └─────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Molecule 정의

### YAML 템플릿

{% raw %}
```yaml
# .beads/molecules/feature-dev.yaml

name: feature-dev
description: Standard feature development workflow
version: "1.0"

variables:
  - name: feature_name
    description: Name of the feature
    required: true
  - name: assignee
    description: Primary assignee
    default: ""

steps:
  - id: epic
    title: "Implement {{ feature_name }}"
    type: epic
    priority: 1

  - id: design
    title: "Design {{ feature_name }}"
    type: task
    priority: 1
    parent: epic
    description: |
      Create technical design document for {{ feature_name }}
      - Architecture decisions
      - API contracts
      - Data models

  - id: implement
    title: "Implement {{ feature_name }}"
    type: task
    priority: 1
    parent: epic
    blocks:
      - design  # design 완료 후 시작
    assignee: "{{ assignee }}"

  - id: test
    title: "Test {{ feature_name }}"
    type: task
    priority: 1
    parent: epic
    blocks:
      - implement

  - id: deploy
    title: "Deploy {{ feature_name }}"
    type: task
    priority: 2
    parent: epic
    blocks:
      - test
```
{% endraw %}

---

## Molecule 사용

### bd molecule 명령어

```bash
# Molecule로 이슈 생성
bd molecule apply feature-dev \
  --var feature_name="User Authentication" \
  --var assignee="agent-1"

# 출력:
# Created 5 issues from molecule 'feature-dev':
#   bd-a1b2     Implement User Authentication (epic)
#   bd-a1b2.1   Design User Authentication (task)
#   bd-a1b2.2   Implement User Authentication (task)
#   bd-a1b2.3   Test User Authentication (task)
#   bd-a1b2.4   Deploy User Authentication (task)
```

```bash
# 사용 가능한 Molecules 목록
bd molecule list

# Molecule 상세 보기
bd molecule show feature-dev

# 드라이런 (미리보기)
bd molecule apply feature-dev --dry-run --var feature_name="OAuth"
```

---

## Molecule 구현

```go
// internal/molecules/loader.go

type Molecule struct {
    Name        string     `yaml:"name"`
    Description string     `yaml:"description"`
    Version     string     `yaml:"version"`
    Variables   []Variable `yaml:"variables"`
    Steps       []Step     `yaml:"steps"`
}

type Variable struct {
    Name        string `yaml:"name"`
    Description string `yaml:"description"`
    Required    bool   `yaml:"required"`
    Default     string `yaml:"default"`
}

type Step struct {
    ID          string   `yaml:"id"`
    Title       string   `yaml:"title"`
    Type        string   `yaml:"type"`
    Priority    int      `yaml:"priority"`
    Parent      string   `yaml:"parent,omitempty"`
    Blocks      []string `yaml:"blocks,omitempty"`
    Description string   `yaml:"description,omitempty"`
    Assignee    string   `yaml:"assignee,omitempty"`
    Labels      []string `yaml:"labels,omitempty"`
}

func LoadMolecule(path string) (*Molecule, error) {
    data, err := os.ReadFile(path)
    if err != nil {
        return nil, err
    }

    var mol Molecule
    if err := yaml.Unmarshal(data, &mol); err != nil {
        return nil, err
    }

    return &mol, nil
}
```

### 템플릿 렌더링

```go
// internal/molecules/render.go

func (m *Molecule) Render(vars map[string]string) ([]*types.Issue, error) {
    // 필수 변수 확인
    for _, v := range m.Variables {
        if v.Required {
            if _, ok := vars[v.Name]; !ok {
                return nil, fmt.Errorf("missing required variable: %s", v.Name)
            }
        }
    }

    // 기본값 적용
    for _, v := range m.Variables {
        if _, ok := vars[v.Name]; !ok && v.Default != "" {
            vars[v.Name] = v.Default
        }
    }

    var issues []*types.Issue
    idMap := make(map[string]string) // step ID → issue ID

    for _, step := range m.Steps {
        issue := &types.Issue{
            ID:        generateIssueID(),
            Title:     renderTemplate(step.Title, vars),
            IssueType: types.IssueType(step.Type),
            Priority:  step.Priority,
            Status:    types.StatusOpen,
            CreatedAt: time.Now(),
            UpdatedAt: time.Now(),
        }

        if step.Description != "" {
            issue.Description = renderTemplate(step.Description, vars)
        }

        if step.Assignee != "" {
            issue.Assignee = renderTemplate(step.Assignee, vars)
        }

        // Parent 처리
        if step.Parent != "" {
            if parentID, ok := idMap[step.Parent]; ok {
                issue.Dependencies = append(issue.Dependencies, types.Dependency{
                    FromID: issue.ID,
                    ToID:   parentID,
                    Type:   types.DepParentChild,
                })
            }
        }

        // Blocks 처리
        for _, blockRef := range step.Blocks {
            if blockerID, ok := idMap[blockRef]; ok {
                issue.Dependencies = append(issue.Dependencies, types.Dependency{
                    FromID: blockerID,
                    ToID:   issue.ID,
                    Type:   types.DepBlocks,
                })
            }
        }

        idMap[step.ID] = issue.ID
        issues = append(issues, issue)
    }

    return issues, nil
}

{% raw %}
func renderTemplate(tmpl string, vars map[string]string) string {
    result := tmpl
    for k, v := range vars {
        result = strings.ReplaceAll(result, "{{ "+k+" }}", v)
    }
    return result
}
{% endraw %}
```

---

## 내장 Molecules

### bug-fix

{% raw %}
```yaml
# .beads/molecules/bug-fix.yaml

name: bug-fix
description: Bug fix workflow

variables:
  - name: bug_title
    required: true
  - name: severity
    default: "P2"

steps:
  - id: investigate
    title: "Investigate: {{ bug_title }}"
    type: task
    priority: "{{ severity }}"

  - id: fix
    title: "Fix: {{ bug_title }}"
    type: bug
    priority: "{{ severity }}"
    blocks:
      - investigate

  - id: verify
    title: "Verify fix: {{ bug_title }}"
    type: task
    priority: "{{ severity }}"
    blocks:
      - fix
```
{% endraw %}

### code-review

{% raw %}
```yaml
# .beads/molecules/code-review.yaml

name: code-review
description: Code review workflow

variables:
  - name: pr_number
    required: true
  - name: reviewer
    default: ""

steps:
  - id: review
    title: "Review PR #{{ pr_number }}"
    type: task
    priority: 1
    assignee: "{{ reviewer }}"

  - id: feedback
    title: "Address feedback PR #{{ pr_number }}"
    type: task
    priority: 1
    blocks:
      - review

  - id: merge
    title: "Merge PR #{{ pr_number }}"
    type: task
    priority: 1
    blocks:
      - feedback
```
{% endraw %}

---

## Wisps

**Wisps**는 더 가벼운 형태의 템플릿으로, 단일 이슈에 대한 메타데이터 프리셋입니다.

```yaml
# .beads/wisps/urgent-bug.yaml

name: urgent-bug
description: P0 urgent bug preset

preset:
  type: bug
  priority: 0
  labels:
    - urgent
    - production
  assignee: on-call
```

### Wisp 사용

```bash
# Wisp 적용
bd create "Database connection failing" --wisp urgent-bug

# 결과:
# Created bd-x1y2:
#   Title: Database connection failing
#   Type: bug
#   Priority: P0
#   Labels: urgent, production
#   Assignee: on-call
```

---

## Molecule 저장 위치

```
.beads/
├── molecules/           # 프로젝트별 Molecules
│   ├── feature-dev.yaml
│   └── bug-fix.yaml
├── wisps/              # 프로젝트별 Wisps
│   └── urgent-bug.yaml
└── config.yaml

~/.beads/
├── molecules/           # 전역 Molecules
└── wisps/              # 전역 Wisps
```

**우선순위:** 프로젝트 > 전역

---

## Molecule 검증

```go
// internal/molecules/validate.go

func (m *Molecule) Validate() error {
    if m.Name == "" {
        return errors.New("molecule name is required")
    }

    stepIDs := make(map[string]bool)
    for _, step := range m.Steps {
        if step.ID == "" {
            return errors.New("step ID is required")
        }
        if stepIDs[step.ID] {
            return fmt.Errorf("duplicate step ID: %s", step.ID)
        }
        stepIDs[step.ID] = true
    }

    // 의존성 참조 검증
    for _, step := range m.Steps {
        if step.Parent != "" && !stepIDs[step.Parent] {
            return fmt.Errorf("unknown parent reference: %s", step.Parent)
        }
        for _, block := range step.Blocks {
            if !stepIDs[block] {
                return fmt.Errorf("unknown blocks reference: %s", block)
            }
        }
    }

    // 순환 의존성 검사
    return m.detectCycles()
}
```

---

## 커스텀 Molecule 생성

```bash
# 인터랙티브 생성
bd molecule create my-workflow

# 기존 이슈에서 생성
bd molecule extract bd-a1b2 --name extracted-workflow

# 다른 프로젝트에서 임포트
bd molecule import https://github.com/org/beads-molecules/feature-dev.yaml
```

---

*다음 글에서는 확장 및 통합을 살펴봅니다.*
