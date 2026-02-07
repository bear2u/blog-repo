---
layout: post
title: "Claude Skills 완벽 가이드 (05) - 효과적인 명령어 작성"
date: 2026-02-07
permalink: /claude-skills-guide-05-writing/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Instructions, Best Practices, Documentation]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "Claude가 정확히 이해하고 실행할 수 있는 명령어 작성 방법"
---

## Description 필드의 중요성

Anthropic 엔지니어링 블로그에 따르면:

> "This metadata...provides just enough information for Claude to know when each skill should be used without loading all of it into context."

Description은 **Progressive Disclosure의 첫 번째 레벨**입니다.

---

## Description 구조

**공식:** `[무엇을 하는가]` + `[언제 사용하는가]` + `[핵심 기능]`

---

### 좋은 Description 예시

```yaml
# ✅ 구체적이고 실행 가능
description: Analyzes Figma design files and generates developer handoff
  documentation. Use when user uploads .fig files, asks for "design specs",
  "component documentation", or "design-to-code handoff".

# ✅ 트리거 문구 포함
description: Manages Linear project workflows including sprint planning, task
  creation, and status tracking. Use when user mentions "sprint", "Linear tasks",
  "project planning", or asks to "create tickets".

# ✅ 명확한 가치 제안
description: End-to-end customer onboarding workflow for PayFlow. Handles account
  creation, payment setup, and subscription management. Use when user says
  "onboard new customer", "set up subscription", or "create PayFlow account".
```

---

### 나쁜 Description 예시

```yaml
# ❌ 너무 모호함
description: Helps with projects.

# ❌ 트리거 조건 없음
description: Creates sophisticated multi-page documentation systems.

# ❌ 너무 기술적, 사용자 관점 부족
description: Implements the Project entity model with hierarchical relationships.
```

---

## 메인 명령어 작성

YAML frontmatter 다음에는 Markdown으로 실제 명령어를 작성합니다.

---

### 권장 구조

이 템플릿을 자신의 스킬에 맞게 조정하세요.

```markdown
---
name: your-skill
description: [...]
---

# Your Skill Name

## Instructions

### Step 1: [첫 번째 주요 단계]
명확한 설명.

```bash
python scripts/fetch_data.py --project-id PROJECT_ID
```

**예상 출력:** [성공 시 어떤 결과가 나오는지 설명]

### Step 2: [두 번째 주요 단계]
다음 작업 설명.

(필요한 만큼 단계 추가)

---

## Examples

### Example 1: [일반적인 시나리오]

**사용자 입력:** "Set up a new marketing campaign"

**실행 동작:**
1. MCP를 통해 기존 캠페인 가져오기
2. 제공된 파라미터로 새 캠페인 생성

**결과:** Campaign created with confirmation link

(필요한 만큼 예시 추가)

---

## Troubleshooting

**에러:** [흔한 에러 메시지]
**원인:** [왜 발생하는가]
**해결:** [어떻게 고치는가]

(필요한 만큼 에러 케이스 추가)
```

---

## 명령어 작성 베스트 프랙티스

### 1. 구체적이고 실행 가능하게

✅ **좋은 예시:**
```markdown
Run `python scripts/validate.py --input {filename}` to check data format.

If validation fails, common issues include:
- Missing required fields (add them to the CSV)
- Invalid date formats (use YYYY-MM-DD)
- Duplicate IDs (check column 'user_id')
```

❌ **나쁜 예시:**
```markdown
Validate the data before proceeding.
```

---

### 2. 번들된 리소스 명확히 참조

```markdown
## API Integration

Before writing queries, consult `references/api-patterns.md` for:
- Rate limiting guidance (max 100 req/min)
- Pagination patterns (use cursor-based)
- Error codes and handling (4xx vs 5xx)

See `references/examples/` for working code samples.
```

**왜 중요한가:**
- Claude가 필요시 해당 파일을 읽을 수 있음
- Progressive Disclosure 3단계 활용

---

### 3. Progressive Disclosure 활용

**SKILL.md:** 핵심 명령어만
**references/:** 상세 문서

```markdown
# SKILL.md
## Step 3: Configure API Client

Set up authentication using the project API key.
For detailed API reference, see `references/api-guide.md`.

---

# references/api-guide.md (별도 파일)
## Complete API Reference

### Authentication
- Bearer token format
- Token refresh logic
- Rate limiting details

### Endpoints
- GET /projects
- POST /projects
- PUT /projects/{id}
...
```

---

### 4. 에러 처리 포함

```markdown
## Common Issues

### MCP Connection Failed

**증상:** "Connection refused" 에러

**해결 단계:**
1. MCP 서버가 실행 중인지 확인
   - Settings > Extensions 확인
2. API 키가 유효한지 확인
   - Settings > [Your Service] > API Key
3. 재연결 시도
   - Settings > Extensions > [Your Service] > Reconnect

### Rate Limit Exceeded

**증상:** "429 Too Many Requests"

**해결:**
1. 1분 대기 후 재시도
2. 배치 크기 줄이기 (100개 → 50개)
3. Rate limit 상태 확인:
   ```bash
   curl -H "Authorization: Bearer $API_KEY" \
     https://api.example.com/rate-limit
   ```
```

---

## 실전 예시: Linear Sprint Planner

```markdown
---
name: linear-sprint-planner
description: End-to-end Linear sprint planning including task creation, team
  assignment, and milestone setup. Use when user says "plan sprint",
  "create Linear sprint", or "set up iteration".
metadata:
  mcp-server: linear
---

# Linear Sprint Planner

## Instructions

### Step 1: Analyze Previous Sprint

Fetch last completed sprint:
```bash
mcp-tool call linear list_sprints --limit 1 --status completed
```

Analyze:
- **Completion rate:** Tasks completed / Total tasks
- **Unfinished tasks:** Move to backlog or next sprint?
- **Team velocity:** Average story points completed

### Step 2: Create New Sprint

Ask user for:
- Sprint name (default: "Sprint [N+1]")
- Duration (default: 2 weeks)
- Start date (default: next Monday)

```bash
mcp-tool call linear create_sprint \
  --name "Sprint 23" \
  --start-date "2025-01-20" \
  --end-date "2025-02-03" \
  --team-id TEAM_ID
```

**Expected output:**
```json
{
  "id": "sprint_abc123",
  "name": "Sprint 23",
  "status": "created"
}
```

### Step 3: Generate Tasks

From user specifications:
1. Break down features into tasks
2. Estimate story points (Fibonacci: 1, 2, 3, 5, 8, 13)
3. Assign priorities (P0: critical → P3: nice-to-have)

```bash
for task in tasks:
  mcp-tool call linear create_issue \
    --title "${task.title}" \
    --description "${task.description}" \
    --estimate ${task.points} \
    --priority ${task.priority} \
    --sprint-id "sprint_abc123"
```

### Step 4: Assign Team Members

Assignment logic:
- **Workload balance:** Current sprint load < 40 points
- **Skill matching:** Frontend tasks → Frontend engineers
- **Previous context:** Continue related tasks

```bash
mcp-tool call linear assign_issue \
  --issue-id ISSUE_ID \
  --assignee-id USER_ID
```

### Step 5: Set Up Milestones

Create checkpoints:
- **Week 1:** Design review (Day 5)
- **Week 2:** Implementation complete (Day 10)
- **End of sprint:** Ready for QA (Day 14)

```bash
mcp-tool call linear create_milestone \
  --name "Design Review" \
  --due-date "2025-01-24" \
  --sprint-id "sprint_abc123"
```

---

## Examples

### Example 1: New Feature Sprint

**User says:** "Plan a sprint for the new dashboard feature"

**Actions:**
1. Review last sprint (Sprint 22): 85% completion rate
2. Create Sprint 23 (2 weeks from Monday)
3. Generate 12 tasks from dashboard spec:
   - 3 design tasks (P0)
   - 6 implementation tasks (P0-P1)
   - 2 testing tasks (P1)
   - 1 documentation task (P2)
4. Assign to team:
   - Alice (Frontend): 5 tasks, 21 points
   - Bob (Backend): 4 tasks, 18 points
   - Charlie (QA): 2 tasks, 8 points
5. Milestones:
   - Jan 24: Design complete
   - Jan 31: Implementation complete
   - Feb 3: Ready for QA

**Result:** Sprint 23 created with 12 tasks, all assigned

### Example 2: Bug Fix Sprint

**User says:** "Create a sprint for P0 bug fixes"

**Actions:**
1. Query Linear for P0 bugs: 8 found
2. Create Sprint 24 (1 week, emergency sprint)
3. Group bugs by component:
   - Auth: 3 bugs
   - Payments: 2 bugs
   - Dashboard: 3 bugs
4. Assign to specialists
5. Daily check-in milestone

**Result:** Sprint 24 created, team notified via Slack

---

## Troubleshooting

### Error: "Team capacity exceeded"

**Cause:** Total story points > team capacity (40 points/person)

**Solution:**
1. Reduce task scope
2. Move P2/P3 tasks to backlog
3. Add team members if available

### Error: "Sprint dates overlap"

**Cause:** Previous sprint not completed

**Solution:**
1. Complete current sprint first:
   ```bash
   mcp-tool call linear complete_sprint --sprint-id CURRENT_SPRINT_ID
   ```
2. Move unfinished tasks to backlog
3. Create new sprint

### Warning: "Unbalanced workload"

**Cause:** One team member has >50% more points than others

**Solution:**
Reassign tasks to balance:
```bash
# Check current assignments
mcp-tool call linear list_issues --sprint-id SPRINT_ID --group-by assignee

# Reassign heavy tasks
mcp-tool call linear assign_issue --issue-id ISSUE_ID --assignee-id NEW_USER_ID
```

---

## Best Practices

1. **Sprint duration:** 1-2 weeks for most teams
2. **Story points:** Total < 40 points per person
3. **Buffer:** Reserve 20% capacity for unexpected tasks
4. **Daily standups:** Use milestone reminders
5. **Retrospectives:** Review previous sprint before planning

---

## References

For more details:
- API rate limits: `references/linear-api-limits.md`
- Story point estimation guide: `references/estimation-guide.md`
- Team structure examples: `references/team-examples.md`
```

---

## 체크리스트

### Description 작성 시
- [ ] 스킬이 무엇을 하는지 명확한가?
- [ ] 언제 사용하는지 명시했는가?
- [ ] 구체적인 트리거 문구를 포함했는가?
- [ ] 사용자 관점에서 작성했는가?

### 명령어 작성 시
- [ ] 단계별로 명확히 구분했는가?
- [ ] 실행 가능한 명령어를 제공했는가?
- [ ] 예상 출력을 설명했는가?
- [ ] 에러 처리를 포함했는가?

### 예시 작성 시
- [ ] 일반적인 유스케이스를 다뤘는가?
- [ ] 입력-처리-출력이 명확한가?
- [ ] 다양한 시나리오를 커버하는가?

### 트러블슈팅 작성 시
- [ ] 흔한 에러들을 포함했는가?
- [ ] 원인과 해결책이 명확한가?
- [ ] 실행 가능한 해결 단계인가?

---

## 다음 단계

명령어 작성이 완료되었다면:

1. 테스트 케이스 정의
2. 트리거 조건 검증
3. 기능 테스트 수행
4. 피드백 기반 개선

---

*다음 글에서는 스킬 테스트 및 반복 개선 방법을 다룹니다.*
