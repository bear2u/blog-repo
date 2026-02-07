---
layout: post
title: "Claude Skills 완벽 가이드 (09) - 트러블슈팅 및 참고 자료"
date: 2026-02-07
permalink: /claude-skills-guide-09-troubleshooting/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Troubleshooting, Reference, Debugging]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "일반적인 문제 해결 방법과 빠른 참고 자료"
---

## 일반적인 문제와 해결책

---

## 1. 스킬 업로드 실패

### 에러: "Could not find SKILL.md in uploaded folder"

**원인:** 파일명이 정확히 `SKILL.md`가 아님

**해결:**
```bash
# 파일명 확인
ls -la

# 올바른 이름으로 변경
mv skill.md SKILL.md
# 또는
mv Skill.md SKILL.md
```

**확인:** `SKILL.md` (정확히 이 철자, 대소문자 구분)

---

### 에러: "Invalid frontmatter"

**원인:** YAML 포맷 오류

**일반적인 실수:**

```yaml
# ❌ 잘못됨 - 구분자 없음
name: my-skill
description: Does things

# ❌ 잘못됨 - 따옴표 미닫힘
name: my-skill
description: "Does things

# ❌ 잘못됨 - 잘못된 들여쓰기
name: my-skill
description: Line 1
 Line 2 (잘못된 들여쓰기)

# ✅ 올바름
---
name: my-skill
description: Does things
---

# ✅ 올바름 - 여러 줄
---
name: my-skill
description: Does multiple things including
  task A, task B, and task C.
---
```

**디버깅 팁:**
- YAML 검증기 사용: [yamllint.com](http://www.yamllint.com/)
- 구분자 `---` 확인
- 들여쓰기는 2칸 공백 사용

---

### 에러: "Invalid skill name"

**원인:** 이름에 공백 또는 대문자 포함

```yaml
# ❌ 잘못됨
name: My Cool Skill
name: my_cool_skill
name: MyCoolSkill

# ✅ 올바름
name: my-cool-skill
```

**규칙:** kebab-case만 사용 (소문자 + 하이픈)

---

## 2. 스킬이 트리거되지 않음

### 증상
- 스킬이 자동으로 로드되지 않음
- 관련 쿼리에도 활성화되지 않음

---

### 해결 방법

**1단계: Description 검토**

```yaml
# ❌ 너무 일반적 - 트리거 안됨
description: Helps with projects.

# ✅ 구체적 + 트리거 문구
description: End-to-end Linear sprint planning including task creation,
  team assignment, and milestone setup. Use when user says "plan sprint",
  "create Linear sprint", "set up iteration", or "organize Linear tasks".
```

**체크리스트:**
- [ ] 스킬이 **무엇을 하는지** 명확한가?
- [ ] **언제 사용하는지** 명시되어 있는가?
- [ ] 사용자가 실제로 말할 법한 문구가 포함되어 있는가?
- [ ] 관련 파일 형식이 있다면 언급했는가?

---

**2단계: Claude에게 물어보기**

```
"When would you use the [skill-name] skill?"
```

Claude가 description을 인용하며 답변합니다. 누락된 것이 무엇인지 확인하고 조정하세요.

---

**3단계: 더 많은 컨텍스트 추가**

```yaml
# Before
description: Manages Notion projects

# After - 더 많은 트리거 단어
description: Manages Notion project workspaces including page creation,
  database setup, and team collaboration. Use when user mentions "Notion project",
  "workspace setup", "create Notion pages", uploads .notion files, or asks
  to "organize in Notion".
```

**추가 팁:**
- 동의어 포함
- 기술 용어 명시
- 파일 확장자 언급
- 구체적인 동사 사용 (create, setup, organize 등)

---

## 3. 스킬이 너무 자주 트리거됨

### 증상
- 관련 없는 쿼리에도 스킬이 로드됨
- 사용자가 스킬을 비활성화함
- 목적에 대한 혼란

---

### 해결 방법

**1. 부정 트리거 추가**

```yaml
description: Advanced data analysis for CSV files including statistical modeling,
  regression analysis, and clustering. Use for "analyze CSV", "statistical analysis",
  "data modeling". Do NOT use for simple data exploration (use data-viz skill instead)
  or for non-CSV formats.
```

---

**2. 범위 명확히**

```yaml
# ❌ 너무 광범위
description: Processes documents

# ✅ 구체적 범위
description: Processes PDF legal documents specifically for contract review
  and clause extraction. Use for "review contract", "analyze legal PDF",
  "extract clauses". Only for legal PDFs, not general documents.
```

---

**3. 구체적으로 제한**

```yaml
description: PayFlow payment processing for e-commerce transactions. Use
  specifically for online payment workflows with PayFlow integration, not
  for general financial queries, accounting, or other payment providers.
```

---

## 4. MCP 연결 문제

### 증상
- 스킬이 로드되지만 MCP 호출 실패
- "Tool not found" 에러
- Authentication 에러

---

### 해결 체크리스트

**1. MCP 서버 연결 확인**

```
Claude.ai:
Settings > Extensions > [Your Service]

상태: "Connected" 확인
```

**2. 인증 확인**

- [ ] API 키가 유효하고 만료되지 않음
- [ ] 적절한 권한/스코프가 부여됨
- [ ] OAuth 토큰이 갱신됨

**3. MCP 독립 테스트**

스킬 없이 MCP만 직접 호출:

```
"Use [Service] MCP to fetch my projects"
```

이것이 실패하면 문제는 스킬이 아닌 MCP입니다.

---

**4. 도구 이름 검증**

```yaml
# ❌ 잘못된 도구 이름
Call MCP tool: `createProject`

# ✅ 올바른 도구 이름 (대소문자 정확히)
Call MCP tool: `create_project`
```

**확인 방법:**
- MCP 서버 문서에서 정확한 도구 이름 확인
- 도구 이름은 대소문자 구분
- 언더스코어 vs. 카멜케이스 확인

---

## 5. 명령어를 따르지 않음

### 증상
- 스킬이 로드되지만 지침을 무시함
- 단계를 건너뜀
- 다르게 동작함

---

### 일반적인 원인과 해결

**1. 지침이 너무 장황함**

```markdown
# ❌ 너무 장황 - Claude가 길을 잃음
Step 1: First you need to do this thing and then after
that you should probably consider doing another thing but
only if the first thing worked well and also keep in mind
that there might be edge cases...

# ✅ 간결하고 명확
Step 1: Fetch user data
```bash
mcp-tool call users get_user --id ${USER_ID}
```
Expected: User object with name, email, role
```

**해결:**
- 간결하게 유지
- 글머리 기호와 번호 목록 사용
- 상세 참조는 별도 파일로

---

**2. 지침이 묻혀 있음**

```markdown
# ❌ 중요한 내용이 하단에
## Background
[Long explanation...]

## Step 1
[Instructions...]

## Important
CRITICAL: Always validate input!

# ✅ 중요한 내용을 상단에
## ⚠️ CRITICAL: Input Validation
ALWAYS validate:
- User ID is non-empty
- Permissions are checked

## Instructions
Step 1: ...
```

**해결:**
- 중요한 지침은 상단에
- `## Important` 또는 `## Critical` 헤더 사용
- 필요시 핵심 사항 반복

---

**3. 모호한 언어**

```markdown
# ❌ 모호함
Make sure to validate things properly

# ✅ 명확함
CRITICAL: Before calling create_project, verify:
1. Project name is non-empty (min 3 characters)
2. At least one team member assigned
3. Start date is not in the past
4. Budget is within company limits ($0-$1M)

If any check fails:
- Do NOT proceed
- Return specific error message
- Suggest correction
```

**고급 팁:** 중요한 검증은 스크립트로 번들링하세요. 코드는 결정적이지만 언어 해석은 그렇지 않습니다.

```bash
# 검증 스크립트 사용
python scripts/validate_project.py --data ${PROJECT_DATA}
if [ $? -ne 0 ]; then
  echo "Validation failed"
  exit 1
fi
```

---

**4. 모델 "게으름"**

명시적인 격려 추가:

```markdown
## Performance Notes

- Take your time to do this thoroughly
- Quality is more important than speed
- Do not skip validation steps
- Double-check all API calls before executing
```

**참고:** 사용자 프롬프트에 이를 추가하는 것이 SKILL.md에 넣는 것보다 효과적입니다.

---

## 6. 대용량 컨텍스트 문제

### 증상
- 스킬이 느림
- 응답 품질 저하
- 토큰 제한 도달

---

### 원인
- 스킬 내용이 너무 큼
- 너무 많은 스킬이 동시에 활성화됨
- Progressive Disclosure 대신 모든 내용 로드

---

### 해결 방법

**1. SKILL.md 크기 최적화**

```markdown
# ❌ 모든 것을 SKILL.md에
## Complete API Reference
[10,000 words of API docs...]

# ✅ 핵심만 SKILL.md에, 나머지는 references/에
## API Integration

For detailed API reference, see `references/api-guide.md`.

Quick reference:
- Authentication: Bearer token
- Rate limit: 100 req/min
- Pagination: cursor-based
```

**목표:** SKILL.md를 5,000 단어 이하로 유지

---

**2. 활성화된 스킬 줄이기**

```
동시에 20-50개 이상의 스킬이 활성화되어 있나요?

해결:
- 선택적 활성화 권장
- 관련 기능별로 "스킬 팩" 구성
- 사용하지 않는 스킬 비활성화
```

---

**3. Progressive Disclosure 활용**

```
Level 1: YAML frontmatter
  → 항상 로드 (최소 정보)

Level 2: SKILL.md
  → 관련 있을 때만 로드 (핵심 지침)

Level 3: references/
  → 필요시에만 로드 (상세 문서)
```

---

## 빠른 참고 자료

---

## 스킬 개발 체크리스트

### 시작하기 전

- [ ] 2-3개의 구체적 유스케이스 식별
- [ ] 필요한 도구 확인 (내장 또는 MCP)
- [ ] 이 가이드와 예시 스킬 검토
- [ ] 폴더 구조 계획

---

### 개발 중

- [ ] 폴더명이 kebab-case인가
- [ ] `SKILL.md` 파일 존재 (정확한 철자)
- [ ] YAML frontmatter에 `---` 구분자
- [ ] `name` 필드: kebab-case, 공백 없음, 대문자 없음
- [ ] `description`에 WHAT과 WHEN 포함
- [ ] XML 태그 (`<` `>`) 미사용
- [ ] 명령어가 명확하고 실행 가능
- [ ] 에러 처리 포함
- [ ] 예시 제공
- [ ] 참조 문서 명확히 링크

---

### 업로드 전

- [ ] 명확한 작업에 트리거 테스트
- [ ] 변형된 표현에도 트리거 테스트
- [ ] 관련 없는 주제에는 트리거되지 않는지 확인
- [ ] 기능 테스트 통과
- [ ] 도구 통합 작동 (해당되는 경우)
- [ ] `.zip` 파일로 압축

---

### 업로드 후

- [ ] 실제 대화에서 테스트
- [ ] Under/over-triggering 모니터링
- [ ] 사용자 피드백 수집
- [ ] Description과 지침 반복 개선
- [ ] metadata에서 버전 업데이트

---

## YAML Frontmatter 스펙

### 필수 필드

```yaml
---
name: skill-name-in-kebab-case
description: What it does and when to use it. Include specific trigger phrases.
---
```

---

### 모든 선택적 필드

```yaml
name: skill-name
description: [required description]
license: MIT                                          # 선택: 오픈소스 라이선스
compatibility: Requires Python 3.8+, npm, network    # 선택: 환경 요구사항
allowed-tools: "Bash(python:*) Bash(npm:*) WebFetch"  # 선택: 도구 액세스 제한
metadata:                                              # 선택: 커스텀 필드
  author: Company Name
  version: 1.0.0
  mcp-server: server-name
  category: productivity
  tags: [project-management, automation]
  documentation: https://example.com/docs
  support: support@example.com
```

---

### 보안 규칙

**허용:**
- 표준 YAML 타입
- 커스텀 metadata 필드
- 긴 설명 (최대 1024자)

**금지:**
- XML 꺾쇠괄호 (`<` `>`)
- YAML에서 코드 실행
- "claude" 또는 "anthropic" 접두사 이름

---

## 완전한 스킬 예시

프로덕션 준비된 완전한 스킬 예시:

- **Document Skills** — PDF, DOCX, PPTX, XLSX 생성
- **Example Skills** — 다양한 워크플로우 패턴
- **Partner Skills Directory** — Asana, Atlassian, Canva, Figma, Sentry, Zapier 등의 스킬

**레포지토리:** [github.com/anthropics/skills](https://github.com/anthropics/skills)

---

## 유용한 리소스

### 공식 문서
- [Skills 문서](https://docs.anthropic.com/skills)
- [API 레퍼런스](https://docs.anthropic.com/api)
- [MCP 문서](https://modelcontextprotocol.io)
- [Best Practices 가이드](https://docs.anthropic.com/best-practices)

---

### 블로그 포스트
- Introducing Agent Skills
- Engineering Blog: Equipping Agents for the Real World
- Skills Explained
- How to Create Skills for Claude
- Building Skills for Claude Code
- Improving Frontend Design through Skills

---

### 커뮤니티 지원
- **Discord:** Claude Developers Discord
- **GitHub:** `anthropics/skills/issues` (버그 리포트)
- **포럼:** Claude Community Forum

---

## 디버깅 플로우차트

```
스킬 문제 발생
    │
    ├─ 업로드 실패?
    │   ├─ SKILL.md 이름 확인
    │   ├─ YAML frontmatter 검증
    │   └─ 폴더 구조 확인
    │
    ├─ 트리거 안됨?
    │   ├─ Description 검토
    │   ├─ 트리거 문구 추가
    │   └─ Claude에게 물어보기
    │
    ├─ 너무 자주 트리거?
    │   ├─ 범위 명확히
    │   ├─ 부정 트리거 추가
    │   └─ 더 구체적으로
    │
    ├─ MCP 실패?
    │   ├─ 서버 연결 확인
    │   ├─ 인증 확인
    │   ├─ 독립 테스트
    │   └─ 도구 이름 확인
    │
    ├─ 지침 무시?
    │   ├─ 간결하게 작성
    │   ├─ 중요한 내용 상단에
    │   ├─ 명확하게 표현
    │   └─ 검증 스크립트 사용
    │
    └─ 성능 문제?
        ├─ SKILL.md 크기 줄이기
        ├─ References/ 활용
        └─ 활성 스킬 줄이기
```

---

## 마무리

**축하합니다!** Claude Skills 완벽 가이드를 완료했습니다.

### 다음 단계

1. **첫 스킬 만들기**
   - 구체적인 유스케이스 선택
   - 작게 시작
   - 반복 개선

2. **커뮤니티 참여**
   - 스킬 공유
   - 피드백 받기
   - 다른 사람의 스킬에서 배우기

3. **계속 학습**
   - 공식 문서 팔로우
   - 새로운 패턴 실험
   - 베스트 프랙티스 기여

---

**Happy Skill Building! 🚀**

---

## 추가 도움이 필요하신가요?

- **버그 발견:** [GitHub Issues](https://github.com/anthropics/skills/issues)
- **질문:** [Discord](https://discord.gg/claude)
- **피드백:** [community@anthropic.com](mailto:community@anthropic.com)

---

*이것으로 Claude Skills 완벽 가이드 시리즈를 마칩니다. 시작부터 배포, 트러블슈팅까지 모든 것을 다뤘습니다. 이제 여러분만의 멋진 스킬을 만들어 보세요!*
