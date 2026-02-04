---
layout: post
title: "OpenClaw 완벽 가이드 (5) - 스킬 시스템"
date: 2025-02-04
permalink: /openclaw-guide-05-skills/
author: Peter Steinberger
category: AI
tags: [OpenClaw, Skills, ClawHub, SKILL.md, AgentSkills]
series: openclaw-guide
part: 5
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw의 스킬 시스템과 커스텀 스킬 작성법을 상세히 알아봅니다."
---

## 스킬이란?

**스킬(Skills)**은 에이전트에게 **도구 사용법을 가르치는 SKILL.md 파일**입니다. 각 스킬은 특정 기능(GitHub, Notion, 브라우저 등)을 에이전트가 활용할 수 있게 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Skills Architecture                           │
│                                                                  │
│   SKILL.md ──▶ 에이전트 프롬프트에 주입 ──▶ 도구 사용 가능     │
│                                                                  │
│   skill/                                                        │
│   ├── SKILL.md          # 메타데이터 + 사용법                   │
│   └── (optional files)  # 헬퍼 스크립트                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 스킬 로드 우선순위

스킬은 **세 곳**에서 로드됩니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Skill Precedence                              │
│                                                                  │
│   1. Workspace Skills (최우선)                                  │
│      ~/.openclaw/workspace/skills/<skill>/                      │
│                                                                  │
│   2. Managed Skills                                             │
│      ~/.openclaw/skills/<skill>/                                │
│                                                                  │
│   3. Bundled Skills (최하위)                                    │
│      <openclaw>/skills/<skill>/                                 │
│                                                                  │
│   같은 이름 충돌 시: Workspace > Managed > Bundled              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 번들 스킬 목록

OpenClaw에는 **50개 이상의 번들 스킬**이 포함됩니다:

### 개발 도구

| 스킬 | 설명 | 필요 조건 |
|------|------|-----------|
| `github` | GitHub 이슈, PR, 검색 | `gh` CLI |
| `coding-agent` | 코딩 에이전트 워크플로우 | - |
| `session-logs` | 세션 로그 분석 | - |

### 생산성

| 스킬 | 설명 | 필요 조건 |
|------|------|-----------|
| `notion` | Notion 페이지/DB 관리 | API 키 |
| `obsidian` | Obsidian 노트 접근 | Obsidian 설치 |
| `apple-notes` | Apple Notes 관리 | macOS |
| `apple-reminders` | Apple Reminders | macOS |
| `bear-notes` | Bear 노트 앱 | Bear 설치 |

### 미디어 & AI

| 스킬 | 설명 | 필요 조건 |
|------|------|-----------|
| `openai-image-gen` | DALL-E 이미지 생성 | OpenAI API |
| `openai-whisper` | 음성 전사 | OpenAI API |
| `openai-whisper-api` | Whisper API | OpenAI API |
| `gemini` | Gemini CLI 사용 | `gemini` CLI |
| `nano-banana-pro` | Gemini 이미지 생성 | Gemini API |

### 브라우저 & 웹

| 스킬 | 설명 | 필요 조건 |
|------|------|-----------|
| `canvas` | Canvas 제어 | - |
| `peekaboo` | 스크린샷 도구 | macOS |
| `camsnap` | 카메라 스냅샷 | 카메라 |

### 유틸리티

| 스킬 | 설명 | 필요 조건 |
|------|------|-----------|
| `1password` | 1Password 통합 | `op` CLI |
| `healthcheck` | 시스템 헬스 체크 | - |
| `model-usage` | 모델 사용량 추적 | - |
| `himalaya` | 이메일 CLI | `himalaya` |

---

## SKILL.md 형식

### 기본 구조

```markdown
---
name: my-skill
description: "내 커스텀 스킬 설명"
---

# My Skill

이 스킬은 에이전트에게 특정 기능을 가르칩니다.

## 사용법

다음과 같이 사용하세요:
- `tool_name arg1 arg2`

## 예시

`run_my_tool --help` 명령으로 도움말을 확인하세요.
```

### 고급 메타데이터

```markdown
---
name: nano-banana-pro
description: "Gemini 3 Pro로 이미지 생성/편집"
homepage: https://docs.openclaw.ai/skills/nano-banana-pro
user-invocable: true
disable-model-invocation: false
metadata: { "openclaw": { "emoji": "🍌", "requires": { "bins": ["uv"], "env": ["GEMINI_API_KEY"] }, "primaryEnv": "GEMINI_API_KEY", "install": [{ "id": "uv", "kind": "go", "package": "github.com/...", "bins": ["uv"] }] } }
---
```

### 메타데이터 필드

| 필드 | 설명 |
|------|------|
| `name` | 스킬 이름 (필수) |
| `description` | 짧은 설명 (필수) |
| `homepage` | 문서 URL |
| `user-invocable` | 슬래시 명령으로 호출 가능 (기본: true) |
| `disable-model-invocation` | 모델 프롬프트에서 제외 |
| `command-dispatch` | `tool`로 설정 시 직접 도구 호출 |
| `command-tool` | dispatch할 도구 이름 |
| `metadata` | OpenClaw 게이팅 정보 (JSON) |

---

## 스킬 게이팅

스킬은 **로드 타임에 필터링**됩니다:

### requires 필드

```json
{
  "openclaw": {
    "requires": {
      "bins": ["node", "npm"],      // 모두 PATH에 존재해야 함
      "anyBins": ["chrome", "brave"], // 하나 이상 존재
      "env": ["API_KEY"],           // 환경 변수 또는 설정
      "config": ["browser.enabled"]  // openclaw.json 경로
    },
    "os": ["darwin", "linux"],      // 운영체제 제한
    "always": false                  // true면 항상 로드
  }
}
```

### 게이팅 예시

```markdown
---
name: apple-notes
description: "Apple Notes 접근"
metadata: { "openclaw": { "requires": { "bins": ["shortcuts"] }, "os": ["darwin"] } }
---
```

이 스킬은:
- macOS에서만 로드
- `shortcuts` 명령이 PATH에 있어야 로드

---

## 스킬 설치

### ClawHub에서 설치

[ClawHub](https://clawhub.com)는 OpenClaw의 공개 스킬 레지스트리입니다.

```bash
# 스킬 설치
clawhub install <skill-slug>

# 모든 스킬 업데이트
clawhub update --all

# 스킬 동기화 (업데이트 발행)
clawhub sync --all

# 스킬 검색
clawhub search "github"
```

### 수동 설치

```bash
# Workspace에 스킬 폴더 생성
mkdir -p ~/.openclaw/workspace/skills/my-skill

# SKILL.md 작성
cat > ~/.openclaw/workspace/skills/my-skill/SKILL.md << 'EOF'
---
name: my-skill
description: "내 커스텀 스킬"
---

# My Skill

사용법 설명...
EOF
```

---

## 설정에서 스킬 관리

### 스킬 활성화/비활성화

```json5
// ~/.openclaw/openclaw.json
{
  skills: {
    entries: {
      // 번들 스킬 활성화
      "github": {
        enabled: true,
      },

      // API 키 필요 스킬
      "notion": {
        enabled: true,
        apiKey: "secret_...",  // primaryEnv에 매핑
      },

      // 환경 변수 직접 설정
      "openai-whisper": {
        enabled: true,
        env: {
          OPENAI_API_KEY: "sk-...",
        },
      },

      // 스킬 비활성화
      "discord": {
        enabled: false,
      },
    },

    // 번들 스킬 허용 목록 (빈 배열 = 모두 차단)
    allowBundled: ["github", "notion", "canvas"],

    // 추가 스킬 폴더
    load: {
      extraDirs: ["/path/to/shared/skills"],
    },

    // 스킬 설치 시 노드 매니저
    install: {
      nodeManager: "pnpm",  // npm|pnpm|yarn|bun
    },
  },
}
```

---

## 커스텀 스킬 작성

### 단순 스킬

```markdown
---
name: weather-check
description: "날씨 정보 조회"
---

# Weather Check Skill

## 사용법

`curl` 명령으로 날씨 정보를 조회합니다:

```bash
curl wttr.in/Seoul
```

## 예시

- 서울 날씨: `curl wttr.in/Seoul`
- 뉴욕 날씨: `curl wttr.in/NewYork`
```

### 도구 바인딩 스킬

```markdown
---
name: my-api-tool
description: "커스텀 API 호출"
command-dispatch: tool
command-tool: bash
---

# My API Tool

사용자가 `/my-api-tool <query>` 명령을 사용하면,
다음 bash 명령이 실행됩니다:

```bash
curl -X POST https://api.example.com/query \
  -H "Authorization: Bearer $MY_API_KEY" \
  -d '{"query": "<query>"}'
```
```

### 바이너리 요구 스킬

```markdown
---
name: docker-helper
description: "Docker 컨테이너 관리"
metadata: { "openclaw": { "requires": { "bins": ["docker"] }, "emoji": "🐳" } }
---

# Docker Helper

## 전제 조건

Docker가 설치되어 있어야 합니다.

## 명령어

- 컨테이너 목록: `docker ps`
- 이미지 목록: `docker images`
- 로그 확인: `docker logs <container>`
```

### 설치 스크립트 포함 스킬

```markdown
---
name: go-tool
description: "Go 기반 도구"
metadata: { "openclaw": { "requires": { "bins": ["mytool"] }, "install": [{ "id": "go", "kind": "go", "package": "github.com/user/mytool@latest", "bins": ["mytool"], "label": "Install mytool (go)" }] } }
---

# Go Tool

## 설치

macOS Skills UI에서 "Install" 버튼을 클릭하거나:

```bash
go install github.com/user/mytool@latest
```
```

---

## 플러그인 스킬

플러그인은 자체 스킬을 포함할 수 있습니다:

```json
// openclaw.plugin.json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "skills": ["./skills/my-skill"]
}
```

플러그인 스킬은 플러그인 활성화 시 자동으로 로드됩니다.

---

## 스킬 디버깅

### 로드된 스킬 확인

```bash
# 에이전트 상태에서 스킬 목록
openclaw status

# 스킬 상세 정보
openclaw skills info github
```

### 게이팅 문제 진단

```bash
# Doctor 실행
openclaw doctor

# 출력 예:
# ⚠ Skill 'apple-notes' skipped: requires macOS
# ⚠ Skill 'notion' skipped: missing env NOTION_API_KEY
# ✓ Skill 'github' loaded
```

---

## 보안 주의사항

1. **서드파티 스킬 신뢰**: 설치 전 SKILL.md 내용 검토
2. **API 키 보호**: 프롬프트/로그에 키 노출 주의
3. **샌드박싱**: 위험한 도구는 샌드박스에서 실행
4. **환경 변수 격리**: `skills.entries.*.env`는 호스트 프로세스에 주입

```json5
// 안전한 API 키 설정
{
  skills: {
    entries: {
      "notion": {
        apiKey: "secret_...",  // 직접 입력보다
        env: {
          NOTION_API_KEY: "${NOTION_API_KEY}",  // 환경 변수 참조 권장
        },
      },
    },
  },
}
```

---

*다음 글에서는 도구와 브라우저 제어를 살펴봅니다.*
