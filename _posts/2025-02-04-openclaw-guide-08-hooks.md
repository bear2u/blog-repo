---
layout: post
title: "OpenClaw 완벽 가이드 (8) - 훅 & 자동화"
date: 2025-02-04
permalink: /openclaw-guide-08-hooks/
author: Peter Steinberger
category: AI
tags: [OpenClaw, Hooks, Automation, Webhook, Cron, Gmail]
series: openclaw-guide
part: 8
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw의 이벤트 훅, Webhook, Cron 작업, Gmail 트리거를 알아봅니다."
---

## 훅 개요

**훅(Hooks)**은 OpenClaw의 이벤트 기반 자동화 시스템입니다. 에이전트 이벤트에 반응하여 스크립트를 실행합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hooks vs Webhooks                             │
│                                                                  │
│   Hooks:                                                        │
│   • Gateway 내부에서 실행                                       │
│   • 에이전트 이벤트에 반응                                      │
│   • TypeScript 핸들러                                           │
│                                                                  │
│   Webhooks:                                                     │
│   • 외부 HTTP 요청 수신                                         │
│   • 다른 시스템에서 트리거                                       │
│   • 에이전트 작업 시작                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 훅 시스템

### 번들 훅

OpenClaw에 포함된 기본 훅:

| 훅 | 이모지 | 설명 |
|----|--------|------|
| `session-memory` | 💾 | `/new` 시 세션 컨텍스트를 memory/에 저장 |
| `command-logger` | 📝 | 모든 명령을 logs/commands.log에 기록 |
| `boot-md` | 🚀 | Gateway 시작 시 BOOT.md 실행 |
| `soul-evil` | 😈 | 특정 조건에서 SOUL.md를 SOUL_EVIL.md로 교체 |

### 훅 CLI 명령

```bash
# 훅 목록
openclaw hooks list

# 훅 활성화
openclaw hooks enable session-memory

# 훅 비활성화
openclaw hooks disable session-memory

# 훅 상태 확인
openclaw hooks check

# 훅 상세 정보
openclaw hooks info session-memory

# 훅 설치 (외부)
openclaw hooks install <path-or-spec>
```

---

## 훅 구조

### 디렉토리 구조

```
my-hook/
├── HOOK.md          # 메타데이터 + 문서
└── handler.ts       # 핸들러 구현
```

### HOOK.md 형식

```markdown
---
name: my-hook
description: "내 커스텀 훅 설명"
homepage: https://example.com/docs
metadata: { "openclaw": { "emoji": "🔔", "events": ["command:new", "agent:complete"], "requires": { "bins": ["node"] } } }
---

# My Hook

이 훅은 /new 명령과 에이전트 완료 시 실행됩니다.

## 기능

- 세션 리셋 시 로그 기록
- 에이전트 응답 완료 시 알림 전송

## 설정

config.json에서 다음 설정 가능:
- `notifyOnComplete`: 완료 알림 활성화
```

### handler.ts 형식

```typescript
// handler.ts
import type { HookHandler, HookEvent } from "openclaw/plugin-sdk"

export const handler: HookHandler = async (event: HookEvent) => {
  const { type, payload, context } = event

  switch (type) {
    case "command:new":
      // /new 명령 처리
      console.log("Session reset by:", payload.userId)
      break

    case "agent:complete":
      // 에이전트 완료 처리
      const { summary, sessionKey } = payload
      await saveToMemory(sessionKey, summary)
      break
  }
}

async function saveToMemory(sessionKey: string, summary: string) {
  // 메모리 저장 로직
}
```

---

## 훅 이벤트

### 명령 이벤트

| 이벤트 | 설명 |
|--------|------|
| `command:new` | `/new` 또는 `/reset` 명령 |
| `command:compact` | `/compact` 명령 |
| `command:stop` | `/stop` 명령 |
| `command:think` | `/think` 레벨 변경 |

### 에이전트 이벤트

| 이벤트 | 설명 |
|--------|------|
| `agent:start` | 에이전트 실행 시작 |
| `agent:complete` | 에이전트 응답 완료 |
| `agent:error` | 에이전트 오류 발생 |

### 라이프사이클 이벤트

| 이벤트 | 설명 |
|--------|------|
| `gateway:start` | Gateway 시작 |
| `gateway:stop` | Gateway 종료 |
| `session:create` | 세션 생성 |
| `session:destroy` | 세션 종료 |

---

## 커스텀 훅 작성

### 예시: 슬랙 알림 훅

```markdown
---
name: slack-notify
description: "에이전트 완료 시 Slack 알림"
metadata: { "openclaw": { "emoji": "💬", "events": ["agent:complete"], "requires": { "env": ["SLACK_WEBHOOK_URL"] } } }
---

# Slack Notify Hook

에이전트가 작업을 완료하면 Slack으로 알림을 전송합니다.
```

```typescript
// handler.ts
import type { HookHandler } from "openclaw/plugin-sdk"

export const handler: HookHandler = async (event) => {
  if (event.type !== "agent:complete") return

  const webhookUrl = process.env.SLACK_WEBHOOK_URL
  if (!webhookUrl) return

  const { summary, sessionKey, model } = event.payload

  await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text: `🤖 에이전트 작업 완료`,
      blocks: [
        {
          type: "section",
          text: {
            type: "mrkdwn",
            text: `*세션:* ${sessionKey}\n*모델:* ${model}\n*요약:* ${summary}`,
          },
        },
      ],
    }),
  })
}
```

### 훅 설치

```bash
# 로컬 훅 설치
mkdir -p ~/.openclaw/hooks/slack-notify
# HOOK.md와 handler.ts 복사

# 훅 활성화
openclaw hooks enable slack-notify
```

---

## 훅 팩

여러 훅을 패키지로 묶을 수 있습니다:

```json
// package.json
{
  "name": "@acme/my-hooks",
  "version": "1.0.0",
  "openclaw": {
    "hooks": [
      "./hooks/slack-notify",
      "./hooks/discord-notify",
      "./hooks/email-digest"
    ]
  }
}
```

```bash
# 훅 팩 설치
openclaw hooks install @acme/my-hooks
```

---

## Webhook

**Webhook**은 외부 시스템에서 OpenClaw로 요청을 보내는 기능입니다.

### 설정

```json5
{
  webhook: {
    enabled: true,
    secret: "your-webhook-secret",  // 서명 검증용

    // 경로별 설정
    routes: {
      "/github": {
        action: "agent",
        sessionKey: "github-events",
      },
      "/deploy": {
        action: "bash",
        command: "~/scripts/deploy.sh",
      },
    },
  },
}
```

### Webhook 엔드포인트

```bash
# Webhook URL
# http://localhost:18789/webhook/<route>

# 예시: GitHub 웹훅
curl -X POST http://localhost:18789/webhook/github \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your-secret" \
  -d '{"action": "push", "repository": "..."}'
```

### GitHub Webhook 예시

1. GitHub 저장소 → Settings → Webhooks
2. Payload URL: `https://your-gateway.com/webhook/github`
3. Content type: `application/json`
4. Secret: `your-webhook-secret`

```json5
{
  webhook: {
    routes: {
      "/github": {
        action: "agent",
        sessionKey: "github",
        template: "GitHub 이벤트: {{action}} on {{repository.full_name}}",
      },
    },
  },
}
```

---

## Cron 작업

예약 작업을 설정할 수 있습니다:

### 설정

```json5
{
  cron: {
    jobs: [
      {
        name: "daily-summary",
        schedule: "0 18 * * *",  // 매일 오후 6시
        action: {
          type: "agent",
          message: "오늘 완료한 작업을 요약해줘",
          sessionKey: "main",
        },
      },
      {
        name: "health-check",
        schedule: "*/30 * * * *",  // 30분마다
        action: {
          type: "bash",
          command: "curl -s http://localhost:3000/health",
        },
      },
      {
        name: "weekly-report",
        schedule: "0 9 * * 1",  // 매주 월요일 오전 9시
        action: {
          type: "webhook",
          url: "https://api.example.com/report",
          method: "POST",
        },
      },
    ],
  },
}
```

### Cron 표현식

```
┌───────────── 분 (0-59)
│ ┌───────────── 시 (0-23)
│ │ ┌───────────── 일 (1-31)
│ │ │ ┌───────────── 월 (1-12)
│ │ │ │ ┌───────────── 요일 (0-6, 0=일요일)
│ │ │ │ │
* * * * *
```

예시:
- `0 9 * * *` - 매일 오전 9시
- `*/15 * * * *` - 15분마다
- `0 0 * * 0` - 매주 일요일 자정
- `0 9 * * 1-5` - 평일 오전 9시

### Cron CLI

```bash
# Cron 작업 목록
openclaw cron list

# Cron 작업 실행 (테스트)
openclaw cron run daily-summary

# Cron 작업 비활성화
openclaw cron disable daily-summary
```

---

## Gmail Pub/Sub 트리거

Gmail 메일 수신 시 에이전트를 트리거할 수 있습니다:

### 설정

```json5
{
  gmail: {
    enabled: true,

    // Google Cloud Pub/Sub 설정
    pubsub: {
      projectId: "your-gcp-project",
      topicId: "gmail-notifications",
      subscriptionId: "openclaw-gmail",
    },

    // 필터
    filters: {
      from: ["important@example.com"],
      subject: ["urgent", "action required"],
    },

    // 액션
    action: {
      type: "agent",
      sessionKey: "email",
      template: "새 이메일: {{subject}} from {{from}}",
    },
  },
}
```

### Gmail Pub/Sub 설정

```bash
# Gmail 웹훅 설정 도우미
openclaw webhooks gmail setup

# 연결 테스트
openclaw webhooks gmail test
```

---

## 자동화 예시

### 1. 일일 스탠드업 자동화

```json5
{
  cron: {
    jobs: [
      {
        name: "standup",
        schedule: "0 9 * * 1-5",
        action: {
          type: "agent",
          message: "GitHub에서 어제 내 활동을 요약하고, 오늘 할 일을 정리해줘",
          sessionKey: "standup",
        },
      },
    ],
  },
}
```

### 2. PR 리뷰 알림

```json5
{
  webhook: {
    routes: {
      "/github-pr": {
        action: "agent",
        sessionKey: "code-review",
        filter: {
          action: ["opened", "review_requested"],
        },
        template: "PR 리뷰 요청: {{pull_request.title}} by {{sender.login}}",
      },
    },
  },
}
```

### 3. 이메일 자동 요약

```json5
{
  gmail: {
    filters: {
      label: ["important"],
    },
    action: {
      type: "agent",
      message: "이 이메일을 요약하고 필요한 액션이 있으면 알려줘: {{body}}",
    },
  },
}
```

---

## 문제 해결

### 훅 디버그

```bash
# 훅 로그 확인
tail -f ~/.openclaw/logs/hooks.log

# 훅 이벤트 테스트
openclaw hooks trigger command:new --payload '{"userId": "test"}'
```

### Cron 디버그

```bash
# Cron 로그
tail -f ~/.openclaw/logs/cron.log

# 다음 실행 시간 확인
openclaw cron next daily-summary
```

---

*다음 글에서는 앱과 노드를 살펴봅니다.*
