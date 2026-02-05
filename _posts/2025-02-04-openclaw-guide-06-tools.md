---
layout: post
title: "OpenClaw 완벽 가이드 (6) - 도구 & 브라우저"
date: 2025-02-04
permalink: /openclaw-guide-06-tools/
author: Peter Steinberger
categories: [AI 에이전트, OpenClaw]
tags: [OpenClaw, Browser, Canvas, Tools, Automation]
original_url: "https://github.com/openclaw/openclaw"
excerpt: "OpenClaw의 브라우저 제어, Canvas, 노드 도구를 상세히 알아봅니다."
---

## 도구 개요

OpenClaw 에이전트는 다양한 **도구(Tools)**를 사용하여 작업을 수행합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    OpenClaw Tools                                │
│                                                                  │
│   Core Tools:                                                   │
│   ├── bash          # 시스템 명령 실행                          │
│   ├── read/write    # 파일 읽기/쓰기                            │
│   ├── edit          # 파일 편집                                 │
│   └── process       # 프로세스 관리                             │
│                                                                  │
│   Browser Tools:                                                │
│   ├── browser.*     # 브라우저 제어                             │
│   └── canvas.*      # 캔버스 제어                               │
│                                                                  │
│   Node Tools:                                                   │
│   ├── camera.*      # 카메라 캡처                               │
│   ├── screen.*      # 화면 녹화                                 │
│   ├── location.*    # 위치 정보                                 │
│   └── system.*      # 시스템 명령 (노드)                        │
│                                                                  │
│   Session Tools:                                                │
│   ├── sessions_list    # 세션 목록                              │
│   ├── sessions_history # 세션 히스토리                          │
│   └── sessions_send    # 세션 간 메시지                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 브라우저 제어

OpenClaw는 **전용 관리 브라우저**를 제공합니다. 개인 브라우저와 완전히 분리된 환경에서 에이전트가 웹을 탐색합니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Browser Architecture                          │
│                                                                  │
│   Agent ──▶ Browser Tools ──▶ CDP (Chrome DevTools Protocol)   │
│                                       │                         │
│                                       ▼                         │
│                              Managed Browser                    │
│                              (openclaw profile)                 │
│                                                                  │
│   Profiles:                                                     │
│   • openclaw: 관리 브라우저 (주황색 테마)                       │
│   • chrome: 확장 릴레이 (시스템 브라우저)                       │
│   • work, remote: 커스텀 프로필                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 설정

```json5
// ~/.openclaw/openclaw.json
{
  browser: {
    enabled: true,
    defaultProfile: "openclaw",  // "openclaw" | "chrome"
    color: "#FF4500",            // 브라우저 UI 색상
    headless: false,             // 헤드리스 모드
    noSandbox: false,            // 샌드박스 비활성화
    attachOnly: false,           // 실행 중인 브라우저만 연결

    // 실행 파일 경로 (자동 감지 가능)
    executablePath: "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",

    // 프로필별 설정
    profiles: {
      openclaw: {
        cdpPort: 18800,
        color: "#FF4500",
      },
      work: {
        cdpPort: 18801,
        color: "#0066CC",
      },
      remote: {
        cdpUrl: "http://10.0.0.42:9222",
        color: "#00AA00",
      },
    },
  },
}
```

### CLI 명령

```bash
# 브라우저 상태
openclaw browser --browser-profile openclaw status

# 브라우저 시작
openclaw browser --browser-profile openclaw start

# URL 열기
openclaw browser --browser-profile openclaw open https://example.com

# 스냅샷 캡처
openclaw browser --browser-profile openclaw snapshot

# 탭 목록
openclaw browser --browser-profile openclaw tabs

# 탭 닫기
openclaw browser --browser-profile openclaw close <tab-id>
```

### 브라우저 도구

에이전트가 사용하는 브라우저 도구:

```typescript
// 탭 관리
browser.tabs.list()                    // 탭 목록
browser.tabs.open({ url: "..." })      // 새 탭
browser.tabs.close({ tabId: "..." })   // 탭 닫기
browser.tabs.focus({ tabId: "..." })   // 탭 포커스

// 페이지 액션
browser.page.snapshot()                // 페이지 스냅샷
browser.page.screenshot()              // 스크린샷
browser.page.pdf()                     // PDF 저장
browser.page.click({ selector: "..." })
browser.page.type({ selector: "...", text: "..." })
browser.page.scroll({ direction: "down" })

// 평가
browser.page.eval({ code: "document.title" })
```

---

## Canvas (A2UI)

**Canvas**는 에이전트가 제어하는 시각적 작업 공간입니다. **A2UI (Agent-to-UI)** 프로토콜로 실시간 HTML을 렌더링합니다.

### 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Canvas Architecture                           │
│                                                                  │
│   Agent ──▶ canvas.push ──▶ Canvas Host ──▶ Renderer           │
│                            (port 18793)    (Browser/App)        │
│                                                                  │
│   Features:                                                     │
│   • 실시간 HTML/CSS/JS 렌더링                                   │
│   • eval로 JavaScript 실행                                      │
│   • 스냅샷 캡처                                                  │
│   • macOS/iOS/Android 앱에서 표시                               │
└─────────────────────────────────────────────────────────────────┘
```

### Canvas 도구

```typescript
// HTML 푸시
canvas.push({
  html: `
    <h1>Hello World</h1>
    <p>현재 시간: ${new Date().toLocaleString()}</p>
  `
})

// JavaScript 실행
canvas.eval({
  code: `
    document.body.style.backgroundColor = '#f0f0f0';
    return document.title;
  `
})

// 스냅샷 캡처
canvas.snapshot()  // Base64 이미지 반환

// 리셋
canvas.reset()
```

### A2UI 예시

```html
<!-- 에이전트가 푸시하는 대시보드 -->
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: system-ui; padding: 20px; }
    .card { background: white; border-radius: 8px; padding: 16px; margin: 8px 0; }
    .metric { font-size: 2em; font-weight: bold; color: #FF4500; }
  </style>
</head>
<body>
  <h1>시스템 대시보드</h1>
  <div class="card">
    <h3>CPU 사용률</h3>
    <div class="metric">42%</div>
  </div>
  <div class="card">
    <h3>메모리</h3>
    <div class="metric">8.2 GB / 16 GB</div>
  </div>
</body>
</html>
```

---

## 노드 도구 (Node Tools)

**노드**는 디바이스 기능을 Gateway에 노출합니다. macOS, iOS, Android 앱이 노드로 연결됩니다.

### 노드 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                    Node Architecture                             │
│                                                                  │
│   Gateway ◀────────────────────────────────────────▶ Nodes     │
│      │                                                 │        │
│      │    node.list     (연결된 노드 목록)             │        │
│      │    node.describe (노드 기능 조회)               │        │
│      │    node.invoke   (명령 실행)                    │        │
│      │                                                 │        │
│      ▼                                                 ▼        │
│   macOS Node          iOS Node           Android Node           │
│   • system.run        • camera.snap      • camera.snap         │
│   • system.notify     • screen.record    • screen.record       │
│   • canvas.*          • location.get     • location.get        │
│   • camera.*          • canvas.*         • canvas.*            │
│   • screen.record                                               │
│   • location.get                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 노드 명령어

```bash
# 연결된 노드 목록
openclaw nodes list

# 노드 상세 정보
openclaw nodes describe macbook-pro

# 노드 명령 실행
openclaw nodes invoke macbook-pro system.notify --title "알림" --body "완료!"
```

### 노드 도구

```typescript
// 시스템 명령 (macOS 노드)
node.invoke({
  nodeId: "macbook-pro",
  command: "system.run",
  params: {
    command: "ls -la",
    needsScreenRecording: false,
  },
})

// 알림 전송
node.invoke({
  nodeId: "macbook-pro",
  command: "system.notify",
  params: {
    title: "빌드 완료",
    body: "프로젝트가 성공적으로 빌드되었습니다.",
  },
})

// 카메라 스냅샷
node.invoke({
  nodeId: "iphone",
  command: "camera.snap",
})

// 화면 녹화
node.invoke({
  nodeId: "iphone",
  command: "screen.record",
  params: {
    duration: 10,  // 초
  },
})

// 위치 정보
node.invoke({
  nodeId: "iphone",
  command: "location.get",
})
```

### macOS 권한

macOS 노드는 **TCC 권한**을 따릅니다:

| 명령 | 필요 권한 |
|------|-----------|
| `system.run` | 화면 녹화 (선택) |
| `system.notify` | 알림 |
| `camera.snap` | 카메라 |
| `screen.record` | 화면 녹화 |
| `location.get` | 위치 서비스 |

권한이 없으면 `PERMISSION_MISSING` 오류가 반환됩니다.

---

## 세션 도구 (Session Tools)

세션 도구로 **에이전트 간 통신**이 가능합니다:

```typescript
// 활성 세션 목록
sessions_list()
// 반환: [{ sessionKey: "main", ... }, { sessionKey: "group:dev", ... }]

// 세션 히스토리 조회
sessions_history({
  sessionKey: "main",
  limit: 10,
})

// 다른 세션에 메시지 전송
sessions_send({
  sessionKey: "group:dev-team",
  message: "배포가 완료되었습니다. 확인해주세요.",
  replyBack: true,      // 응답 대기
  announceSkip: false,  // 안내 메시지 생략
})
```

### 멀티 에이전트 시나리오

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Communication                     │
│                                                                  │
│   Main Session ────▶ sessions_send ────▶ DevOps Session        │
│        │                                       │                │
│        │ "배포해줘"                              │ 작업 수행     │
│        │                                       │                │
│        ◀────────── replyBack ◀──────────────────              │
│          "배포 완료"                                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Cron 작업

에이전트가 예약 작업을 실행할 수 있습니다:

```typescript
// Cron 작업 생성
cron.create({
  name: "daily-report",
  schedule: "0 9 * * *",  // 매일 오전 9시
  action: {
    type: "agent",
    message: "오늘의 일정을 요약해줘",
  },
})

// Cron 작업 목록
cron.list()

// Cron 작업 삭제
cron.delete({ name: "daily-report" })
```

### 설정에서 Cron

```json5
{
  cron: {
    jobs: [
      {
        name: "daily-standup",
        schedule: "0 9 * * 1-5",  // 평일 오전 9시
        action: {
          type: "agent",
          message: "오늘 할 일을 정리해줘",
          sessionKey: "main",
        },
      },
      {
        name: "weekly-backup",
        schedule: "0 2 * * 0",   // 일요일 새벽 2시
        action: {
          type: "bash",
          command: "~/scripts/backup.sh",
        },
      },
    ],
  },
}
```

---

## 도구 허용/차단

### 샌드박스 모드

```json5
{
  agents: {
    defaults: {
      sandbox: {
        mode: "non-main",  // "off" | "non-main" | "all"

        // 허용 도구
        allow: [
          "bash",
          "read",
          "write",
          "edit",
          "sessions_list",
          "sessions_history",
          "sessions_send",
        ],

        // 차단 도구
        deny: [
          "browser",
          "canvas",
          "nodes",
          "cron",
          "discord",
          "gateway",
        ],
      },
    },
  },
}
```

### 도구 모드 설명

| 모드 | 설명 |
|------|------|
| `off` | 샌드박싱 비활성화 |
| `non-main` | main 세션 외 모든 세션 샌드박싱 |
| `all` | 모든 세션 샌드박싱 |

---

## 도구 디버깅

```bash
# 에이전트 상태 (사용 가능한 도구 목록)
openclaw status --all

# 도구 테스트
openclaw agent --message "현재 디렉토리 파일 목록 보여줘" --thinking high

# 브라우저 도구 테스트
openclaw browser open https://example.com
openclaw browser snapshot
```

---

*다음 글에서는 음성 및 Talk Mode를 살펴봅니다.*
