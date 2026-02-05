---
layout: post
title: "Superset 완벽 가이드 (10) - 확장 및 커스터마이징"
date: 2025-02-05
permalink: /superset-guide-10-customization/
author: Superset Team
categories: [AI 에이전트, Superset]
tags: [Superset, Customization, Configuration, Scripts, Presets]
original_url: "https://github.com/superset-sh/superset"
excerpt: "Superset을 프로젝트에 맞게 커스터마이징하고 확장하는 방법을 알아봅니다."
---

## 설정 개요

Superset은 프로젝트별, 사용자별로 다양한 커스터마이징이 가능합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    설정 계층 구조                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   앱 설정 (전역)                                            │
│   └─→ ~/Library/Application Support/Superset/settings.json │
│                                                              │
│   프로젝트 설정 (레포별)                                     │
│   └─→ .superset/config.json                                 │
│                                                              │
│   워크스페이스 설정 (워크스페이스별)                         │
│   └─→ 설정/정리 스크립트, 환경 변수                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 설정 (.superset/config.json)

### 기본 구조

```json
{
  "setup": ["./.superset/setup.sh"],
  "teardown": ["./.superset/teardown.sh"],
  "presets": [
    {
      "name": "Dev Server",
      "command": "bun run dev"
    },
    {
      "name": "Tests",
      "command": "bun test"
    }
  ],
  "env": {
    "NODE_ENV": "development"
  }
}
```

### 설정 옵션

| 옵션 | 타입 | 설명 |
|------|------|------|
| `setup` | `string[]` | 워크스페이스 생성 시 실행할 스크립트 |
| `teardown` | `string[]` | 워크스페이스 삭제 시 실행할 스크립트 |
| `presets` | `Preset[]` | 터미널 프리셋 (Ctrl+1-9) |
| `env` | `object` | 환경 변수 설정 |
| `shell` | `string` | 기본 셸 (예: `/bin/zsh`) |

---

## 설정 스크립트

### setup.sh 예시

```bash
#!/bin/bash
# .superset/setup.sh

set -e  # 에러 시 중단

echo "🚀 Setting up workspace: $SUPERSET_WORKSPACE_NAME"

# ===== 환경 변수 =====
# SUPERSET_WORKSPACE_NAME - 워크스페이스 이름
# SUPERSET_ROOT_PATH      - 메인 레포 경로
# SUPERSET_TASK_ID        - 연결된 태스크 ID (있는 경우)

# ===== 1. 환경 파일 복사 =====
if [ -f "$SUPERSET_ROOT_PATH/.env" ]; then
  cp "$SUPERSET_ROOT_PATH/.env" .env
  echo "✅ Copied .env file"
fi

if [ -f "$SUPERSET_ROOT_PATH/.env.local" ]; then
  cp "$SUPERSET_ROOT_PATH/.env.local" .env.local
  echo "✅ Copied .env.local file"
fi

# ===== 2. 의존성 설치 =====
if [ -f "package.json" ]; then
  # 캐시된 node_modules 사용 (있는 경우)
  CACHE_DIR="$SUPERSET_ROOT_PATH/../.superset-cache/node_modules"
  if [ -d "$CACHE_DIR" ]; then
    echo "📦 Using cached node_modules"
    cp -r "$CACHE_DIR" ./node_modules
  fi

  # 설치
  if command -v bun &> /dev/null; then
    bun install
  elif command -v pnpm &> /dev/null; then
    pnpm install
  else
    npm install
  fi
  echo "✅ Installed dependencies"
fi

# ===== 3. 데이터베이스 설정 =====
if [ -f "prisma/schema.prisma" ]; then
  npx prisma generate
  echo "✅ Generated Prisma client"
fi

if [ -f "drizzle.config.ts" ]; then
  bun run db:push
  echo "✅ Pushed database schema"
fi

# ===== 4. 빌드 (필요한 경우) =====
if [ -f "turbo.json" ]; then
  bun run build --filter=./packages/*
  echo "✅ Built packages"
fi

# ===== 5. 완료 마커 =====
touch .setup-complete

echo "✨ Workspace ready!"
```

### teardown.sh 예시

```bash
#!/bin/bash
# .superset/teardown.sh

echo "🧹 Cleaning up workspace: $SUPERSET_WORKSPACE_NAME"

# ===== 1. 캐시 저장 (선택사항) =====
CACHE_DIR="$SUPERSET_ROOT_PATH/../.superset-cache"
if [ -d "node_modules" ] && [ ! -d "$CACHE_DIR/node_modules" ]; then
  mkdir -p "$CACHE_DIR"
  cp -r node_modules "$CACHE_DIR/"
  echo "📦 Cached node_modules for future workspaces"
fi

# ===== 2. 무거운 디렉토리 정리 =====
rm -rf node_modules
rm -rf .next
rm -rf dist
rm -rf build
rm -rf .turbo

# ===== 3. 임시 파일 정리 =====
rm -rf .env.local
rm -rf .setup-complete

echo "✅ Cleanup complete!"
```

---

## 터미널 프리셋

### 프리셋 설정

```json
{
  "presets": [
    {
      "name": "Dev Server",
      "command": "bun run dev",
      "icon": "play"
    },
    {
      "name": "Tests",
      "command": "bun test --watch",
      "icon": "test-tube"
    },
    {
      "name": "Build",
      "command": "bun run build",
      "icon": "package"
    },
    {
      "name": "Lint",
      "command": "bun run lint:fix",
      "icon": "check"
    },
    {
      "name": "DB Studio",
      "command": "bun run db:studio",
      "icon": "database"
    },
    {
      "name": "Storybook",
      "command": "bun run storybook",
      "icon": "book"
    }
  ]
}
```

### 단축키

| 단축키 | 프리셋 |
|--------|--------|
| `Ctrl+1` | 첫 번째 프리셋 |
| `Ctrl+2` | 두 번째 프리셋 |
| ... | ... |
| `Ctrl+9` | 아홉 번째 프리셋 |

---

## 키보드 단축키 커스터마이징

### 단축키 설정 열기

`Settings > Keyboard Shortcuts` 또는 `⌘/`

### 커스텀 단축키 예시

```json
// ~/Library/Application Support/Superset/keybindings.json
{
  "workspaces.new": "cmd+shift+n",
  "workspaces.switch.next": "cmd+alt+down",
  "workspaces.switch.prev": "cmd+alt+up",
  "terminal.new": "cmd+t",
  "terminal.close": "cmd+w",
  "terminal.split.right": "cmd+d",
  "terminal.split.down": "cmd+shift+d",
  "terminal.clear": "cmd+k",
  "changes.toggle": "cmd+l",
  "sidebar.toggle": "cmd+b"
}
```

---

## 앱 설정

### 설정 파일 위치

```
macOS: ~/Library/Application Support/Superset/settings.json
```

### 사용 가능한 설정

```json
{
  "theme": "dark",
  "fontSize": 14,
  "fontFamily": "JetBrains Mono",

  "terminal": {
    "shell": "/bin/zsh",
    "cursorStyle": "bar",
    "cursorBlink": true,
    "scrollback": 10000
  },

  "editor": {
    "wordWrap": true,
    "minimap": false,
    "lineNumbers": true
  },

  "notifications": {
    "enabled": true,
    "sound": true,
    "agentComplete": true,
    "agentError": true
  },

  "confirmOnQuit": true,
  "autoUpdate": true
}
```

---

## 에이전트 훅

Superset은 에이전트 이벤트에 대한 훅을 지원합니다.

### 훅 설정

```json
// .superset/config.json
{
  "hooks": {
    "onAgentStart": "./.superset/hooks/agent-start.sh",
    "onAgentComplete": "./.superset/hooks/agent-complete.sh",
    "onAgentError": "./.superset/hooks/agent-error.sh"
  }
}
```

### 훅 스크립트 예시

```bash
#!/bin/bash
# .superset/hooks/agent-complete.sh

# 환경 변수로 전달되는 정보:
# SUPERSET_AGENT_NAME - 에이전트 이름
# SUPERSET_WORKSPACE_NAME - 워크스페이스 이름
# SUPERSET_EXIT_CODE - 종료 코드

if [ "$SUPERSET_EXIT_CODE" -eq 0 ]; then
  # macOS 알림
  osascript -e "display notification \"$SUPERSET_AGENT_NAME completed in $SUPERSET_WORKSPACE_NAME\" with title \"Superset\""

  # 슬랙 알림 (선택사항)
  curl -X POST -H 'Content-type: application/json' \
    --data "{\"text\":\"✅ Agent completed: $SUPERSET_WORKSPACE_NAME\"}" \
    "$SLACK_WEBHOOK_URL"
fi
```

---

## IDE 통합

### 외부 에디터 설정

```json
// settings.json
{
  "externalEditor": {
    "name": "VSCode",
    "command": "code",
    "args": ["{path}"]
  }
}
```

### 지원되는 에디터

| 에디터 | 명령어 |
|--------|--------|
| VSCode | `code {path}` |
| Cursor | `cursor {path}` |
| WebStorm | `webstorm {path}` |
| Sublime | `subl {path}` |
| Vim/Neovim | `nvim {path}` |

---

## GitHub 통합

### gh CLI 사용

Superset은 GitHub 작업에 `gh` CLI를 활용합니다.

```bash
# PR 생성
gh pr create --title "Fix: Login bug" --body "..."

# 이슈 조회
gh issue view 123

# PR 체크아웃
gh pr checkout 456
```

### PR 자동 연결

태스크와 PR을 자동으로 연결할 수 있습니다.

```json
// .superset/config.json
{
  "github": {
    "autoLinkPR": true,
    "prTemplate": ".github/pull_request_template.md"
  }
}
```

---

## 플러그인 시스템 (향후)

Superset은 플러그인 시스템을 계획하고 있습니다.

### 플러그인 구조 (예정)

```
~/.superset/plugins/
├── my-plugin/
│   ├── package.json
│   ├── index.ts
│   └── manifest.json
```

### manifest.json (예정)

```json
{
  "name": "my-plugin",
  "version": "1.0.0",
  "description": "My custom plugin",
  "main": "index.ts",
  "hooks": ["onAgentComplete", "onWorkspaceCreate"],
  "commands": [
    {
      "name": "my-command",
      "title": "My Command",
      "shortcut": "cmd+shift+m"
    }
  ]
}
```

---

## 디버깅 & 문제 해결

### 로그 확인

```bash
# macOS
tail -f ~/Library/Logs/Superset/main.log

# 또는 앱 내에서
View > Toggle Developer Tools
```

### 설정 초기화

```bash
# 설정 백업
cp -r ~/Library/Application\ Support/Superset ~/Desktop/superset-backup

# 설정 초기화
rm -rf ~/Library/Application\ Support/Superset
rm -rf ~/Library/Caches/Superset
```

### 일반적인 문제

| 문제 | 해결책 |
|------|--------|
| 워크스페이스 생성 실패 | Git 버전 확인 (2.20+), worktree 지원 확인 |
| 터미널 반응 없음 | `bun run clean:workspaces` 후 재설치 |
| 단축키 작동 안함 | keybindings.json 구문 확인 |
| 테마 적용 안됨 | 앱 재시작 |

---

## 베스트 프랙티스

### 1. 프로젝트별 설정 템플릿

```bash
# 새 프로젝트에 Superset 설정 추가
mkdir -p .superset
cat > .superset/config.json << 'EOF'
{
  "setup": ["./.superset/setup.sh"],
  "teardown": ["./.superset/teardown.sh"],
  "presets": []
}
EOF
```

### 2. 설정 스크립트 재사용

```bash
# 공통 스크립트를 별도 레포로 관리
git clone https://github.com/my-org/superset-scripts ~/.superset-scripts

# config.json에서 참조
{
  "setup": ["~/.superset-scripts/setup-node.sh"]
}
```

### 3. 팀 표준화

```bash
# .superset/ 디렉토리를 버전 관리
git add .superset/
git commit -m "Add Superset configuration"
```

### 4. 민감 정보 보호

```bash
# .gitignore에 추가
.superset/secrets/
.superset/*.local.sh
```

---

## 마무리

이 가이드 시리즈에서 Superset의 주요 기능과 구조를 살펴보았습니다.

### 핵심 정리

1. **병렬 에이전트 실행**: 10개 이상의 코딩 에이전트 동시 실행
2. **Worktree 격리**: Git worktree로 태스크별 완벽한 격리
3. **통합 모니터링**: 모든 에이전트 상태를 한 곳에서 확인
4. **유연한 커스터마이징**: 프로젝트별 설정, 프리셋, 훅 지원

### 추가 리소스

- **[GitHub](https://github.com/superset-sh/superset)** - 소스 코드
- **[공식 문서](https://docs.superset.sh)** - 상세 문서
- **[Discord](https://discord.gg/cZeD9WYcV7)** - 커뮤니티

---

*이 가이드가 Superset을 효과적으로 활용하는 데 도움이 되길 바랍니다. Happy coding!*
