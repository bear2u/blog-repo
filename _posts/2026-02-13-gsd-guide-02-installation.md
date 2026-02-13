---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (02) - 설치 및 시작"
date: 2026-02-13
permalink: /gsd-guide-02-installation/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, OpenCode, Gemini CLI, Installation, Setup]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD 설치 및 기본 설정 가이드"
---

## 빠른 설치

GSD는 **npx** 한 줄로 설치할 수 있습니다:

```bash
npx get-shit-done-cc@latest
```

**지원 플랫폼:** Mac, Windows, Linux

---

## 설치 옵션

설치 시 다음을 선택합니다:

1. **Runtime** — Claude Code, OpenCode, Gemini 또는 모두
2. **Location** — Global (모든 프로젝트) 또는 Local (현재 프로젝트만)

### 대화형 설치

```bash
npx get-shit-done-cc@latest
```

설치 프로그램이 런타임과 위치를 물어봅니다.

### 비대화형 설치 (Docker, CI, 스크립트용)

```bash
# Claude Code
npx get-shit-done-cc --claude --global   # ~/.claude/에 설치
npx get-shit-done-cc --claude --local    # ./.claude/에 설치

# OpenCode (오픈소스, 무료 모델)
npx get-shit-done-cc --opencode --global # ~/.config/opencode/에 설치

# Gemini CLI
npx get-shit-done-cc --gemini --global   # ~/.gemini/에 설치

# 모든 런타임
npx get-shit-done-cc --all --global      # 모든 디렉토리에 설치
```

| 옵션 | 설명 |
|------|------|
| `--global` 또는 `-g` | 전역 설치 (모든 프로젝트) |
| `--local` 또는 `-l` | 로컬 설치 (현재 프로젝트만) |
| `--claude` | Claude Code만 |
| `--opencode` | OpenCode만 |
| `--gemini` | Gemini CLI만 |
| `--all` | 모든 런타임 |

---

## 설치 확인

설치 후 선택한 런타임에서 확인:

```
/gsd:help
```

명령어 목록이 표시되면 설치 성공입니다.

---

## 개발 설치

저장소를 클론하여 로컬에서 테스트:

```bash
git clone https://github.com/gsd-build/get-shit-done.git
cd get-shit-done
node bin/install.js --claude --local
```

`./.claude/`에 설치되어 수정 사항을 기여 전에 테스트할 수 있습니다.

---

## 권장: 권한 건너뛰기 모드

GSD는 **마찰 없는 자동화**를 위해 설계되었습니다. Claude Code를 다음과 같이 실행하세요:

```bash
claude --dangerously-skip-permissions
```

> [!TIP]
> 이것이 GSD의 의도된 사용 방식입니다. `date`와 `git commit`을 50번 승인하는 것은 목적에 어긋납니다.

### 대안: 세부 권한 설정

해당 플래그를 사용하지 않으려면 프로젝트의 `.claude/settings.json`에 추가:

```json
{
  "permissions": {
    "allow": [
      "Bash(date:*)",
      "Bash(echo:*)",
      "Bash(cat:*)",
      "Bash(ls:*)",
      "Bash(mkdir:*)",
      "Bash(wc:*)",
      "Bash(head:*)",
      "Bash(tail:*)",
      "Bash(sort:*)",
      "Bash(grep:*)",
      "Bash(tr:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git status:*)",
      "Bash(git log:*)",
      "Bash(git diff:*)",
      "Bash(git tag:*)"
    ]
  }
}
```

---

## 업데이트

GSD는 빠르게 발전합니다. 주기적으로 업데이트하세요:

```bash
npx get-shit-done-cc@latest
```

---

## Docker/컨테이너 환경

파일 읽기가 틸드 경로(`~/.claude/...`)로 실패하는 경우, 설치 전에 `CLAUDE_CONFIG_DIR`을 설정:

```bash
CLAUDE_CONFIG_DIR=/home/youruser/.claude npx get-shit-done-cc --global
```

이렇게 하면 컨테이너에서 제대로 확장되지 않을 수 있는 `~` 대신 절대 경로가 사용됩니다.

---

## 제거

GSD를 완전히 제거하려면:

```bash
# 전역 설치
npx get-shit-done-cc --claude --global --uninstall
npx get-shit-done-cc --opencode --global --uninstall

# 로컬 설치 (현재 프로젝트)
npx get-shit-done-cc --claude --local --uninstall
npx get-shit-done-cc --opencode --local --uninstall
```

다른 설정은 유지하면서 모든 GSD 명령어, 에이전트, 훅, 설정을 제거합니다.

---

## 문제 해결

### 설치 후 명령어를 찾을 수 없음

- Claude Code를 재시작하여 슬래시 명령어 다시 로드
- `~/.claude/commands/gsd/` (전역) 또는 `./.claude/commands/gsd/` (로컬)에 파일이 있는지 확인

### 명령어가 예상대로 작동하지 않음

- `/gsd:help`로 설치 확인
- `npx get-shit-done-cc` 재실행하여 재설치

---

*다음 글에서는 GSD의 핵심 워크플로우를 살펴봅니다.*
