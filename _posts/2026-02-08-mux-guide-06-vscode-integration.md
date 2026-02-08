---
layout: post
title: "Mux 완벽 가이드 (06) - VS Code 통합"
date: 2026-02-08 00:00:00 +0900
categories: [AI 코딩, 개발 도구]
tags: [Mux, VSCode, Cursor, 확장, 워크스페이스, IDE통합]
author: cataclysm99
original_url: "https://github.com/coder/mux"
excerpt: "VS Code/Cursor 확장으로 Mux 워크스페이스 빠르게 점프하고 효율적인 개발 워크플로우 구축"
permalink: /mux-guide-06-vscode-integration/
toc: true
related_posts:
  - /blog-repo/2026-02-08-mux-guide-05-multimodel
  - /blog-repo/2026-02-08-mux-guide-07-advanced-features
---

## VS Code 확장 개요

Mux VS Code 확장은 Mux 워크스페이스와 IDE를 연결하여 **Last Mile 완성**과 **초기 아키텍처 설정**을 효율적으로 수행할 수 있게 합니다.

### 핵심 기능

```
┌─────────────────────────────────────┐
│  VS Code / Cursor                   │
├─────────────────────────────────────┤
│  1. Workspace Jump                  │
│     - Command Palette에서 선택      │
│     - 워크스페이스 → 새 윈도우       │
│                                     │
│  2. Secondary Sidebar Chat (Preview)│
│     - VS Code 내 Mux 채팅           │
│     - 실시간 동기화                  │
│                                     │
│  3. SSH Workspace Support           │
│     - Remote-SSH 자동 연동           │
└─────────────────────────────────────┘
```

---

## 설치

### 다운로드

[GitHub Releases](https://github.com/coder/mux/releases)에서 최신 `.vsix` 파일 다운로드:

```bash
# 예시
mux-vscode-extension-0.x.x.vsix
```

### 명령줄 설치

#### VS Code

```bash
code --install-extension mux-*.vsix
```

#### Cursor

```bash
cursor --install-extension mux-*.vsix
```

### UI에서 설치

```
1. Command Palette (⌘+Shift+P / Ctrl+Shift+P)
2. "Extensions: Install from VSIX..."
3. 다운로드한 .vsix 파일 선택
```

---

## 워크스페이스 점프

### 기본 사용법

```
1. Command Palette 열기 (⌘+Shift+P)
2. "mux: Open Workspace" 입력
3. 워크스페이스 선택
4. 새 VS Code 윈도우에서 열림
```

### 워크스페이스 목록 표시

```
┌───────────────────────────────────────┐
│  Mux: Open Workspace                  │
├───────────────────────────────────────┤
│  📁 [my-app] feature-auth-x7k2        │
│  📁 [my-app] fix-bug-p3m9             │
│  🔗 [my-app] deploy-staging (ssh: staging-server)
│  📁 [other-project] explore-codebase  │
└───────────────────────────────────────┘
```

#### 아이콘 의미

| 아이콘 | 워크스페이스 타입 |
|-------|------------------|
| 📁 | Local 또는 Worktree |
| 🔗 | SSH |

### 커스텀 키바인딩 (선택사항)

```json
// settings.json 또는 keybindings.json
{
  "key": "cmd+shift+o",  // 원하는 단축키
  "command": "mux.openWorkspace"
}
```

---

## Secondary Sidebar Chat (Preview)

> **경고**: 프리뷰 기능으로 버그 및 변경사항 예상

### 개요

VS Code의 **Secondary Sidebar**에서 Mux 채팅을 직접 사용할 수 있습니다.

```
┌─────────────────────────────────────┐
│  VS Code                            │
├─────────┬───────────────────┬───────┤
│ Primary │  Editor           │ Second│
│ Sidebar │                   │ Sidebar│
│         │                   │       │
│ Files   │  src/auth.ts      │ mux   │
│ Search  │                   │ Chat  │
│ Git     │  function auth()  │ (Prev)│
│         │  { ... }          │       │
└─────────┴───────────────────┴───────┘
```

### 활성화

```
1. VS Code 우측 사이드바 열기
2. "mux" 컨테이너 찾기
3. "Chat (Preview)" 선택
4. 워크스페이스 선택 (드롭다운)
```

### 요구사항

```
Mux가 서버/API 모드로 실행 중이어야 함

# 데스크톱 앱 실행 중이면 자동 연결
# 또는 명령줄로 서버 시작
npx mux server --port 3000
```

### 기능

```
- 채팅 메시지 전송
- 도구 호출 표시
- 스트리밍 응답
- 워크스페이스 전환 (드롭다운)
- 새 윈도우로 열기 (연필 아이콘)
```

### 제한사항 (Preview)

```
- Markdown 렌더링 제한적
- 일부 UI 요소 누락 가능
- 성능 최적화 미완
- 버그 발생 가능

문제 발생 시:
https://github.com/coder/mux/issues
```

---

## SSH 워크스페이스 지원

### Remote-SSH 확장 필요

#### VS Code

```bash
# 확장 ID
ms-vscode-remote.remote-ssh

# 설치
code --install-extension ms-vscode-remote.remote-ssh
```

#### Cursor

```bash
# 확장 ID
anysphere.remote-ssh

# 설치 (Cursor 마켓플레이스에서)
```

> **자동 감지**: Mux 확장이 자동으로 설치된 Remote-SSH 확장 감지

### SSH 호스트 설정

#### ~/.ssh/config

```bash
# ~/.ssh/config
Host staging-server
  HostName 192.168.1.100
  User deploy
  IdentityFile ~/.ssh/id_ed25519
  Port 22

Host build-server
  HostName build.example.com
  User ci
  IdentityFile ~/.ssh/build_key
```

#### Remote-SSH UI

```
1. Command Palette (⌘+Shift+P)
2. "Remote-SSH: Add New SSH Host..."
3. ssh user@hostname
4. ~/.ssh/config 자동 업데이트
```

### SSH 워크스페이스 열기

```
1. Mux에서 SSH 워크스페이스 생성
   - 런타임: SSH
   - Host: staging-server

2. VS Code에서 "mux: Open Workspace"
3. SSH 워크스페이스 선택
4. Remote-SSH가 자동으로 연결
5. 원격 디렉토리가 새 윈도우에서 열림
```

### 동작 흐름

```
Mux 확장
    ↓
Remote-SSH 확장 호출
    ↓
SSH 연결 (user@hostname)
    ↓
워크스페이스 디렉토리로 이동
    ↓
새 VS Code 윈도우 열림
```

---

## 워크플로우 예시

### 워크플로우 1: Plan → Review → Exec

```
1. Mux (Plan 모드)
   사용자: "Add OAuth2 authentication"
   에이전트: [플랜 작성]

2. VS Code 확장
   ⌘+Shift+P → "mux: Open Workspace" → feature-auth-x7k2
   → VS Code에서 플랜 파일 검토 (~/.mux/plans/...)
   → 수정 및 저장

3. Mux (Exec 모드)
   에이전트: [플랜 변경 감지] → 구현 시작

4. VS Code
   실시간으로 파일 변경 확인
   필요 시 수정 (Last Mile)
```

### 워크플로우 2: Explore → Implement in IDE

```
1. Mux (Ask 모드)
   사용자: "Where is the database connection logic?"
   에이전트: [Explore 서브에이전트] → 결과 보고

2. VS Code 확장
   ⌘+Shift+P → "mux: Open Workspace" → explore-db-x7k2
   → 보고서에 나온 파일 열기
   → 코드 이해

3. VS Code (수동)
   직접 코드 수정 (에이전트 도움 없이)
```

### 워크플로우 3: Mux → IDE → Mux (반복)

```
1. Mux (Exec 모드)
   에이전트: [초기 구현] → 커밋

2. VS Code
   ⌘+Shift+P → "mux: Open Workspace"
   → UI 확인 (npm run dev)
   → 스타일 미세 조정 (CSS)
   → 커밋

3. Mux
   사용자: "Add validation to the form"
   에이전트: [추가 구현]

4. VS Code
   → 최종 검증 및 조정
```

---

## Secondary Sidebar Chat 활용

### 사용 사례 1: 빠른 질문

```
VS Code 편집 중
→ Secondary Sidebar Chat 열기
→ "What does this function do?" + 코드 붙여넣기
→ 즉시 답변 확인 (윈도우 전환 없음)
```

### 사용 사례 2: 인라인 수정 요청

```
VS Code 편집 중
→ Secondary Sidebar Chat
→ "Fix the TypeScript error in line 42"
→ 에이전트가 파일 수정
→ VS Code에서 즉시 확인
```

### 사용 사례 3: 워크스페이스 전환

```
Secondary Sidebar Chat
→ 드롭다운에서 다른 워크스페이스 선택
→ 컨텍스트 즉시 전환
→ 연필 아이콘 클릭 → 새 윈도우로 열기
```

---

## 커맨드 참조

### Mux 확장 커맨드

| 커맨드 | 설명 |
|--------|------|
| `mux.openWorkspace` | 워크스페이스를 새 윈도우에서 열기 |
| `mux.refreshWorkspaces` | 워크스페이스 목록 새로고침 |
| `mux.openChat` | Secondary Sidebar Chat 열기 |

### 사용 예시

```
Command Palette (⌘+Shift+P)
→ "mux: Open Workspace"
→ "mux: Refresh Workspaces"
→ "mux: Open Chat"
```

---

## 문제 해결

### 워크스페이스가 목록에 나타나지 않음

```bash
# 원인: Mux 앱 미실행 또는 연결 실패

# 해결책 1: Mux 앱 실행 확인
open -a Mux  # macOS
./Mux.AppImage  # Linux

# 해결책 2: 워크스페이스 목록 새로고침
Command Palette → "mux: Refresh Workspaces"

# 해결책 3: 확장 재시작
Command Palette → "Developer: Reload Window"
```

### SSH 워크스페이스 연결 실패

```bash
# 원인: Remote-SSH 확장 미설치

# 해결책: 확장 설치
# VS Code
code --install-extension ms-vscode-remote.remote-ssh

# Cursor
# Cursor 마켓플레이스에서 "Remote-SSH" 검색 및 설치
```

### SSH 호스트 미설정

```bash
# ~/.ssh/config 확인
cat ~/.ssh/config

# 호스트 추가
cat >> ~/.ssh/config <<EOF
Host myserver
  HostName 192.168.1.100
  User deploy
  IdentityFile ~/.ssh/id_ed25519
EOF

# 또는 Remote-SSH UI 사용
Command Palette → "Remote-SSH: Add New SSH Host..."
```

### Secondary Sidebar Chat 연결 실패

```bash
# 원인: Mux 서버 미실행

# 해결책 1: 데스크톱 앱 실행
open -a Mux

# 해결책 2: 서버 모드 실행
npx mux server --port 3000

# 포트 확인
lsof -i :3000
```

---

## 고급 설정

### Workspace Ignore 패턴

```jsonc
// .vscode/settings.json (프로젝트별)
{
  "mux.workspaceIgnorePatterns": [
    "node_modules/**",
    ".git/**",
    "dist/**"
  ]
}
```

### Custom Server URL

```jsonc
// settings.json
{
  "mux.serverUrl": "http://localhost:3000"  // 기본값
}
```

### SSH Remote 설정

```jsonc
// settings.json
{
  "remote.SSH.configFile": "~/.ssh/config",
  "remote.SSH.connectTimeout": 30
}
```

---

## 개발 워크플로우 최적화

### 패턴 1: Mux First, IDE Second

```
1. Mux에서 초기 구현 (70-80%)
2. VS Code로 전환 (워크스페이스 점프)
3. UI/스타일 미세 조정 (20-30%)
4. Mux로 돌아가서 테스트 작성 요청
```

### 패턴 2: IDE First, Mux Second

```
1. VS Code에서 아키텍처 설계 (파일 구조)
2. Mux에서 구현 요청
3. VS Code로 전환하여 검증
4. Mux에서 추가 기능 요청
```

### 패턴 3: 병렬 작업

```
VS Code 윈도우 1: feature-auth-x7k2 (수동 UI 작업)
VS Code 윈도우 2: fix-bug-p3m9 (Mux 에이전트 작업 모니터링)
Mux: 워크스페이스 2에서 에이전트 실행

→ 윈도우 2에서 실시간 변경사항 확인
→ 필요 시 개입 또는 승인
```

---

## 다음 단계

VS Code 통합을 마스터했다면:

1. **[챕터 07: 고급 기능](/blog-repo/mux-guide-07-advanced-features)** - Opportunistic Compaction, Mode Prompts, Instruction Files
2. **[챕터 08: 개발 및 확장](/blog-repo/mux-guide-08-development)** - Mux 자체 개발 및 커스터마이징
3. **워크플로우 실험** - 자신만의 Mux + VS Code 패턴 찾기

---

## 참고 자료

- [VS Code Extension 문서](https://mux.coder.com/integrations/vscode-extension)
- [VS Code Remote-SSH](https://code.visualstudio.com/docs/remote/ssh)
- [Cursor 문서](https://cursor.sh/docs)
- [GitHub Repository](https://github.com/coder/mux)
