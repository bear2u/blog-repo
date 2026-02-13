---
layout: post
title: "GSD (Get Shit Done) 완벽 가이드 (07) - 보안 및 문제 해결"
date: 2026-02-13
permalink: /gsd-guide-07-security-troubleshooting/
author: TÂCHES
categories: [AI 코딩, 개발 도구]
tags: [Claude Code, Security, Troubleshooting, Best Practices]
original_url: "https://github.com/gsd-build/get-shit-done"
excerpt: "GSD 보안 설정 및 일반적인 문제 해결"
---

## 보안

### 민감한 파일 보호

GSD의 코드베이스 매핑 및 분석 명령어는 프로젝트를 이해하기 위해 파일을 읽습니다. **비밀이 포함된 파일을 보호**하려면 Claude Code의 거부 목록에 추가하세요.

1. Claude Code 설정 열기 (`.claude/settings.json` 또는 전역)
2. 민감한 파일 패턴을 거부 목록에 추가:

```json
{
  "permissions": {
    "deny": [
      "Read(.env)",
      "Read(.env.*)",
      "Read(**/secrets/*)",
      "Read(**/*credential*)",
      "Read(**/*.pem)",
      "Read(**/*.key)"
    ]
  }
}
```

이렇게 하면 실행하는 명령어와 관계없이 Claude가 이 파일을 완전히 읽지 못합니다.

> [!IMPORTANT]
> GSD에는 비밀 커밋을 방지하는 기본 보호 기능이 포함되어 있지만, **심층 방어(defense-in-depth)**가 모범 사례입니다. 민감한 파일에 대한 읽기 액세스 거부를 최전방 방어선으로 사용하세요.

---

## 문제 해결

### 설치 후 명령어를 찾을 수 없음

**원인:** Claude Code가 슬래시 명령어를 다시 로드하지 않음

**해결:**
1. Claude Code 재시작
2. 파일 존재 확인:
   - 전역: `~/.claude/commands/gsd/`
   - 로컬: `./.claude/commands/gsd/`

---

### 명령어가 예상대로 작동하지 않음

**해결:**
1. `/gsd:help`로 설치 확인
2. 재설치:
   ```bash
   npx get-shit-done-cc
   ```

---

### 최신 버전으로 업데이트

```bash
npx get-shit-done-cc@latest
```

---

### Docker/컨테이너 환경 문제

**증상:** 틸드 경로(`~/.claude/...`)로 파일 읽기 실패

**해결:** 설치 전에 `CLAUDE_CONFIG_DIR` 설정:

```bash
CLAUDE_CONFIG_DIR=/home/youruser/.claude npx get-shit-done-cc --global
```

이렇게 하면 컨테이너에서 제대로 확장되지 않을 수 있는 `~` 대신 절대 경로가 사용됩니다.

---

### 컨텍스트 품질 저하

**증상:** Claude의 응답 품질이 세션이 진행됨에 따라 저하됨

**원인:** 컨텍스트 윈도우가 채워짐

**해결:**
1. GSD 워크플로우 사용 - 자동으로 컨텍스트 관리
2. 각 단계가 별도의 컨텍스트에서 실행됨
3. `/gsd:pause-work`와 `/gsd:resume-work`로 세션 분할

---

### 계획 실행 중 에러

**증상:** execute-phase 중 에러 발생

**자동 처리 규칙:**

| 규칙 | 에러 유형 | 동작 |
|------|----------|------|
| Rule 1 | 버그 | 자동 수정 |
| Rule 2 | 누락된 필수 기능 | 자동 추가 |
| Rule 3 | 차단 이슈 | 자동 해결 |
| Rule 4 | 아키텍처 변경 필요 | 사용자에게 질문 |

Rule 4의 경우 시스템이 자동으로 정지하고 사용자 결정을 요청합니다.

---

### 인증 게이트 문제

**증상:** "Not authenticated", "401", "403" 에러

**동작:** 인증 에러는 실패가 아닌 **게이트**로 처리됩니다.

**해결:**
1. 시스템이 자동으로 체크포인트 생성
2. 사용자가 인증 수행
3. `/gsd:resume-work`로 계속

---

### Git 충돌

**증상:** 실행 중 git 충돌 발생

**해결:**
1. GSD는 각 태스크를 원자적으로 커밋
2. 충돌 시 해당 태스크만 영향 받음
3. `git log`로 마지막 성공 커밋 확인
4. 필요시 해당 커밋으로 되돌리기

---

### .planning 디렉토리 문제

**증상:** STATE.md 누락 또는 .planning/ 없음

**원인:** 프로젝트가 초기화되지 않음

**해결:**
```bash
/gsd:new-project
```

또는 기존 프로젝트의 경우:
```bash
/gsd:map-codebase
/gsd:new-project
```

---

## 제거

GSD를 완전히 제거하려면:

```bash
# 전역 설치 제거
npx get-shit-done-cc --claude --global --uninstall
npx get-shit-done-cc --opencode --global --uninstall

# 로컬 설치 제거
npx get-shit-done-cc --claude --local --uninstall
npx get-shit-done-cc --opencode --local --uninstall
```

다른 설정은 유지하면서 모든 GSD 명령어, 에이전트, 훅, 설정을 제거합니다.

---

## 일반적인 에러 메시지

| 에러 | 원인 | 해결 |
|------|------|------|
| "Project not initialized" | .planning/ 없음 | `/gsd:new-project` 실행 |
| "Phase not found" | 잘못된 단계 번호 | `/gsd:progress`로 확인 |
| "Plan already executed" | 이미 완료된 계획 | 다음 단계로 진행 |
| "Context window full" | 컨텍스트 한계 초과 | 새 세션 시작 |

---

*다음 글에서는 GSD의 템플릿 시스템을 살펴봅니다.*
