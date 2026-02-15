---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (02) - 설치/초기화: gh extension, init, 엔진/시크릿"
date: 2026-02-15
permalink: /gh-aw-guide-02-install-and-init/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, gh, Installation, init, Copilot, Claude, Codex, Secrets]
original_url: "https://github.github.com/gh-aw/setup/cli/"
excerpt: "gh-aw 설치 방법(gh extension/standalone installer)과 레포 초기화(gh aw init), 엔진과 시크릿 설정 흐름을 정리합니다."
---

## 설치: gh extension이 기본

문서의 기본 설치는 GitHub CLI 확장 설치입니다.

```bash
gh extension install github/gh-aw
gh aw version
```

팀/프로덕션에서 버전 고정이 필요하면 태그/커밋으로 설치를 핀할 수 있습니다.

---

## 설치가 실패하면: standalone installer

Codespaces/제한된 네트워크/인증 문제 등으로 확장 설치가 실패할 수 있습니다.
이 때 문서는 standalone 설치 스크립트를 안내합니다.

```bash
curl -sL https://raw.githubusercontent.com/github/gh-aw/main/install-gh-aw.sh | bash
gh aw version
```

---

## 레포 초기화: gh aw init

`gh aw init`은 레포를 “에이전트 워크플로우를 굴릴 준비” 상태로 맞추는 명령입니다.

대표적으로 아래 성격의 파일/설정을 만듭니다.

- `.gitattributes`(예: `.lock.yml`을 generated로 취급)
- `.github/aw/` 문서/프롬프트들
- `.github/agents/` 에이전트 파일(엔진별 커스텀 에이전트 등)
- (옵션) MCP 연동 구성, VSCode 설정 등

실행 예:

```bash
gh aw init --engine copilot
```

---

## 엔진(Engine) 선택과 시크릿

gh-aw는 엔진(코딩 에이전트)을 선택할 수 있습니다.

- Copilot CLI(기본)
- Claude
- Codex

각 엔진은 필요한 API 키/토큰이 다릅니다. 문서의 방향은:

1. `gh aw init`으로 엔진을 선택하고
2. 필요한 시크릿을 `gh aw secrets ...`로 설정
3. 워크플로우를 추가/작성하고 컴파일

Copilot의 경우 “copilot-requests” 범위가 있는 PAT 같은 요구사항이 있을 수 있으니, 문서의 엔진 섹션을 함께 보는 편이 안전합니다.

---

## 다음 단계 체크리스트

1. `gh aw version`으로 설치 확인
2. `gh aw init`으로 레포 초기화
3. 워크플로우 추가/생성(`gh aw add`, `gh aw new`)
4. 컴파일(`gh aw compile`)

---

*다음 글에서는 워크플로우(.md)의 구조와 frontmatter를 기준으로 “어디를 수정하면 컴파일이 필요한지”까지 같이 정리합니다.*

