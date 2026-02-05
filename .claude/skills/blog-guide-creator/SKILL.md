---
name: blog-guide-creator
description: |
  GitHub 레포지토리나 블로그 URL을 분석하여 챕터별 가이드 시리즈를 생성하는 스킬.

  사용 시점:
  - 사용자가 GitHub 레포지토리 URL을 주고 "가이드 만들어줘", "분석해서 블로그 작성해줘" 요청 시
  - "소스 분석해서 챕터별로 작성해줘" 요청 시
  - 외부 블로그/문서 URL을 주고 "번역해서 시리즈로 만들어줘" 요청 시
  - 기존 블로그에 새로운 가이드 시리즈 추가 요청 시

  생성 결과물:
  - 메인 네비게이션 드롭다운 메뉴
  - 가이드 인덱스 페이지 (전체 목차)
  - 챕터별 개별 포스트 파일들
  - 시리즈 이전/다음 네비게이션
---

# Blog Guide Creator - 완전 자동화 워크플로우

GitHub 레포지토리 URL을 받으면 자동으로 분석하여 블로그 가이드 시리즈를 생성합니다.

## 자동화 워크플로우

### Phase 1: 레포지토리 클론 및 분석 (자동)

```bash
# 1. /tmp에 레포지토리 클론
cd /tmp && rm -rf {repo-name} && git clone --depth 1 {github-url}

# 2. 분석할 파일들 확인
ls -la /tmp/{repo-name}/
find /tmp/{repo-name} -maxdepth 2 -type f -name "*.md" | head -20
find /tmp/{repo-name} -maxdepth 2 -type d | head -30
```

### Phase 2: 핵심 파일 분석 (자동)

필수 분석 파일:
1. `README.md` - 프로젝트 소개, 기능, 설치 방법
2. `package.json` / `Cargo.toml` / `go.mod` 등 - 의존성, 스크립트
3. `AGENTS.md` / `CLAUDE.md` / `CONTRIBUTING.md` - 개발 가이드
4. 핵심 소스 디렉토리 구조

분석 포인트:
- **프로젝트 목적**: 무엇을 하는 도구/라이브러리인가?
- **기술 스택**: 언어, 프레임워크, 주요 라이브러리
- **아키텍처**: 디렉토리 구조, 모듈 분리 방식
- **핵심 기능**: 주요 기능 5-10개 파악
- **설치/실행 방법**: 필요한 의존성, 명령어

### Phase 3: 챕터 구조 자동 설계

기본 10개 챕터 템플릿:

| # | 제목 | 내용 |
|---|------|------|
| 01 | 소개 및 개요 | 프로젝트란?, 주요 특징, 왜 사용하는가 |
| 02 | 설치 및 시작 | 요구사항, 설치 방법, 빠른 시작 |
| 03 | 아키텍처 분석 | 전체 구조, 디렉토리, 설계 원칙 |
| 04 | 핵심 모듈 1 | 가장 중요한 모듈 상세 분석 |
| 05 | 핵심 모듈 2 | 두 번째 주요 모듈 |
| 06 | 핵심 모듈 3 | 세 번째 주요 모듈 |
| 07 | API/라우터 | API 구조, 엔드포인트 |
| 08 | 외부 연동 | 외부 서비스, 플러그인 |
| 09 | UI/프론트엔드 | UI 컴포넌트, 상태 관리 |
| 10 | 확장 및 커스터마이징 | 설정, 확장 방법, 결론 |

### Phase 4: 포스트 파일 자동 생성

각 챕터별 파일 생성:
- 파일명: `_posts/YYYY-MM-DD-{series}-guide-{part번호}-{slug}.md`
- 예: `_posts/2025-02-05-superset-guide-01-intro.md`

Front Matter 템플릿:
```yaml
---
layout: post
title: "{프로젝트명} 완벽 가이드 ({번호}) - {챕터 제목}"
date: {오늘날짜}
permalink: /{series}-guide-{part번호}-{slug}/
author: {원저자/팀}
categories: [{카테고리1}, {카테고리2}]
tags: [{태그1}, {태그2}, ...]
original_url: "{GitHub URL}"
excerpt: "{한 줄 요약}"
---
```

### Phase 5: 인덱스 페이지 생성

파일: `{series}-guide.md` (루트 디렉토리)

```markdown
---
layout: page
title: {프로젝트명} 가이드
permalink: /{series}-guide/
icon: fas fa-{아이콘}
---

# {프로젝트명} 완벽 가이드

> **{한 줄 설명}**

{프로젝트 소개 2-3문장}

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/{series}-guide-01-intro/) | ... |
| 02 | [설치](/blog-repo/{series}-guide-02-installation/) | ... |
...

---

## 주요 특징

- **특징1** - 설명
- **특징2** - 설명
...

---

## 빠른 시작

```bash
# 설치/실행 명령어
```

---

## 관련 링크

- [GitHub 저장소]({github-url})
- [공식 문서]({docs-url})
```

### Phase 6: 가이드 목록에 추가

파일: `_tabs/guides.md`

적절한 섹션에 새 가이드 카드 추가:
```html
<div class="guide-card">
  <span class="badge badge-{카테고리}">카테고리</span>
  <h3><a href="/blog-repo/{series}-guide/">{프로젝트명}</a></h3>
  <p>{한 줄 설명}</p>
</div>
```

### Phase 7: Git Commit & Push

```bash
git add _posts/{series}-guide-*.md {series}-guide.md _tabs/guides.md
git commit -m "{프로젝트명} 가이드 추가 - {한 줄 설명}

- {챕터수}개 챕터로 구성된 완벽 가이드 시리즈
- 인덱스 페이지 및 가이드 네비게이션 추가

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
git push origin main
```

---

## 명명 규칙

| 항목 | 형식 | 예시 |
|-----|------|-----|
| 시리즈 slug | `{project}-guide` | `superset-guide` |
| 포스트 파일 | `YYYY-MM-DD-{series}-{part}-{slug}.md` | `2025-02-05-superset-guide-01-intro.md` |
| permalink | `/{series}-{part}-{slug}/` | `/superset-guide-01-intro/` |
| 인덱스 파일 | `{series}.md` | `superset-guide.md` |

---

## 카테고리 분류

| 카테고리 | badge class | 예시 프로젝트 |
|---------|-------------|--------------|
| AI 코딩 | `badge-coding` | Claude Code, OpenCode, Ralph, Superset |
| AI 에이전트 | `badge-agent` | WrenAI, UI-TARS, OpenClaw, Beads |
| 개발 도구 | `badge-tool` | Maestro, TrendRadar, RS-SDK |
| LLM | `badge-llm` | MiniMind |

---

## 체크리스트

생성 완료 후 자동 확인:
- [ ] 모든 포스트 파일 생성됨 (`_posts/{series}-guide-*.md`)
- [ ] 인덱스 페이지 생성됨 (`{series}-guide.md`)
- [ ] _tabs/guides.md에 새 가이드 추가됨
- [ ] git commit & push 완료

---

## 참조 파일

- `references/chapter-structure.md` - 챕터 구성 가이드
- `references/post-template.md` - 포스트 템플릿
- `references/index-template.md` - 인덱스 템플릿
- `references/nav-template.md` - 네비게이션 템플릿

---

## 예시 실행

입력:
```
https://github.com/superset-sh/superset
```

출력:
```
✅ 레포지토리 클론 완료: /tmp/superset
✅ 분석 완료: CLI 코딩 에이전트를 위한 터보차지 터미널
✅ 10개 챕터 구조 설계 완료
✅ 포스트 파일 10개 생성 완료
✅ 인덱스 페이지 생성: superset-guide.md
✅ guides.md 업데이트 완료
✅ Git commit & push 완료: 34447ff
```
