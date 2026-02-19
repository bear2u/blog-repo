# 인덱스 페이지 템플릿

## 파일 위치
블로그 루트: `{series}-guide.md`

## 마크다운 템플릿

```markdown
---
layout: page
title: {프로젝트명} 가이드
permalink: /{series}-guide/
icon: fas fa-{아이콘}
---

# {프로젝트명} 완벽 가이드

> **{한 줄 설명}**

**{프로젝트명}**은 {2-3문장 소개}.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/{series}-guide-01-intro/) | 프로젝트란? 주요 특징 |
| 02 | [설치 및 시작](/blog-repo/{series}-guide-02-installation/) | 요구사항, 설치 방법 |
| 03 | [아키텍처](/blog-repo/{series}-guide-03-architecture/) | 전체 구조, 설계 원칙 |
| 04 | [{핵심모듈1}](/blog-repo/{series}-guide-04-{slug}/) | {설명} |
| 05 | [{핵심모듈2}](/blog-repo/{series}-guide-05-{slug}/) | {설명} |
| 06 | [{핵심모듈3}](/blog-repo/{series}-guide-06-{slug}/) | {설명} |
| 07 | [{API/라우터}](/blog-repo/{series}-guide-07-{slug}/) | {설명} |
| 08 | [{외부연동}](/blog-repo/{series}-guide-08-{slug}/) | {설명} |
| 09 | [{UI}](/blog-repo/{series}-guide-09-{slug}/) | {설명} |
| 10 | [확장 및 커스터마이징](/blog-repo/{series}-guide-10-customization/) | 설정, 확장 방법 |

---

## 주요 특징

- **{특징1}** - {설명}
- **{특징2}** - {설명}
- **{특징3}** - {설명}
- **{특징4}** - {설명}
- **{특징5}** - {설명}

---

## 빠른 시작

```bash
# 설치
{설치 명령어}

# 실행
{실행 명령어}
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    {프로젝트명}                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   {ASCII 다이어그램}                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| {기술1} | {용도} |
| {기술2} | {용도} |
| {기술3} | {용도} |

---

## 관련 링크

- [GitHub 저장소]({github-url})
- [공식 문서]({docs-url})
- [Discord/Community]({community-url})
```

## 아이콘 참고

| 프로젝트 유형 | 아이콘 |
|--------------|--------|
| 터미널/CLI | `fa-terminal` |
| AI/에이전트 | `fa-robot` |
| 데이터베이스 | `fa-database` |
| 개발 도구 | `fa-tools` |
| 웹 | `fa-globe` |
| 모바일 | `fa-mobile` |
| 문서 | `fa-book` |
| 코드 | `fa-code` |

## 예시

실제 생성된 인덱스 페이지 참고:
- `/home/blog/superset-guide.md`
- `/home/blog/wrenai-guide.md`
- `/home/blog/ui-tars-guide.md`
