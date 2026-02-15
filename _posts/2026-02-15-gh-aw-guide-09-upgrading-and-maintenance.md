---
layout: post
title: "GitHub Agentic Workflows(gh-aw) 가이드 (09) - 업그레이드/유지: upgrade, fix, 그리고 락파일 운영"
date: 2026-02-15
permalink: /gh-aw-guide-09-upgrading-and-maintenance/
author: GitHub
categories: [개발 도구, GitHub]
tags: [gh-aw, Upgrade, Codemods, fix, Lockfiles, Maintenance]
original_url: "https://github.github.com/gh-aw/guides/upgrading/"
excerpt: "gh-aw는 스키마/보안 가드레일이 계속 진화합니다. upgrade/fix/compile을 묶어서 안전하게 업데이트하고, .md/.lock.yml을 일관되게 운영하는 방법을 정리합니다."
---

## 왜 업그레이드가 중요한가

gh-aw는 “에이전트 워크플로우”라는 성격상,
보안/검증 계층이 제품의 핵심입니다.

그래서 버전 업은 단순 기능 추가가 아니라:

- deprecated 필드 제거
- 보안 강화(strict)
- 스키마 변화(컴파일 실패로 드러남)

같은 “운영 안전성” 변화가 섞입니다.

---

## 기본 업그레이드: gh aw upgrade

문서의 기본 업그레이드 흐름은 한 커맨드로 묶입니다.

```bash
gh aw upgrade
```

이 명령은 보통 다음을 수행합니다.

- 에이전트/프롬프트 파일 업데이트
- codemod 기반 마이그레이션 적용
- 전체 워크플로우 컴파일로 `.lock.yml` 갱신

---

## 고장난 문법/구식 필드는 fix로 먼저

워크플로우 문법이 바뀌거나 deprecated 필드가 남아 있으면 컴파일이 흔들립니다.

이 때는:

```bash
gh aw fix --write
gh aw compile --validate --strict
```

같은 루틴이 실전적입니다.

---

## 운영 원칙: .md와 .lock.yml은 같이 관리

업그레이드/유지에서 가장 흔한 실수는 이겁니다.

- `.md`만 바꿔놓고 `.lock.yml`을 안 올림
- `.lock.yml`을 직접 편집함
- orphaned lock file이 쌓임

권장 규칙:

1. frontmatter 수정 후엔 항상 `gh aw compile`
2. `.md`와 `.lock.yml`을 함께 커밋
3. 삭제한 워크플로우는 `gh aw compile --purge`로 정리

---

*다음 글에서는 설치/정책/컴파일/도구에서 자주 마주치는 문제들을 트러블슈팅 관점에서 정리합니다.*

