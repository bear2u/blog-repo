---
name: divine-mobile-guide-creator
description: |
  https://github.com/divinevideo/divine-mobile (diVine/OpenVine) 레포를 분석해서
  이 저장소(/home/blog)의 Jekyll 블로그에 "한국어 챕터형 가이드 시리즈"를 생성/업데이트한다.

  사용 시점:
  - "divine-mobile(diVine/OpenVine) 내용을 블로그 가이드 시리즈로 써줘"
  - Flutter 기반 diVine/OpenVine 모바일 앱의 구조/빌드/배포/업로드/노스트르(Nostr) 연동을 챕터로 정리해줘
  - divine-mobile 레포 변경을 반영해 기존 가이드 시리즈를 업데이트해줘
---

# Divine Mobile Guide Creator

## Overview

`divinevideo/divine-mobile` 레포를 기반으로, 이 블로그의 관례에 맞춘 "챕터형 한국어 가이드 시리즈"를 생성/갱신한다.
결과물은 `_posts/` 챕터 글, 시리즈 인덱스(루트 페이지), 그리고 `/home/blog/_tabs/guides.md` 카드 추가까지 포함한다.

## Outputs (this repo conventions)

- 챕터 포스트: `/home/blog/_posts/YYYY-MM-DD-divine-mobile-guide-XX-{slug}.md`
- 시리즈 인덱스(루트 페이지): `/home/blog/divine-mobile-guide.md`
- 가이드 목록 카드: `/home/blog/_tabs/guides.md`

## Inputs to Confirm (ask user)

- `AUTHOR` (front matter), `PROJECT_NAME` (타이틀용, 기본: `diVine`)
- `series` 슬러그(기본: `divine-mobile`)
- 챕터 수: 자동(권장) 또는 지정(N개)
- guides 탭 섹션: 애매하면 `개발 도구` 또는 `모바일`로 분류

## Workflow

### 1) Collect sources (clone)

```bash
SCRATCH="$(mktemp -d /tmp/codex-divine-mobile.XXXXXX)"
cd "$SCRATCH"
git clone --depth 1 https://github.com/divinevideo/divine-mobile
cd divine-mobile
```

우선 분석 대상:
- `README.md`
- `docs/README.md`, `docs/ARCHITECTURE.md`
- `docs/CF_STREAM_SETUP.md`, `docs/VIDEO_UPLOAD_ARCHITECTURE-*.md`
- `mobile/pubspec.yaml`, `mobile/lib/` (서비스/프로바이더/업로드 파이프라인)

### 2) Design chapter plan

권장 초안은 `references/chapter-map.md`를 참고하고, 레포의 실제 문서 구조에 맞춰 8-14개 범위로 조정한다.

### 3) Write posts using existing blog templates

이 블로그 템플릿은 기존 스킬을 그대로 재사용한다:
- 포스트 템플릿: `/home/blog/.codex/skills/blog-guide-creator/references/post-template.md`
- 시리즈 인덱스: `/home/blog/.codex/skills/blog-guide-creator/references/index-template.md`
- 네비/목차: `/home/blog/.codex/skills/blog-guide-creator/references/nav-template.md`

파일명/퍼머링크 규칙:
- 날짜: `CURRENT_DATE=$(date +%Y-%m-%d)`
- `_posts/${CURRENT_DATE}-divine-mobile-guide-${num}-${slug}.md`
- permalink: `/divine-mobile-guide-${num}-${slug}/`

주의(보안/저작권):
- 레포 문서에 `CF_STREAM_TOKEN` 예시 값이 있어도 블로그에는 실제 토큰을 절대 그대로 싣지 말고 `<YOUR_CF_STREAM_TOKEN>` 형태로 마스킹한다.
- 문서 원문을 길게 복붙하지 말고, 핵심을 요약/재구성하고 코드/명령은 필요한 최소만 인용한다.

### 4) Update guides tab

`/home/blog/_tabs/guides.md`에서 적절한 섹션의 `guide-grid` 맨 위에 카드(최신)를 추가한다.

### 5) Validate and clean up

```bash
cd /home/blog
git status
```

가능하면:

```bash
bundle exec jekyll build
```

정리:

```bash
rm -rf "$SCRATCH"
```

## Git workflow

기본 동작은 `commit` 후 `push`까지 완료한다. 사용자가 "push는 하지 마"라고 하면 `push`만 생략한다.

```bash
cd /home/blog
git add _posts/*divine-mobile-guide-*.md divine-mobile-guide.md _tabs/guides.md
git commit -m "diVine(divine-mobile) 가이드 시리즈 추가"
git push
```
