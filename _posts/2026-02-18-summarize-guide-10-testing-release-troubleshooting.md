---
layout: post
title: "Summarize 가이드 (10) - 테스트/릴리즈/트러블슈팅: 실전 운영 체크리스트"
date: 2026-02-18
permalink: /summarize-guide-10-testing-release-troubleshooting/
author: steipete
categories: ['개발 도구', '트러블슈팅']
tags: [Vitest, Playwright, Release, Daemon, Troubleshooting]
original_url: "https://github.com/steipete/summarize/blob/main/RELEASING.md"
excerpt: "대규모 테스트 세트와 릴리즈 문서를 기반으로 운영 시 자주 만나는 문제와 점검 순서를 정리합니다."
---

## 테스트 구조

저장소는 `tests/`에 매우 큰 범위의 테스트를 갖고 있습니다.

- CLI 입력/플래그/출력 테스트
- daemon/extension 흐름 테스트
- cache/transcript/youtube/slides 세부 테스트
- 일부 live 테스트

기본 검증 커맨드:

```bash
pnpm test
pnpm check
```

---

## 릴리즈 관점

루트 문서와 AGENTS 가이드 기준으로 모노레포 릴리즈는 lockstep 성격을 가집니다.

- `@steipete/summarize-core` 먼저
- CLI 패키지 이후

버전/배포 절차는 `RELEASING.md`, `scripts/release.sh`를 기준으로 따라가는 구조입니다.

---

## 자주 만나는 운영 이슈

문서에서 반복되는 이슈 패턴:

1. daemon unreachable
2. extension과 daemon 토큰 불일치
3. yt-dlp/ffmpeg/tesseract 미설치
4. 브라우저 site access/권한 문제
5. provider API key 누락

우선 점검 명령:

```bash
summarize daemon status
summarize daemon restart
```

---

## 소스 기준 체크 포인트

실무 적용 전 아래를 확인하면 안정성이 올라갑니다.

- Node 22+ 환경 통일
- `~/.summarize/config.json` 정책화
- 팀 공통 `.env` 템플릿 관리
- OpenRouter/free preset 사용 시 fallback 정책 명시
- media 도구 설치 가이드(yt-dlp/ffmpeg/tesseract) 사전 배포

이 체크리스트를 기준으로 잡으면 CLI 단독 사용부터 브라우저 확장 연동까지 안정적으로 운영할 수 있습니다.

