---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (01) - 소개 및 개요"
date: 2026-02-13
permalink: /divine-mobile-guide-01-intro/
author: divinevideo
categories: [모바일, Flutter]
tags: [diVine, OpenVine, Nostr, Flutter, Short-form Video]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "Nostr 기반 탈중앙 숏폼 비디오 앱 diVine(OpenVine)와 레포 구조를 한 번에 훑습니다."
---

## diVine(OpenVine)은 무엇인가

`divinevideo/divine-mobile`은 **Nostr 프로토콜** 위에서 동작하는 Vine 스타일의 숏폼 비디오 앱(Flutter)을 담고 있는 레포입니다.
README 기준으로는 다음 키워드로 요약할 수 있습니다.

- **탈중앙(Decentralized)**: 중앙 서버에 종속되지 않고 Nostr 릴레이를 통해 이벤트를 주고받음
- **짧은 반복 비디오**: "루프" 재생을 전제로 한 콘텐츠 경험
- **크로스 플랫폼**: iOS/Android/Web/macOS 타깃

---

## 이 레포에서 무엇을 읽어야 하나

문서와 코드가 모두 큰 편이라, 시작 지점을 정해두는 게 중요합니다.

### 문서(entry points)

- `README.md`: 제품/기능 요약 + 빌드 안내(최상단 설명)
- `docs/README.md`: 개발 셋업과 실행/빌드 스크립트 안내
- `docs/NOSTR_EVENT_TYPES.md`: 앱이 요구하는 Nostr 이벤트(kind) 정리
- `docs/nip-vine.md`: kind `32222`(NIP-32222) 스펙 초안
- `docs/CF_STREAM_SETUP.md`: 업로드 구성에 대한 안내(단, 코드와 불일치할 수 있음)

### 코드(entry points)

- `mobile/lib/main.dart`: 앱 엔트리포인트
- `mobile/lib/providers/`: Riverpod provider 레이어(서비스 조립)
- `mobile/lib/services/`: 업로드/피드/소셜/인증/모더레이션 등 도메인 로직

---

## 레포 전체 구조(큰 그림)

실제 파일은 더 많지만, “가이드 읽기” 관점에서는 아래 3개만 잡고 가도 충분합니다.

```text
divine-mobile/
  docs/               # 아키텍처/운영/테스트 문서
  mobile/             # Flutter 앱(실제 제품)
  website/            # 웹 관련(별도 구성)
```

이 시리즈는 기본적으로 **`mobile/` 중심**으로 설명합니다. 다만, `docs/`는 “왜 이렇게 구성했는지”를 설명하는 자료로 자주 인용됩니다.

---

## 이 시리즈의 목표

이 가이드 시리즈는 “코드베이스를 처음 보는 사람”이 다음 질문에 답할 수 있게 만드는 것이 목표입니다.

- `mobile/`에서 **어떻게 실행**하고, **어디를 만져야** 하는가?
- Nostr 이벤트(kind) 관점에서 **무엇을 구독/발행**하는가?
- 비디오 업로드는 어디서 처리되고, 퍼블리시는 어떤 이벤트로 나가는가?
- 피드/페이지네이션/모더레이션은 어떤 서비스 조합으로 동작하는가?

---

## 주의: 문서와 코드가 어긋날 수 있다

레포 문서에는 `CF_STREAM_TOKEN`을 주입해 업로드를 구성하라는 안내가 있습니다(`docs/README.md`, `docs/CF_STREAM_SETUP.md`).
하지만 최신 `mobile/run_dev.sh`는 `.env` 기반 `--dart-define`을 구성하며, 코드 전체에서 `CF_STREAM_TOKEN` 문자열은 문서에만 등장합니다.

즉, “문서에 적힌 그림”과 “현재 코드의 실제 흐름”이 다를 수 있습니다.
이 시리즈에서는 문서를 소개하되, 각 챕터에서 **실제 코드 파일 경로**를 함께 찍어가며 정리합니다.

---

*다음 글에서는 개발 환경 구성과 `mobile/` 실행 흐름(`run_dev.sh`)부터 시작합니다.*

