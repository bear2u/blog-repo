---
layout: page
title: diVine Mobile 가이드
permalink: /divine-mobile-guide/
icon: fas fa-mobile
---

# 🎬 diVine(OpenVine) 완벽 가이드

> **Nostr 기반 탈중앙 숏폼 비디오 앱 divine-mobile 레포 정리**

**diVine(OpenVine)**은 Nostr 프로토콜 위에 구축된 Vine 스타일(짧고 반복 재생되는) 숏폼 비디오 공유 앱입니다. 이 시리즈는 `divinevideo/divine-mobile` 레포의 문서와 코드 구조를 바탕으로, 빌드부터 아키텍처, 업로드/퍼블리시, 피드/모더레이션, 테스트, 배포까지를 챕터 형태로 정리합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/divine-mobile-guide-01-intro/) | diVine란? 레포 구성, 핵심 개념 |
| 02 | [개발 환경과 빠른 시작](/blog-repo/divine-mobile-guide-02-getting-started/) | Flutter 설정, `mobile/` 실행 스크립트 |
| 03 | [레포 구조 한눈에 보기](/blog-repo/divine-mobile-guide-03-repo-structure/) | top-level 폴더, `mobile/lib/` 구성 |
| 04 | [Nostr 클라이언트 아키텍처](/blog-repo/divine-mobile-guide-04-nostr-architecture/) | `NostrClient` 구성, 릴레이/인증 흐름 |
| 05 | [이벤트 타입과 NIP-32222](/blog-repo/divine-mobile-guide-05-event-types-nip32222/) | kind 32222, imeta 태그, 주소가능 이벤트 |
| 06 | [상태 관리와 의존성 주입(Riverpod)](/blog-repo/divine-mobile-guide-06-state-management-riverpod/) | Provider 레이어, 서비스 조립 |
| 07 | [비디오 녹화 UX와 파이프라인](/blog-repo/divine-mobile-guide-07-video-recording/) | 카메라/클립, 썸네일, 사용자 흐름 |
| 08 | [업로드와 퍼블리시(비디오 이벤트 발행)](/blog-repo/divine-mobile-guide-08-video-upload-publishing/) | `UploadManager`와 `VideoEventPublisher` |
| 09 | [피드 로딩과 페이지네이션](/blog-repo/divine-mobile-guide-09-feed-pagination/) | kind 32222/16 수집, 페이징 전략 |
| 10 | [모더레이션과 신고](/blog-repo/divine-mobile-guide-10-moderation-reporting/) | NIP-51 기반 뮤트/필터링, 신고 |
| 11 | [테스트 전략](/blog-repo/divine-mobile-guide-11-testing/) | 유닛/위젯/통합 테스트, 디버깅 |
| 12 | [빌드, 릴리스, 배포](/blog-repo/divine-mobile-guide-12-build-release-deploy/) | 빌드 스크립트, iOS/macOS CocoaPods, web 배포 |

---

## 주요 특징(레포 관점)

- **Flutter 멀티 플랫폼**: iOS/Android/Web/macOS 타깃을 동시에 다룸 (`mobile/`).
- **Nostr 기반 소셜 그래프**: 릴레이 구독/발행을 중심으로 프로필/반응/댓글을 구성.
- **Vine 스타일 비디오 이벤트**: kind `32222`(NIP-32222) 중심으로 피드를 구성.
- **서비스 레이어가 두꺼움**: `mobile/lib/services/`에 도메인 로직이 집중.
- **운영 자동화 스크립트**: `mobile/` 아래 다양한 빌드/배포 스크립트 제공.

---

## 빠른 시작

```bash
git clone https://github.com/divinevideo/divine-mobile
cd divine-mobile/mobile

flutter pub get
./run_dev.sh chrome debug
```

참고:
- 문서(`docs/README.md`, `docs/CF_STREAM_SETUP.md`)에는 `CF_STREAM_TOKEN`을 개발 시 주입하라는 안내가 있지만, 최신 `mobile/run_dev.sh`는 주로 `.env` 기반 `--dart-define`(Zendesk/Proofmode 등)을 구성합니다. 업로드 설정은 코드와 스크립트를 함께 확인하는 것이 안전합니다.

---

## 아키텍처 개요(요약)

```
┌─────────────────────────────────────────────────────────────┐
│                         diVine                              │
├─────────────────────────────────────────────────────────────┤
│  UI (screens/widgets)                                       │
│      ↓                                                      │
│  Provider (Riverpod)                                        │
│      ↓                                                      │
│  Services (upload/feed/social/auth/moderation/...)           │
│      ↓                                                      │
│  NostrClient (relays + signer + local cache/db)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| Flutter / Dart | 모바일/웹/데스크톱 앱 |
| Riverpod | 상태 관리, 의존성 주입 |
| Nostr (`nostr_client`, `nostr_sdk`) | 릴레이 통신, 이벤트 처리 |
| 각종 스크립트(`mobile/*.sh`) | 빌드/배포/운영 자동화 |

---

## 관련 링크

- [GitHub 저장소](https://github.com/divinevideo/divine-mobile)
- [diVine 웹](https://divine.video/discovery)

