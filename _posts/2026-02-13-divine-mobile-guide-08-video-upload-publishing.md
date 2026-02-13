---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (08) - 업로드와 퍼블리시(비디오 이벤트 발행)"
date: 2026-02-13
permalink: /divine-mobile-guide-08-video-upload-publishing/
author: divinevideo
categories: [모바일, Nostr]
tags: [Upload, Blossom, Nostr, NIP-32222, Flutter, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "UploadManager(업로드 큐/재시도)와 VideoEventPublisher(이벤트 생성/발행)의 경계를 기준으로 업로드 파이프라인을 정리합니다."
---

## 업로드 파이프라인을 “두 덩어리”로 나누기

이 레포의 업로드 관련 코드는 매우 크지만, 개념적으로는 두 덩어리로 나뉩니다.

1. **업로드 상태/큐 관리**: `UploadManager`
2. **업로드 결과를 Nostr 이벤트로 퍼블리시**: `VideoEventPublisher`

간단한 흐름도로 보면:

```text
로컬 파일/클립
   ↓
UploadManager (queue, retry, persistence)
   ↓
(CDN URL / thumbnail / metadata)
   ↓
VideoEventPublisher (kind 32222 event build + sign + publish)
   ↓
Nostr relays
```

---

## UploadManager: 업로드 상태 머신 + 영속화

핵심 파일:

- `mobile/lib/services/upload_manager.dart`

파일 상단 ABOUTME 주석이 말하듯, `UploadManager`의 목적은 “업로드 상태와 로컬 영속화”입니다.

### 주요 책임

- 업로드 큐 관리(대기/진행/처리중/게시 가능 등 상태)
- 네트워크/서버 오류 시 재시도(backoff)
- 앱 종료/재시작 이후에도 업로드 상태를 유지(Hive box)
- 중단된 업로드 재개

### 코드에서 눈에 띄는 구현 디테일

- `UploadRetryConfig`: `maxRetries`, backoff 설정
- `UploadMetrics`: 업로드 성능/에러 카테고리 기록
- `VideoCircuitBreaker`: 실패율 기반으로 업로드를 보호(연쇄 실패 방지)
- `initialize()`에서 “robust initialization helper”를 사용해 복구 전략까지 포함

또 하나 중요한 힌트:

- `setUploadTarget(...)`는 deprecated 처리되어 있고,
- provider 주석도 “Upload manager uses only Blossom upload service”로 되어 있습니다.

즉, “여러 업로드 타깃 중 선택”보다는 “현재는 Blossom 업로드 중심으로 단일화”된 상태로 보는 게 맞습니다.

---

## BlossomUploadService: 실제 업로드 구현

provider 정의는 다음을 명시합니다.

- `BlossomUploadService blossomUploadService(Ref ref)`
  - `authService`에 의존
  - “user-configured Blossom server”를 사용

관련 파일:

- `mobile/lib/services/blossom_upload_service.dart`

업로드가 실패할 때는 UploadManager만 보지 말고,
Blossom 서비스가 “어떤 서버/인증으로” 요청을 보내는지도 같이 확인해야 합니다.

관련 provider들:

- `blossomAuthServiceProvider`: age-restricted 콘텐츠용 인증(BUD-01) 언급
- `mediaAuthInterceptorProvider`: 401 응답 처리용 인터셉터

즉, 업로드가 “네트워크 문제”처럼 보여도 실제로는 “인증/연령 확인/서버 설정” 문제일 수 있습니다.

---

## VideoEventPublisher: 업로드 결과를 kind 32222 이벤트로 발행

핵심 파일:

- `mobile/lib/services/video_event_publisher.dart`

파일 상단 ABOUTME가 말하듯, 이 서비스는 “백엔드 후처리 없이 Nostr로 직접 발행”하는 역할을 목표로 합니다.

### 코드에서 보이는 특징

- 내부에 `_publishEventToNostr(Event event)`가 있고,
  - 릴레이 연결 상태를 진단 로그로 출력
  - `_nostrService.isInitialized`가 false면 `initialize()`를 시도
  - 마지막에 `_nostrService.publishEvent(event)`를 호출

즉, 퍼블리시에서 가장 먼저 터지는 실패는 보통:

- NostrClient가 아직 초기화되지 않았거나
- 연결된 릴레이 수가 0이거나
- 릴레이가 write를 막고 있거나

같은 상태 문제입니다.

### provider 관점에서의 의존성 폭

`mobile/lib/providers/app_providers.dart`에서 `VideoEventPublisher`는 다음에 의존합니다.

- `UploadManager`: 업로드 결과/상태
- `NostrClient`: 릴레이 발행
- `AuthService`: 현재 키/세션
- `PersonalEventCacheService`: 개인 이벤트 캐시
- `VideoEventService`: 비디오 이벤트 도메인 처리
- `BlossomUploadService`: 업로드 연동(필요 시)
- `UserProfileService`: 프로필 관련(표시/캐시)

즉, “업로드 성공”만으로 끝나지 않고, “이벤트 구성 및 로컬 캐시 반영”까지 한 덩어리로 봐야 합니다.

---

## 문서 vs 코드: CF Stream 토큰 안내는 어떻게 봐야 하나

문서(`docs/README.md`, `docs/CF_STREAM_SETUP.md`)는 Cloudflare Stream과 `CF_STREAM_TOKEN` 주입을 강조합니다.
하지만 현재 코드 전체에서 `CF_STREAM_TOKEN`은 문서에만 등장하고,
실제 provider/서비스 구성은 Blossom 업로드 중심입니다.

이럴 때의 안전한 접근:

1. 문서를 “의도/방향성”으로 읽고
2. 실제 구현은 provider(조립 지점)와 서비스 파일(로직 지점)에서 확인

즉, 업로드 경로를 바꾸거나 디버깅할 때는 “코드가 진실”입니다.

---

## 빠른 디버깅 체크리스트

- `UploadManager.initialize()`가 호출되어 Hive box가 열렸는가?
- `BlossomUploadService.isBlossomEnabled()`가 true인가?
- 업로드 완료 후 `PendingUpload`에 `cdnUrl`/`thumbnailPath`가 채워지는가?
- `NostrClient`가 초기화됐는가?(`configuredRelayCount`, `connectedRelayCount`)
- kind `32222` 이벤트의 `d/title/imeta` 태그가 정상 구성되는가?

---

*다음 글에서는 퍼블리시된 kind 32222 이벤트를 어떻게 피드로 모으고, 페이지네이션을 어떻게 설계하는지 정리합니다.*
