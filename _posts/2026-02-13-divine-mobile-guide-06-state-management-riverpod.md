---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (06) - 상태 관리와 의존성 주입(Riverpod)"
date: 2026-02-13
permalink: /divine-mobile-guide-06-state-management-riverpod/
author: divinevideo
categories: [모바일, Flutter]
tags: [Riverpod, Provider, Dependency Injection, diVine, OpenVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "`providers/`에서 서비스가 어떻게 조립되는지 보고, 복잡한 기능을 '의존성 그래프'로 정리합니다."
---

## 왜 Riverpod을 먼저 보면 좋은가

대형 Flutter 앱에서 “코드를 어디서부터 읽을지” 결정하기 어려운 이유는,
기능이 `screens/`에만 있는 게 아니라 **서비스/리포지토리/클라이언트가 얽혀** 있기 때문입니다.

이 레포는 Riverpod을 사용하고, 특히 `mobile/lib/providers/app_providers.dart`에 많은 서비스가 “생성자 주입 형태”로 등록되어 있습니다.
즉, provider 파일을 보면 “이 기능이 어떤 서비스들을 필요로 하는지”를 바로 알 수 있습니다.

---

## 핵심 파일

- `mobile/lib/providers/app_providers.dart`: 대부분의 서비스 provider가 여기에 모여 있음
- `mobile/lib/providers/nostr_client_provider.dart`: NostrClient 생성/수명 관리

---

## 예시 1: 업로드 파이프라인이 provider에서 보인다

`mobile/lib/providers/app_providers.dart`에는 업로드 관련 provider들이 연달아 정의되어 있습니다.
핵심은 아래 3단계입니다.

1. `BlossomUploadService` 생성
2. `UploadManager` 생성(Blossom 서비스에 의존)
3. `VideoEventPublisher` 생성(업로드 + nostr + auth + 캐시 + 프로필 등 다수 의존)

의존성 그래프를 “그림”으로 그리면 대략 이렇습니다.

```text
BlossomUploadService
        ↓
   UploadManager (keepAlive)
        ↓
VideoEventPublisher (keepAlive)
  ├─ nostrServiceProvider
  ├─ authServiceProvider
  ├─ personalEventCacheServiceProvider
  ├─ videoEventServiceProvider
  └─ userProfileServiceProvider
```

이걸 보면 “업로드는 단독 기능이 아니라, 퍼블리시/캐시/프로필까지 한 덩어리”라는 것을 바로 알 수 있습니다.

---

## 예시 2: 인증/서명/요청 인터셉터

문서에 NIP-98(HTTP 인증)이나 Blossom 인증이 언급되는 이유도 provider 구성을 보면 이해가 쉽습니다.

- `Nip98AuthService`: authService를 기반으로 생성
- `BlossomAuthService`: authService를 기반으로 생성
- `MediaAuthInterceptor`: age verification + blossom auth를 묶어서 401 응답을 처리

즉, “요청” 계층에서도 provider로 주입해 재사용하는 구조입니다.

---

## keepAlive를 어디에 쓰는가

`@Riverpod(keepAlive: true)`가 붙은 provider들은 “앱 전체에서 재사용해도 되는, 상태/리소스가 큰 객체”인 경우가 많습니다.
이 레포에서 대표적인 예:

- `uploadManager`
- `videoEventPublisher`
- `curationService`

이런 객체는 내부에 큐/캐시/구독/타이머 같은 것을 들고 있을 수 있고,
화면 전환마다 매번 새로 만들면 비효율 또는 버그(중복 구독 등)가 생길 수 있습니다.

---

## 읽는 요령: provider에서 서비스 파일로 내려가기

이 레포를 빠르게 파악하려면 다음 루프를 추천합니다.

1. `app_providers.dart`에서 “내가 관심 있는 기능” provider를 찾는다
2. 생성자 파라미터(의존성)를 적는다
3. 해당 서비스 파일로 내려가서 실제 로직을 읽는다

예를 들어 “업로드”가 궁금하면:

- `UploadManager` → `mobile/lib/services/upload_manager.dart`
- `VideoEventPublisher` → `mobile/lib/services/video_event_publisher.dart`
- `BlossomUploadService` → `mobile/lib/services/blossom_upload_service.dart`

---

*다음 글에서는 UI가 아닌 ‘카메라/녹화’ 기능이 어떤 식으로 서비스와 엮이는지, 비디오 녹화 UX 흐름을 정리합니다.*

