---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (04) - Nostr 클라이언트 아키텍처"
date: 2026-02-13
permalink: /divine-mobile-guide-04-nostr-architecture/
author: divinevideo
categories: [모바일, Flutter]
tags: [diVine, OpenVine, Nostr, Relay, Riverpod, NIP-65]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "NostrClient가 어떻게 생성/초기화되고, 릴레이 목록과 인증 상태 변화에 어떻게 반응하는지 정리합니다."
---

## 큰 그림: 앱의 “네트워크 코어”

diVine(OpenVine)의 핵심은 Nostr 이벤트를 **구독하고**, **발행**하는 것입니다.
코드 관점에서 “Nostr 코어”는 대략 이런 흐름을 갖습니다.

```
AuthService(키/세션) → NostrServiceFactory → NostrClient
                                    ↓
                           addRelays() + initialize()
```

---

## NostrService Provider: Riverpod에서 NostrClient 만들기

가장 중요한 파일은:

- `mobile/lib/providers/nostr_client_provider.dart`

여기에는 `@Riverpod(keepAlive: true)`로 선언된 `NostrService`가 있고, `build()`에서 `NostrClient`를 구성합니다.
핵심 포인트는 3가지입니다.

### 1) Auth 상태에 반응해 “클라이언트 재생성”

`NostrService`는 `AuthService`의 `authStateStream`을 구독하고,
현재 pubkey가 바뀌면 내부 `state`를 dispose 한 뒤 **새 `NostrClient`를 다시 만들어** 교체합니다.

이 패턴의 장점:

- “로그아웃/로그인” 같은 키 전환을 안정적으로 처리
- 릴레이/구독/서명자(signer) 상태가 꼬이는 문제를 줄임

### 2) 사용자 릴레이 목록(NIP-65)을 먼저 추가한 뒤 initialize()

`build()` 내부 주석에 “race condition 방지”가 명시되어 있습니다.

- 사용자 릴레이 URL을 모은 뒤(`authService.userRelays`)
- `client.addRelays(userRelayUrls)`를 먼저 수행하고
- 그 다음 `client.initialize()`를 호출합니다.

즉, “초기화 전에 릴레이를 붙인다”가 이 레포의 중요한 초기화 규칙입니다.

### 3) keepAlive + onDispose로 수명 관리

`@Riverpod(keepAlive: true)`로 NostrClient를 앱 전체에서 재사용하게 만들고,
provider disposal 시점에는:

- auth subscription 취소
- `client.dispose()`

를 호출해 자원을 정리합니다.

---

## NostrServiceFactory: 생성 책임을 분리

`mobile/lib/services/nostr_service_factory.dart`는 “클라이언트를 어떻게 만들지”를 factory로 분리해 둔 파일입니다.
provider가 너무 비대해지는 것을 막고, 생성 전략(예: RPC signer 지원 여부)을 한 곳에서 통제할 수 있게 됩니다.

이 구조를 따라가면 “서명자/키 컨테이너가 어디서 오는지”도 자연스럽게 연결됩니다.

---

## 핵심 의존성(코드에서 보이는 것)

`mobile/lib/providers/nostr_client_provider.dart`를 기준으로 보면 NostrClient는 아래 구성 요소들과 강하게 결합됩니다.

- `AuthService`: 현재 사용자 키/릴레이 목록, (가능하면) RPC signer
- `relayStatisticsServiceProvider`: 릴레이 성능/상태 통계를 추적하는 서비스
- `currentEnvironmentProvider`: 환경 설정(릴레이/서버 엔드포인트 등이 바뀔 수 있음)
- `appDbClientProvider`: 로컬 DB/캐시와의 연결(오프라인/성능에 영향)

즉, NostrClient는 단순히 “웹소켓 연결”이 아니라,
**인증 상태 + 환경 + 통계 + 캐시**까지 묶인 “앱 네트워크 서브시스템”에 가깝습니다.

---

## 디버깅 포인트(실무적으로 유용한 것)

Nostr 관련 문제는 원인이 다양합니다. 이 레포 구조에서 빠르게 점검할 포인트를 체크리스트로 두면 좋습니다.

### 릴레이가 안 붙는다

- `authService.userRelays`가 비어있는지
- `addRelays()`가 `initialize()`보다 먼저 실행되는지

### 키를 바꾸면 동작이 이상해진다

- pubkey 변경 시 `_onAuthStateChanged()`가 호출되는지
- 이전 `state.dispose()`가 수행되는지(기존 연결/구독 정리 여부)

### “특정 기능”만 안 된다(댓글/좋아요/팔로우 등)

Nostr는 기능마다 이벤트 kind가 다르고, 필터/구독도 다릅니다.
이럴 때는 코드보다 먼저 `docs/NOSTR_EVENT_TYPES.md`를 보고 “내가 지금 찾는 기능이 어떤 kind인지”부터 확인하는 게 빠릅니다.

---

*다음 글에서는 앱이 실제로 어떤 kind를 쓰는지(특히 kind 32222), 그리고 NIP-32222가 왜 중요한지 정리합니다.*
