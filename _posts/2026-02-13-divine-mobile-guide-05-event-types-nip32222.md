---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (05) - 이벤트 타입과 NIP-32222"
date: 2026-02-13
permalink: /divine-mobile-guide-05-event-types-nip32222/
author: divinevideo
categories: [모바일, Nostr]
tags: [Nostr, NIP-01, NIP-18, NIP-25, NIP-92, NIP-32222, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "diVine(OpenVine)이 사용하는 Nostr 이벤트(kind)와, 핵심 비디오 이벤트(kind 32222)를 정리합니다."
---

## 왜 kind 목록을 먼저 봐야 하나

Nostr 앱을 이해할 때 가장 빠른 지름길은 “이 앱이 어떤 kind를 쓰는지”를 먼저 잡는 것입니다.
diVine(OpenVine)은 문서로 이벤트 요구사항을 잘 정리해 두었습니다.

- `docs/NOSTR_EVENT_TYPES.md`
- `docs/nip-vine.md` (NIP-32222 초안)

---

## 핵심 kind 요약

`docs/NOSTR_EVENT_TYPES.md` 기준으로, 앱에서 중요도가 높은 kind는 아래입니다.

| kind | 용도 | 관련 NIP |
|------|------|----------|
| 0 | 프로필(이름/아바타/바이오) | NIP-01 |
| 1 | 텍스트 노트(댓글 등) | NIP-01 |
| 3 | 팔로우/연락처 리스트 | NIP-02 |
| 5 | 삭제 이벤트(언라이크 등) | NIP-09 |
| 7 | 반응(좋아요) | NIP-25 |
| 16 | 리포스트 | NIP-18 |
| 32222 | “주소가능” 숏폼 루프 비디오 | NIP-32222(draft) |

이 중에서 피드의 1차 콘텐츠는 kind `32222`입니다.

---

## kind 32222: “주소가능(Addressable)” 비디오 이벤트

`docs/nip-vine.md`는 kind `32222`의 핵심 동기를 이렇게 요약합니다.

- 짧은 반복 비디오는 “메타데이터 수정”이 자주 필요하다
- 이벤트를 다시 발행하기보다, **같은 식별자를 유지하면서 업데이트**할 수 있으면 좋다

kind `32222`는 주소가능 이벤트 범위(30000-39999)를 쓰기 때문에, 식별을 위해 반드시 `d` 태그가 필요합니다.

### 최소 예시(형태만)

```json
{
  "kind": 32222,
  "content": "short description",
  "tags": [
    ["d", "unique-id"],
    ["title", "video title"],
    ["imeta",
      "url https://video.host/video.mp4",
      "m video/mp4",
      "dim 480x480",
      "image https://video.host/thumb.jpg"
    ]
  ]
}
```

여기서 중요한 것은 “콘텐츠 문자열”보다도 `tags`입니다.

---

## imeta 태그: 비디오/썸네일 메타데이터의 컨테이너

문서에서 `imeta`는 NIP-92 스타일로 설명됩니다.
요약하면:

- 비디오 URL(`url`)
- MIME 타입(`m`)
- 해상도(`dim`)
- 썸네일 URL(`image`)

같은 정보가 `imeta` 안에 들어갑니다.
클라이언트는 이를 기반으로 플레이어 로딩/프리뷰/레이아웃을 결정할 수 있습니다.

---

## “주소”로 참조하기: a 태그

주소가능 이벤트는 id 대신 “주소 문자열”로 참조되는 경우가 많습니다.
문서에서는 kind 32222를 참조할 때 다음 형태를 예시로 둡니다.

```text
32222:<pubkey>:<d-tag-value>
```

리포스트(kind 16) 같은 “2차 이벤트”가 원본 비디오를 가리킬 때도, 이런 주소 체계를 활용할 수 있습니다.

---

## 코드 관점: 어디서 kind를 다루나

“이벤트 종류는 문서에서 이해”하고, “실제 처리/필터링은 코드에서 확인”하는 식으로 접근하면 좋습니다.

시작 지점:

- `mobile/lib/services/social_service.dart`: 좋아요/리포스트/댓글 등 소셜 동작의 이벤트 발행
- `mobile/lib/services/curation_service.dart`: 리스트/큐레이션 관련(문서상 NIP-51도 언급)
- `mobile/lib/providers/video_events_providers.dart`: 피드 쿼리/구독 단

그리고 항상 기억해야 할 것:

- “피드가 비었다”는 문제는 종종 `Filter(kinds: [...])` 구성 오류에서 시작합니다.
- kind `32222`와 kind `16`(리포스트)을 동시에 다뤄야 “실제 피드”가 완성됩니다(문서에도 강조).

---

*다음 글에서는 kind를 주고받는 NostrClient가 provider에서 어떻게 조립되는지에 이어, 앱 전반의 의존성 주입(Riverpod) 패턴을 정리합니다.*

