---
layout: post
title: "diVine(OpenVine) 완벽 가이드 (09) - 피드 로딩과 페이지네이션"
date: 2026-02-13
permalink: /divine-mobile-guide-09-feed-pagination/
author: divinevideo
categories: [모바일, Nostr]
tags: [Feed, Pagination, Nostr, Filter, Relay, diVine]
original_url: "https://github.com/divinevideo/divine-mobile"
excerpt: "kind 32222 피드를 'until' 기반으로 어떻게 계속 과거로 확장하는지, 중복/리셋 버그를 어떻게 막는지 정리합니다."
---

## 문제: 스크롤을 내리는데 같은 영상이 반복해서 뜬다

Nostr 피드 페이지네이션의 고전적인 문제는 단순합니다.

- “더 보기”를 요청했는데
- **항상 최신 이벤트만 다시 받아와서**
- 기존과 같은 콘텐츠를 반복 표시한다

이 레포는 이 문제를 문서로 정리해 두었습니다.

- `docs/nostr_pagination_docs.md`

---

## 핵심 아이디어: `until` 파라미터를 올바르게 유지하라

Nostr relay 쿼리는 보통 filter에 `until`을 넣어 “특정 시점 이전의 이벤트”를 요청합니다.

```json
{
  "kinds": [32222],
  "until": 1754262575,
  "limit": 50
}
```

요약하면:

- `until`이 없으면: relay는 최신 이벤트부터 준다
- `until`이 있으면: `created_at <= until` 범위의 이벤트를 준다

따라서 페이지네이션은 “현재 피드에서 가장 오래된(created_at이 작은) 이벤트의 timestamp”를 추적해서,
다음 요청의 `until`로 사용해야 합니다.

---

## VideoEventService의 PaginationState

문서가 가리키는 구현 지점은:

- `mobile/lib/services/video_event_service.dart`

여기에는 `PaginationState`가 있고, 대략 아래 상태를 관리합니다.

- `oldestTimestamp`: 현재 피드에서 가장 오래된 이벤트 시각
- `hasMore`: 더 가져올 게 있는지
- `seenEventIds`: 중복 방지를 위한 집합

중복 방지는 특히 경계 조건에서 중요합니다.
relay가 `until`과 “같은 시각” 이벤트를 포함해서 주는 경우가 있기 때문에, 이벤트 id 기반으로 한 번 더 걸러줘야 합니다.

---

## 리셋 버그의 원인과 해결(문서 요약)

문서가 설명하는 root cause는 다음과 같습니다.

- `hasMore=false`가 된 뒤 페이지네이션 상태를 리셋하면 `oldestTimestamp`가 null로 초기화된다
- 그런데 피드 리스트 자체는 메모리에 남아있다
- 다음 요청에서 `until`이 null이 되면서 relay가 최신 이벤트를 다시 준다

해결 전략:

- “상태가 리셋되었는데 피드 이벤트가 남아 있으면”
- 피드 리스트에서 다시 **oldestTimestamp를 재계산**해서 `until`을 세팅한다

이런 방식은 “상태 머신”과 “실제 데이터(피드 리스트)”가 어긋났을 때 자동으로 복구하는 패턴입니다.

---

## Embedded relay(로컬 프록시) 관점

문서는 “앱이 로컬 embedded relay에 붙고, 그 relay가 외부 relay와 통신한다”는 아키텍처를 언급합니다.
이 구조가 있으면:

- 캐시/오프라인 지원을 강화할 수 있고
- 클라이언트 단에서 쿼리/필터링 전략을 단순화할 수도 있습니다

하지만 디버깅할 때는 오히려 “어디에서 끊겼는지” 지점이 늘어나기 때문에,
로그(요청/응답/중복 필터링)가 중요해집니다.

---

## 디버깅 체크리스트

- `until`이 실제로 세팅되어 나가는가?
- `oldestTimestamp`가 리셋 이후에도 재계산되는가?
- `seenEventIds`가 제대로 작동해 경계 중복을 제거하는가?
- “더 보기”가 실행될 때 `isLoading` 플래그가 중복 호출을 막는가?

---

*다음 글에서는 모더레이션/신고가 어떤 레이어(백엔드/분산)로 구성되어 있고, NIP-51/32/56을 어떻게 조합하는지 정리합니다.*

