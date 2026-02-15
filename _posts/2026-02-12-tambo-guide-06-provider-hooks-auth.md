---
layout: post
title: "Tambo 완벽 가이드 (06) - 프로바이더와 훅, 인증/컨텍스트"
date: 2026-02-12
permalink: /tambo-guide-06-provider-hooks-auth/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, 프로바이더, 훅, 인증, 컨텍스트]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "TamboProvider와 핵심 훅 정리"
---

## TamboProvider

README는 앱 루트에 `TamboProvider`를 감싸는 패턴을 기본으로 제시합니다.

```tsx
<TamboProvider
  apiKey={process.env.NEXT_PUBLIC_TAMBO_API_KEY!}
  userKey={currentUserId}
  components={components}
>
  <Chat />
</TamboProvider>
```

핵심 입력:
- `components`: 등록한 UI 컴포넌트들
- `apiKey`: 백엔드 호출용 키
- `userKey` 또는 `userToken`: 스레드 소유자 식별

---

## userKey vs userToken

README 설명을 정리하면:
- `userKey`: 서버 사이드/신뢰 환경(내부 시스템)에서 사용
- `userToken`: 클라이언트 사이드에서 토큰 자체에 사용자 식별이 포함되는 경우(OAuth access token 등)

---

## 훅

README에 등장하는 주요 훅:

```tsx
const { messages, isStreaming } = useTambo();
const { value, setValue, submit, isPending } = useTamboThreadInput();
```

- `useTambo()`: 메시지, 스트리밍 상태, 스레드 관리
- `useTamboThreadInput()`: 입력 상태와 전송

---

## 추가 컨텍스트와 추천 프롬프트

추가 컨텍스트(예: 선택 항목, 현재 페이지)와 추천 프롬프트 UI도 제공됩니다.

{% raw %}
```tsx
<TamboProvider
  userToken={userToken}
  contextHelpers={{
    selectedItems: () => ({ key: "selectedItems", value: "..." }),
    currentPage: () => ({ key: "page", value: window.location.pathname }),
  }}
/>
```
{% endraw %}

```tsx
const { suggestions, accept } = useTamboSuggestions({ maxSuggestions: 3 });
```

*다음 글에서는 로컬 툴과 MCP 통합을 다룹니다.*
