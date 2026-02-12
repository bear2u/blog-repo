---
layout: post
title: "Tambo 완벽 가이드 (05) - 상호작용 컴포넌트"
date: 2026-02-12
permalink: /tambo-guide-05-interactable-components/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, React, 상호작용, 상태, Zod]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "지속되고 업데이트되는 상태형 UI"
---

## 상호작용 컴포넌트란

상호작용 컴포넌트는 한 번 렌더링된 뒤에도 화면에 남아 있고,
사용자의 추가 요청(수정/추가/정리)에 따라 프롭스 또는 내부 상태가 업데이트되는 UI 패턴입니다.

쇼핑카트, 노트, 스프레드시트, 태스크 보드 같은 UI에 적합합니다.

---

## 예시: withInteractable(README)

README는 `withInteractable`로 컴포넌트를 감싸 "상호작용 가능"하게 만드는 패턴을 보여줍니다.

```tsx
const InteractableNote = withInteractable(Note, {
  componentName: "Note",
  description: "A note supporting title, content, and color modifications",
  propsSchema: z.object({
    title: z.string(),
    content: z.string(),
    color: z.enum(["white", "yellow", "blue", "green"]).optional(),
  }),
});
```

이렇게 만들면 사용자 요청에 따라 "이미 렌더된 Note"의 속성이 업데이트되는 경험을 설계할 수 있습니다.

---

## 설계 팁

- 상호작용 컴포넌트의 핵심은 "무엇을 업데이트 대상으로 볼 것인지"를 스키마로 명확히 하는 것입니다.
- 가능한 수정 범위를 제한하면(예: color enum) 모델이 UI를 더 안정적으로 제어합니다.

*다음 글에서는 TamboProvider와 훅, 인증/컨텍스트를 정리합니다.*
