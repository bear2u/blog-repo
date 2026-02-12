---
layout: post
title: "Tambo 완벽 가이드 (04) - 생성형 컴포넌트"
date: 2026-02-12
permalink: /tambo-guide-04-generative-components/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, React, Zod, 생성형, 컴포넌트, 차트]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "1회성 렌더링 UI 컴포넌트 패턴"
---

## 생성형 컴포넌트란

생성형 컴포넌트는 "메시지 1개에 반응해 1회 렌더링"되는 UI입니다.
예를 들어 차트/요약/데이터 시각화처럼, 결과가 확정되면 더 이상 유지할 필요가 없는 UI에 적합합니다.

---

## 예시: Graph 컴포넌트 등록(README)

README 예시를 기준으로 보면, 등록 데이터는 대략 아래 형태입니다.

```tsx
const components: TamboComponent[] = [
  {
    name: "Graph",
    description: "Displays data as charts using Recharts library",
    component: Graph,
    propsSchema: z.object({
      data: z.array(z.object({ name: z.string(), value: z.number() })),
      type: z.enum(["line", "bar", "pie"]),
    }),
  },
];
```

핵심 포인트:
- `name`/`description`은 LLM이 선택할 때 힌트가 됩니다.
- `propsSchema`가 도구의 입력 스키마가 됩니다.

---

## 설계 팁

- props는 "화면을 그리는 데 필요한 최소"로 유지합니다.
- 스키마는 엄격하게(유효성 검증은 곧 UX 품질) 가져가고, 애매한 값은 enum으로 제한합니다.
- 컴포넌트 설명은 "언제 쓰는지"가 드러나게 씁니다.

*다음 글에서는 상태를 유지하고 업데이트하는 상호작용 컴포넌트를 다룹니다.*
