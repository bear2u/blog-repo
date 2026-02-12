---
layout: post
title: "Tambo 완벽 가이드 (07) - 로컬 툴과 MCP 통합"
date: 2026-02-12
permalink: /tambo-guide-07-tools-mcp/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, 도구, MCP, 연동, 브라우저]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "브라우저 도구와 MCP 서버 연결"
---

## 로컬 도구

README가 강조하는 포인트는 "브라우저에서 실행되어야 하는 함수"가 있다는 점입니다.
DOM 조작, 인증이 필요한 fetch, React state 접근 등은 서버가 아니라 클라이언트에서 실행되는 편이 자연스럽습니다.

README 예시:

```tsx
const tools: TamboTool[] = [
  {
    name: "getWeather",
    description: "Fetches weather for a location",
    tool: async (params: { location: string }) =>
      fetch(`/api/weather?q=${encodeURIComponent(params.location)}`).then((r) =>
        r.json(),
      ),
    inputSchema: z.object({
      location: z.string(),
    }),
    outputSchema: z.object({
      temperature: z.number(),
      condition: z.string(),
      location: z.string(),
    }),
  },
];
```

---

## MCP 연동

Tambo는 MCP(Model Context Protocol)를 "기본 내장"으로 소개합니다.
README는 Linear/Slack/DB 등과의 연결을 예로 들며, tools/prompts/elicitations/sampling까지 지원한다고 명시합니다.

간단 예시(README):

```tsx
import { MCPTransport } from "@tambo-ai/react/mcp";

const mcpServers = [
  {
    name: "filesystem",
    url: "http://localhost:8261/mcp",
    transport: MCPTransport.HTTP,
  },
];

<TamboProvider mcpServers={mcpServers} />;
```

---

## 설계 팁

- 로컬 툴은 "권한 경계"가 핵심입니다. 브라우저에서 가능한 것만 열고, 민감 작업은 서버로 둡니다.
- MCP 서버는 "신뢰할 수 있는 서버"만 붙입니다. 특히 로컬 MCP 허용 옵션은 운영 환경에서 보수적으로 접근하는 게 안전합니다.

*다음 글에서는 Tambo 모노레포 구조를 큰 그림으로 정리합니다.*
