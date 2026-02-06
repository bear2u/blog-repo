---
layout: post
title: "Acontext 완벽 가이드 (2) - 빠른 시작"
date: 2026-02-06
permalink: /acontext-guide-02-quickstart/
author: memodb-io
categories: [AI 에이전트, Acontext]
tags: [Acontext, Quickstart, Hosted, Self-hosted, SDK]
original_url: "https://docs.acontext.io/quick"
excerpt: "Hosted와 Self-hosted 두 경로로 Acontext를 빠르게 시작합니다."
---

## 시작 경로 2가지

공식 Quickstart 기준으로 시작 방법은 두 가지입니다.

1. **Hosted(권장)**: 대시보드 가입 후 API Key 발급
2. **Self-hosted**: `acontext-cli`로 로컬 서버 구동

## Hosted

```bash
export ACONTEXT_API_KEY="your-api-key"
```

이후 Python/TypeScript SDK에서 같은 키를 사용합니다.

## Self-hosted

```bash
curl -fsSL https://install.acontext.io | sh
mkdir acontext_server && cd acontext_server
acontext server up
```

기본 엔드포인트:

- API: `http://localhost:8029/api/v1`
- Dashboard: `http://localhost:3000/`

## SDK 초기화

```python
import os
from acontext import AcontextClient

client = AcontextClient(api_key=os.getenv("ACONTEXT_API_KEY"))
```

## 첫 세션/메시지 저장

```python
session = client.sessions.create()
client.sessions.store_message(
    session_id=session.id,
    blob={"role": "user", "content": "Hello"},
    format="openai"
)
```

이 시점부터 Acontext는 단순 저장소가 아니라, 추후 Task 추출/요약/편집 전략 적용의 기반 데이터를 갖게 됩니다.
