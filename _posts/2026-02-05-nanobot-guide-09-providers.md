---
layout: post
title: "Nanobot 완벽 가이드 (9) - Providers 시스템"
date: 2026-02-05
permalink: /nanobot-guide-09-providers/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Providers, LLM, OpenRouter, vLLM, API]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 LLM 프로바이더 시스템과 다양한 모델 연동 방법을 알아봅니다."
---

## Providers 시스템 개요

Nanobot은 다양한 LLM 프로바이더를 지원하여 유연한 모델 선택이 가능합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Providers 시스템                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  LLMProvider                         │   │
│  │                  (Base Class)                        │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│     ┌────────────────────┼────────────────────┐            │
│     │                    │                    │            │
│     ▼                    ▼                    ▼            │
│  ┌─────────┐      ┌─────────────┐      ┌─────────┐        │
│  │OpenRouter│      │  Anthropic  │      │  OpenAI │        │
│  │Provider │      │  Provider   │      │ Provider│        │
│  └─────────┘      └─────────────┘      └─────────┘        │
│                                                              │
│  ┌─────────┐      ┌─────────────┐      ┌─────────┐        │
│  │DeepSeek │      │    Groq     │      │  Gemini │        │
│  │Provider │      │  Provider   │      │ Provider│        │
│  └─────────┘      └─────────────┘      └─────────┘        │
│                                                              │
│  ┌─────────┐                                                │
│  │  vLLM   │      (로컬 모델)                               │
│  │Provider │                                                │
│  └─────────┘                                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 지원 프로바이더

| 프로바이더 | 모델 예시 | 특징 |
|----------|----------|------|
| **OpenRouter** | `anthropic/claude-opus-4-5` | 모든 모델 통합 접근 |
| **Anthropic** | `claude-opus-4-5-20251101` | Claude 직접 연동 |
| **OpenAI** | `gpt-4-turbo` | GPT 직접 연동 |
| **DeepSeek** | `deepseek-chat` | DeepSeek 모델 |
| **Groq** | `llama-3.1-70b-versatile` | 초고속 추론 |
| **Gemini** | `gemini-pro` | Google 모델 |
| **vLLM** | 로컬 모델 | 자체 서버 |

---

## LLMProvider 베이스 클래스

```python
# providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    """LLM 응답"""
    content: str
    tool_calls: list[ToolCall] | None = None
    usage: dict | None = None

@dataclass
class ToolCall:
    """도구 호출"""
    id: str
    name: str
    arguments: dict

class LLMProvider(ABC):
    """LLM 프로바이더 베이스 클래스"""

    @abstractmethod
    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> LLMResponse:
        """LLM 호출"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """기본 모델 반환"""
        pass

    @abstractmethod
    def list_models(self) -> list[str]:
        """사용 가능한 모델 목록"""
        pass
```

---

## OpenRouter Provider

모든 주요 LLM에 단일 API로 접근할 수 있는 권장 프로바이더입니다.

### 설정

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5"
    }
  }
}
```

### 구현

```python
# providers/openrouter.py

class OpenRouterProvider(LLMProvider):
    """OpenRouter 프로바이더"""

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> LLMResponse:
        """LLM 호출"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "messages": messages,
            }

            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            response = await client.post(
                f"{self.API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://nanobot.ai",
                    "X-Title": "Nanobot",
                },
                json=payload,
                timeout=120.0,
            )

            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]

            # 도구 호출 파싱
            tool_calls = None
            if message.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=json.loads(tc["function"]["arguments"]),
                    )
                    for tc in message["tool_calls"]
                ]

            return LLMResponse(
                content=message.get("content") or "",
                tool_calls=tool_calls,
                usage=data.get("usage"),
            )

    def get_default_model(self) -> str:
        return "anthropic/claude-opus-4-5"

    def list_models(self) -> list[str]:
        return [
            "anthropic/claude-opus-4-5",
            "anthropic/claude-sonnet-4",
            "openai/gpt-4-turbo",
            "openai/gpt-4o",
            "meta-llama/llama-3.1-70b-instruct",
            "google/gemini-pro-1.5",
        ]
```

### 지원 모델

```python
# OpenRouter를 통해 접근 가능한 모델들
OPENROUTER_MODELS = {
    # Anthropic
    "anthropic/claude-opus-4-5": "가장 강력한 Claude",
    "anthropic/claude-sonnet-4": "균형 잡힌 성능",

    # OpenAI
    "openai/gpt-4-turbo": "GPT-4 Turbo",
    "openai/gpt-4o": "최신 GPT-4",
    "openai/o1-preview": "추론 특화",

    # Meta
    "meta-llama/llama-3.1-405b-instruct": "가장 큰 오픈소스",
    "meta-llama/llama-3.1-70b-instruct": "균형 잡힌 오픈소스",

    # Google
    "google/gemini-pro-1.5": "Gemini Pro",

    # DeepSeek
    "deepseek/deepseek-chat": "효율적인 중국 모델",
}
```

---

## Anthropic Provider

Claude 모델 직접 연동입니다.

### 설정

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "sk-ant-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "claude-opus-4-5-20251101"
    }
  }
}
```

### 구현

```python
# providers/anthropic.py

import anthropic

class AnthropicProvider(LLMProvider):
    """Anthropic 프로바이더"""

    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> LLMResponse:
        """Claude 호출"""
        # 시스템 메시지 분리
        system_prompt = ""
        chat_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)

        # Anthropic 형식으로 도구 변환
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"],
                }
                for tool in tools
            ]

        response = await self.client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=chat_messages,
            tools=anthropic_tools,
        )

        # 응답 파싱
        content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def get_default_model(self) -> str:
        return "claude-opus-4-5-20251101"

    def list_models(self) -> list[str]:
        return [
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-haiku-20240307",
        ]
```

---

## Groq Provider

초고속 추론을 제공하는 Groq 연동입니다.

### 특징

- **매우 빠른 응답 속도**
- **Whisper 음성 전사 지원**
- 무료 티어 제공

### 설정

```json
{
  "providers": {
    "groq": {
      "apiKey": "gsk_xxx"
    }
  }
}
```

### 구현

```python
# providers/groq.py

class GroqProvider(LLMProvider):
    """Groq 프로바이더"""

    API_BASE = "https://api.groq.com/openai/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> LLMResponse:
        """Groq LLM 호출"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "messages": messages,
            }

            if tools:
                payload["tools"] = tools

            response = await client.post(
                f"{self.API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )

            # OpenAI 호환 응답 파싱
            data = response.json()
            # ... 파싱 로직

    async def transcribe(self, audio_path: str) -> str:
        """Whisper 음성 전사"""
        async with httpx.AsyncClient() as client:
            with open(audio_path, "rb") as f:
                response = await client.post(
                    f"{self.API_BASE}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": f},
                    data={"model": "whisper-large-v3"},
                )

            return response.json()["text"]

    def get_default_model(self) -> str:
        return "llama-3.1-70b-versatile"

    def list_models(self) -> list[str]:
        return [
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]
```

---

## vLLM Provider (로컬 모델)

자체 GPU 서버에서 로컬 모델을 실행할 수 있습니다.

### vLLM 서버 시작

```bash
# vLLM 설치
pip install vllm

# 서버 시작
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
```

### 설정

```json
{
  "providers": {
    "vllm": {
      "apiKey": "dummy",
      "apiBase": "http://localhost:8000/v1"
    }
  },
  "agents": {
    "defaults": {
      "model": "meta-llama/Llama-3.1-8B-Instruct"
    }
  }
}
```

### 구현

```python
# providers/vllm.py

class VLLMProvider(LLMProvider):
    """vLLM/OpenAI 호환 프로바이더"""

    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base

    async def complete(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs
    ) -> LLMResponse:
        """로컬 LLM 호출"""
        async with httpx.AsyncClient() as client:
            payload = {
                "model": model,
                "messages": messages,
            }

            # 도구 지원 여부 확인
            if tools:
                payload["tools"] = tools

            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=300.0,  # 로컬 모델은 느릴 수 있음
            )

            data = response.json()
            # ... OpenAI 호환 응답 파싱

    def get_default_model(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"
```

### 추천 로컬 모델

| 모델 | VRAM | 용도 |
|-----|------|------|
| `Llama-3.1-8B-Instruct` | 16GB | 일반 대화 |
| `Llama-3.1-70B-Instruct` | 80GB+ | 고품질 추론 |
| `Mistral-7B-Instruct` | 14GB | 빠른 응답 |
| `Qwen2.5-72B-Instruct` | 80GB+ | 다국어 지원 |

---

## Provider Factory

설정에 따라 적절한 프로바이더를 생성합니다.

```python
# providers/factory.py

class ProviderFactory:
    """프로바이더 팩토리"""

    PROVIDERS = {
        "openrouter": OpenRouterProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
        "vllm": VLLMProvider,
    }

    @classmethod
    def create(cls, config: dict) -> LLMProvider:
        """설정에서 프로바이더 생성"""
        providers_config = config.get("providers", {})

        # 첫 번째 활성화된 프로바이더 사용
        for name, provider_config in providers_config.items():
            if name in cls.PROVIDERS and provider_config.get("apiKey"):
                provider_class = cls.PROVIDERS[name]

                if name == "vllm":
                    return provider_class(
                        api_key=provider_config["apiKey"],
                        api_base=provider_config["apiBase"],
                    )
                else:
                    return provider_class(api_key=provider_config["apiKey"])

        raise ValueError("No valid provider configured")

    @classmethod
    def create_by_name(cls, name: str, config: dict) -> LLMProvider:
        """이름으로 특정 프로바이더 생성"""
        if name not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {name}")

        provider_config = config.get("providers", {}).get(name, {})
        provider_class = cls.PROVIDERS[name]

        if name == "vllm":
            return provider_class(
                api_key=provider_config.get("apiKey", ""),
                api_base=provider_config.get("apiBase", "http://localhost:8000/v1"),
            )
        else:
            return provider_class(api_key=provider_config.get("apiKey", ""))
```

---

## 모델 선택 가이드

### 용도별 추천

| 용도 | 추천 모델 | 이유 |
|-----|---------|------|
| **범용** | `anthropic/claude-opus-4-5` | 최고 품질 |
| **코딩** | `anthropic/claude-sonnet-4` | 코드 특화 |
| **빠른 응답** | `groq/llama-3.1-70b` | 초저지연 |
| **비용 효율** | `deepseek/deepseek-chat` | 저렴한 가격 |
| **프라이버시** | `vllm/local` | 로컬 실행 |

### 비용 비교 (대략적)

| 프로바이더 | 입력 (1M 토큰) | 출력 (1M 토큰) |
|----------|--------------|---------------|
| Anthropic Claude Opus | $15 | $75 |
| Anthropic Claude Sonnet | $3 | $15 |
| OpenAI GPT-4 Turbo | $10 | $30 |
| DeepSeek | $0.14 | $0.28 |
| Groq | 무료 (제한적) | 무료 (제한적) |

---

## 프로바이더 전환

런타임에 프로바이더를 전환할 수 있습니다.

```python
# 설정 파일에서 여러 프로바이더 정의
config = {
    "providers": {
        "openrouter": {"apiKey": "..."},
        "groq": {"apiKey": "..."},
    }
}

# 용도에 따라 선택
main_provider = ProviderFactory.create_by_name("openrouter", config)
fast_provider = ProviderFactory.create_by_name("groq", config)

# 빠른 응답이 필요할 때
response = await fast_provider.complete(...)

# 품질이 중요할 때
response = await main_provider.complete(...)
```

---

*다음 글에서는 확장 및 커스터마이징 방법을 알아봅니다.*
