---
layout: post
title: "OpenCode 가이드 - AI 프로바이더"
date: 2025-02-04
categories: [AI 코딩 에이전트, OpenCode]
tags: [opencode, providers, anthropic, openai, google, azure]
author: anomalyco
original_url: https://github.com/anomalyco/opencode
---

## 멀티 프로바이더 지원

OpenCode의 핵심 차별점 중 하나는 **프로바이더 독립성**입니다. 단일 AI 벤더에 종속되지 않고, 다양한 프로바이더를 자유롭게 선택할 수 있습니다.

## 지원 프로바이더 목록

### 번들 프로바이더

OpenCode에 기본 내장된 AI SDK 프로바이더들:

| 프로바이더 | SDK 패키지 | 주요 모델 |
|-----------|-----------|-----------|
| **Anthropic** | `@ai-sdk/anthropic` | Claude 4 Opus, Sonnet, Haiku |
| **OpenAI** | `@ai-sdk/openai` | GPT-5, GPT-4o, o1 |
| **Google** | `@ai-sdk/google` | Gemini 2.5, Gemini 2.0 |
| **Azure** | `@ai-sdk/azure` | Azure OpenAI 모델 |
| **AWS Bedrock** | `@ai-sdk/amazon-bedrock` | Claude, Titan 등 |
| **Google Vertex** | `@ai-sdk/google-vertex` | Vertex AI 모델 |
| **OpenRouter** | `@openrouter/ai-sdk-provider` | 다양한 모델 중개 |
| **xAI** | `@ai-sdk/xai` | Grok |
| **Mistral** | `@ai-sdk/mistral` | Mistral Large, Medium |
| **Groq** | `@ai-sdk/groq` | LLaMA, Mixtral |
| **DeepInfra** | `@ai-sdk/deepinfra` | 오픈소스 모델 |
| **Cerebras** | `@ai-sdk/cerebras` | Cerebras 모델 |
| **Cohere** | `@ai-sdk/cohere` | Command R+ |
| **Together AI** | `@ai-sdk/togetherai` | 오픈소스 모델 |
| **Perplexity** | `@ai-sdk/perplexity` | Perplexity 모델 |
| **GitLab** | `@gitlab/gitlab-ai-provider` | GitLab Duo |
| **GitHub Copilot** | 커스텀 SDK | Copilot 모델 |

### 특수 프로바이더

- **OpenCode Zen**: OpenCode에서 제공하는 관리형 서비스
- **OpenAI Compatible**: OpenAI API 호환 서버 (Ollama, LM Studio 등)

## 프로바이더 설정

### 환경 변수 방식

```bash
# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google
export GOOGLE_API_KEY="..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# Mistral
export MISTRAL_API_KEY="..."

# Groq
export GROQ_API_KEY="..."
```

### 설정 파일 방식

`~/.config/opencode/opencode.json` 또는 프로젝트 `opencode.json`:

```json
{
  "provider": {
    "anthropic": {
      "apiKey": "sk-ant-..."
    },
    "openai": {
      "apiKey": "sk-...",
      "options": {
        "baseURL": "https://api.openai.com/v1"
      }
    },
    "google": {
      "apiKey": "..."
    },
    "azure": {
      "apiKey": "...",
      "options": {
        "resourceName": "your-resource"
      }
    }
  }
}
```

## 프로바이더별 상세 설정

### Anthropic

```json
{
  "provider": {
    "anthropic": {
      "apiKey": "sk-ant-...",
      "options": {
        "headers": {
          "anthropic-beta": "claude-code-20250219"
        }
      }
    }
  }
}
```

### OpenAI

```json
{
  "provider": {
    "openai": {
      "apiKey": "sk-...",
      "options": {
        "organization": "org-...",
        "project": "proj-..."
      }
    }
  }
}
```

### Azure OpenAI

```json
{
  "provider": {
    "azure": {
      "apiKey": "...",
      "options": {
        "resourceName": "your-resource",
        "deploymentName": "gpt-4",
        "apiVersion": "2024-02-01"
      }
    }
  }
}
```

### AWS Bedrock

```json
{
  "provider": {
    "amazon-bedrock": {
      "options": {
        "region": "us-east-1",
        "profile": "default"
      }
    }
  }
}
```

환경 변수로 자격 증명:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."  # 임시 자격 증명 시
export AWS_REGION="us-east-1"
export AWS_PROFILE="default"    # 프로필 사용 시
```

### Google Vertex AI

```json
{
  "provider": {
    "google-vertex": {
      "options": {
        "project": "your-project-id",
        "location": "us-central1"
      }
    }
  }
}
```

### OpenRouter

```json
{
  "provider": {
    "openrouter": {
      "apiKey": "sk-or-...",
      "options": {
        "siteUrl": "https://your-app.com",
        "appName": "Your App"
      }
    }
  }
}
```

### OpenAI Compatible (Ollama, LM Studio 등)

```json
{
  "provider": {
    "openai-compatible": {
      "options": {
        "baseURL": "http://localhost:11434/v1",
        "apiKey": "ollama"
      },
      "models": {
        "llama3": {
          "name": "llama3:latest"
        }
      }
    }
  }
}
```

### GitHub Copilot

```json
{
  "provider": {
    "github-copilot": {
      "enabled": true
    }
  }
}
```

Copilot 인증:

```bash
opencode auth github-copilot
```

## 모델 선택

### 기본 모델 설정

```json
{
  "model": "anthropic/claude-sonnet-4-20250514"
}
```

### 모델 형식

```
{provider}/{model-id}

예:
- anthropic/claude-opus-4-20250514
- openai/gpt-5
- google/gemini-2.5-pro
- azure/gpt-4
- openrouter/anthropic/claude-sonnet-4
```

### 에이전트별 모델 지정

```json
{
  "agent": {
    "fast": {
      "name": "fast",
      "model": "anthropic/claude-haiku-3-5-20241022"
    },
    "powerful": {
      "name": "powerful",
      "model": "anthropic/claude-opus-4-20250514"
    }
  }
}
```

## OpenCode Zen

OpenCode에서 제공하는 관리형 서비스입니다.

### 인증

```bash
opencode auth
```

브라우저에서 인증 후 자동으로 설정됩니다.

### 특징

- API 키 관리 불필요
- 여러 모델 통합 접근
- 사용량 기반 과금
- 무료 모델 제공

### 무료 모델

API 키 없이도 사용 가능한 무료 모델이 제공됩니다:

```typescript
// 무료 모델은 cost.input === 0
if (!hasKey) {
  for (const [key, value] of Object.entries(input.models)) {
    if (value.cost.input === 0) continue
    delete input.models[key]
  }
}
```

## 인증 관리

### 인증 상태 확인

```bash
opencode auth status
```

### 프로바이더별 인증

```bash
# Anthropic
opencode auth anthropic

# OpenAI
opencode auth openai

# GitHub Copilot
opencode auth github-copilot
```

### 인증 제거

```bash
opencode auth remove anthropic
```

## OAuth 지원

일부 프로바이더는 OAuth 인증을 지원합니다:

```json
{
  "provider": {
    "custom-oauth": {
      "auth": {
        "type": "oauth",
        "clientId": "your-client-id",
        "authorizationUrl": "https://provider.com/oauth/authorize",
        "tokenUrl": "https://provider.com/oauth/token",
        "scope": "read write"
      }
    }
  }
}
```

## Well-Known 설정

조직 수준의 원격 설정을 지원합니다:

```json
// https://your-org.com/.well-known/opencode
{
  "config": {
    "provider": {
      "anthropic": {
        "apiKey": "${ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

## 프로바이더 선택 가이드

| 용도 | 추천 프로바이더 | 모델 |
|------|----------------|------|
| 일반 코딩 | Anthropic | Claude Sonnet 4 |
| 복잡한 추론 | Anthropic | Claude Opus 4 |
| 빠른 응답 | Anthropic | Claude Haiku 3.5 |
| 비용 절감 | Groq | LLaMA 3.3 70B |
| 로컬 실행 | OpenAI Compatible | Ollama + LLaMA |
| 엔터프라이즈 | Azure | Azure OpenAI |

## 다음 단계

다음 챕터에서는 상세한 설정 옵션과 권한 시스템을 알아봅니다.

---

**이전 글**: [내장 도구](/opencode-guide-05-tools/)

**다음 글**: [설정 및 권한](/opencode-guide-07-configuration/)
