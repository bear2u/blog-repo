---
layout: post
title: "Nanobot 완벽 가이드 (10) - 확장 및 커스터마이징"
date: 2026-02-05
permalink: /nanobot-guide-10-customization/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Customization, Extension, Workspace, Development]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot을 확장하고 커스터마이징하는 방법을 알아봅니다."
---

## 워크스페이스 커스터마이징

Nanobot의 동작은 워크스페이스 파일들을 통해 세밀하게 조정할 수 있습니다.

```
~/.nanobot/workspace/
├── AGENTS.md      # 에이전트 지침
├── SOUL.md        # 성격 정의
├── USER.md        # 사용자 정보
├── TOOLS.md       # 도구 문서
├── HEARTBEAT.md   # 주기적 작업
├── MEMORY.md      # 장기 메모리
├── memory/        # 일별 메모리
└── skills/        # 커스텀 스킬
```

---

## AGENTS.md - 에이전트 지침

에이전트의 행동 방식을 정의합니다.

### 기본 템플릿

```markdown
# Agent Configuration

## Role
You are a personal AI assistant specialized in software development.

## Guidelines

### Response Style
- Be concise and direct
- Use code examples when explaining technical concepts
- Always explain the "why" behind suggestions

### Tool Usage
- Prefer `read_file` before `edit_file`
- Use `exec` for quick commands, `spawn` for long-running tasks
- Always confirm before destructive operations

### Communication
- Use Korean when the user writes in Korean
- Include code blocks with language tags
- Break down complex tasks into steps

## Skills

Enable these skills:
- github
- weather

## Constraints

Do NOT:
- Access sensitive files without permission
- Make irreversible changes without confirmation
- Execute commands that affect system stability
```

---

## SOUL.md - 성격 정의

에이전트의 성격과 가치관을 정의합니다.

### 예시: 친근한 어시스턴트

```markdown
# Soul

## Personality
I am a friendly and helpful assistant with a warm personality.
I enjoy making complex topics accessible and celebrating small wins together.

## Values
- **Helpfulness**: Always prioritize being genuinely useful
- **Honesty**: Be truthful, even when the answer is "I don't know"
- **Patience**: Take time to understand the user's actual needs
- **Growth**: Encourage learning and experimentation

## Communication Style
- Warm but professional
- Uses occasional humor when appropriate
- Acknowledges emotions and frustrations
- Celebrates achievements, no matter how small

## Quirks
- Occasionally uses relevant analogies
- Asks clarifying questions when needed
- Admits limitations openly
```

### 예시: 전문 개발자

```markdown
# Soul

## Personality
I am a senior software engineer with 15+ years of experience.
I value clean code, proper architecture, and thoughtful design.

## Values
- **Quality**: Code should be maintainable and well-tested
- **Simplicity**: The best solution is often the simplest one
- **Pragmatism**: Perfect is the enemy of good
- **Teaching**: Share knowledge and explain reasoning

## Communication Style
- Direct and technical
- Uses proper terminology
- Provides context for decisions
- References best practices and patterns

## Preferences
- Prefer composition over inheritance
- Value readability over cleverness
- Test-driven development when appropriate
- Document public APIs
```

---

## USER.md - 사용자 정보

사용자에 대한 컨텍스트를 제공합니다.

```markdown
# User Profile

## About
- Name: John
- Role: Full-stack developer
- Experience: 5 years
- Location: Seoul, Korea

## Preferences
- Programming: TypeScript, Python, Go
- Editor: VS Code with Vim keybindings
- Terminal: iTerm2 with Zsh
- OS: macOS

## Current Projects
- Building a SaaS application with Next.js
- Learning Rust for systems programming
- Contributing to open source

## Work Style
- Prefers detailed explanations for complex topics
- Likes seeing alternative approaches
- Appreciates performance considerations
- Values security best practices

## Communication
- Primary language: Korean
- Can read English documentation
- Prefers code examples over lengthy explanations
```

---

## TOOLS.md - 도구 문서

도구 사용법에 대한 추가 문서입니다.

```markdown
# Tools Documentation

## File Operations

### read_file
Use this to read file contents before editing.
Always read the current state before making changes.

### write_file
Creates or overwrites a file.
Use for new files or complete rewrites.

### edit_file
For small, targeted changes.
Requires exact `old_text` match.

## Shell Operations

### exec
For quick commands that complete within 60 seconds.
Examples: `ls`, `git status`, `npm install`

### spawn (subagent)
For long-running tasks.
Examples: `npm run dev`, `pytest`, complex multi-step tasks

## Web Operations

### web_search
Use Brave Search API.
Best for: current events, documentation, general knowledge

### web_fetch
Fetch and parse web pages.
Best for: reading specific URLs, documentation pages

## Custom Tools

Any additional tools from skills will be documented here.
```

---

## MEMORY.md - 장기 메모리

에이전트가 기억해야 할 중요한 정보입니다.

```markdown
# Memory

## User Preferences
- Prefers tabs over spaces
- Uses 2-space indentation for JS/TS
- Likes functional programming style

## Project Context

### Main Project: SaaS App
- Repository: ~/projects/saas-app
- Stack: Next.js + Prisma + PostgreSQL
- Deployment: Vercel

### Side Project: CLI Tool
- Repository: ~/projects/cli-tool
- Language: Rust
- Status: Early development

## Past Interactions

### 2025-01-15
Helped set up ESLint and Prettier configuration.
User prefers strict type checking.

### 2025-01-20
Debugged a database connection issue.
PostgreSQL runs on port 5433 (not default 5432).

## Important Notes
- API keys are in `.env.local` (never commit!)
- Production database requires VPN connection
- CI/CD runs on GitHub Actions
```

---

## 커스텀 도구 추가

### 1. 도구 클래스 작성

```python
# ~/.nanobot/workspace/skills/my_tools/tools/custom_tool.py

from agent.tools.base import Tool

class DatabaseQueryTool(Tool):
    """커스텀 데이터베이스 쿼리 도구"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string

    @property
    def name(self) -> str:
        return "db_query"

    @property
    def description(self) -> str:
        return "Execute a read-only SQL query on the database."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute"
                }
            },
            "required": ["query"]
        }

    async def execute(self, query: str) -> str:
        # 안전성 검사
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed"

        import asyncpg

        conn = await asyncpg.connect(self.connection_string)
        try:
            rows = await conn.fetch(query)

            if not rows:
                return "No results found"

            # 결과 포맷팅
            headers = list(rows[0].keys())
            result = " | ".join(headers) + "\n"
            result += "-" * len(result) + "\n"

            for row in rows[:100]:  # 최대 100행
                result += " | ".join(str(v) for v in row.values()) + "\n"

            return result
        finally:
            await conn.close()
```

### 2. system.md 작성

```markdown
# Database Tools

You have access to a read-only database query tool.

## db_query

Use this tool to query the database when the user asks about data.

### Guidelines
- Only SELECT queries are allowed
- Limit results to avoid overwhelming output
- Use proper SQL formatting

### Examples
- "Show me the top 10 users" → `SELECT * FROM users LIMIT 10`
- "Count active subscriptions" → `SELECT COUNT(*) FROM subscriptions WHERE active = true`
```

---

## 커스텀 프로바이더 추가

OpenAI API 호환 서비스를 추가할 수 있습니다.

```python
# providers/custom.py

class CustomProvider(LLMProvider):
    """커스텀 OpenAI 호환 프로바이더"""

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
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                },
            )
            # ... 응답 파싱
```

### 설정

```json
{
  "providers": {
    "custom": {
      "apiKey": "xxx",
      "apiBase": "https://my-llm-server.com/v1"
    }
  }
}
```

---

## 채널 확장

새로운 메시징 플랫폼을 추가할 수 있습니다.

```python
# channels/discord.py

class DiscordChannel(Channel):
    """Discord 채널"""

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        import discord

        intents = discord.Intents.default()
        intents.message_content = True

        self.client = discord.Client(intents=intents)

        @self.client.event
        async def on_message(message):
            if message.author == self.client.user:
                return

            if not self.is_allowed(str(message.author.id)):
                return

            await self.bus.publish_inbound(InboundMessage(
                channel="discord",
                chat_id=str(message.channel.id),
                user_id=str(message.author.id),
                content=message.content,
            ))

        await self.client.start(self.config["token"])

    async def send_message(self, chat_id: str, content: str) -> None:
        channel = self.client.get_channel(int(chat_id))
        await channel.send(content)
```

---

## 스케줄 작업 확장

복잡한 스케줄 작업을 정의할 수 있습니다.

### HEARTBEAT.md 고급 예시

```markdown
# Advanced Heartbeat

## Schedule
Run every: 2 hour

## Tasks

### Market Analysis
Analyze current cryptocurrency market:
1. Fetch BTC, ETH prices from CoinGecko
2. Compare with 24h ago
3. If change > 5%, notify immediately

### Project Health Check
Check all my GitHub repos:
1. List repos with open PRs
2. Check for failing CI
3. Summarize issues needing attention

### Learning Reminder
Based on my learning goals in USER.md:
1. Suggest one Rust concept to study today
2. Find a relevant exercise or tutorial
3. Create a small coding challenge

## Conditions

### Time Restrictions
Only run: 9 AM - 10 PM (Asia/Seoul)

### Day Restrictions
Skip: Weekends

### Rate Limits
- Market Analysis: Max 12 times/day
- Learning Reminder: Max 2 times/day
```

---

## 디버깅 및 로깅

### 로그 레벨 설정

```python
import logging

# 상세 로깅 활성화
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("nanobot")
```

### 디버그 모드

```bash
# 디버그 모드로 실행
NANOBOT_DEBUG=1 nanobot agent -m "Hello"
```

### 대화 로그 저장

```python
# 모든 대화를 파일로 저장
import json
from datetime import datetime

def log_conversation(session_id: str, messages: list[dict]):
    log_path = Path(f"~/.nanobot/logs/{session_id}.jsonl").expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "a") as f:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
        }
        f.write(json.dumps(entry) + "\n")
```

---

## 베스트 프랙티스

### 1. 점진적 커스터마이징

```markdown
1. 기본 설정으로 시작
2. 필요한 부분만 수정
3. 변경사항 테스트
4. 문서화
```

### 2. 백업 유지

```bash
# 워크스페이스 백업
cp -r ~/.nanobot/workspace ~/.nanobot/workspace.backup.$(date +%Y%m%d)
```

### 3. 버전 관리

```bash
cd ~/.nanobot/workspace
git init
git add .
git commit -m "Initial workspace configuration"
```

### 4. 환경 분리

```bash
# 개발용 워크스페이스
export NANOBOT_WORKSPACE=~/.nanobot/workspace-dev

# 프로덕션 워크스페이스
export NANOBOT_WORKSPACE=~/.nanobot/workspace-prod
```

---

## 결론

Nanobot은 **~4,000줄의 코드**로 강력한 개인 AI 어시스턴트를 구현합니다.

### 핵심 장점

```
┌─────────────────────────────────────────────────────────────┐
│                    Nanobot의 가치                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ Ultra-Lightweight - 읽고 수정하기 쉬운 코드              │
│  ✅ Multi-Channel - Telegram, WhatsApp, Feishu 지원         │
│  ✅ Multi-Provider - 모든 주요 LLM 연동                      │
│  ✅ Extensible - 도구, 스킬, 프로바이더 확장 가능            │
│  ✅ Automatable - Cron, Heartbeat로 자동화                   │
│  ✅ Customizable - 워크스페이스 파일로 세밀한 조정           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 다음 단계

1. **[GitHub](https://github.com/HKUDS/nanobot)** 에서 소스 코드 확인
2. **[Discord](https://discord.gg/MnCvHqpUGB)** 커뮤니티 참여
3. 자신만의 스킬과 도구 개발
4. 기여하기 (PR 환영!)

---

## 부록: 빠른 참조

### CLI 명령어

```bash
nanobot onboard          # 초기화
nanobot agent            # 대화형 모드
nanobot agent -m "..."   # 단일 메시지
nanobot gateway          # 채널 게이트웨이
nanobot status           # 상태 확인
nanobot cron list        # Cron 작업 목록
nanobot channels login   # WhatsApp 로그인
```

### 설정 파일 위치

```
~/.nanobot/
├── config.json          # 메인 설정
└── workspace/
    ├── AGENTS.md        # 에이전트 지침
    ├── SOUL.md          # 성격
    ├── USER.md          # 사용자 정보
    ├── TOOLS.md         # 도구 문서
    ├── HEARTBEAT.md     # 주기적 작업
    ├── MEMORY.md        # 장기 메모리
    └── skills/          # 커스텀 스킬
```

### 지원 프로바이더

| 프로바이더 | 설정 키 |
|----------|---------|
| OpenRouter | `providers.openrouter.apiKey` |
| Anthropic | `providers.anthropic.apiKey` |
| OpenAI | `providers.openai.apiKey` |
| Groq | `providers.groq.apiKey` |
| DeepSeek | `providers.deepseek.apiKey` |
| Gemini | `providers.gemini.apiKey` |
| vLLM | `providers.vllm.apiKey` + `apiBase` |

---

*이 가이드 시리즈가 Nanobot 이해에 도움이 되었기를 바랍니다!*
