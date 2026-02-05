---
layout: post
title: "Nanobot 완벽 가이드 (7) - Skills 시스템"
date: 2026-02-05
permalink: /nanobot-guide-07-skills/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Skills, Plugins, Extensions, Automation]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 스킬 시스템과 커스텀 스킬 개발 방법을 알아봅니다."
---

## Skills 시스템 개요

**Skills**는 Nanobot의 기능을 확장하는 플러그인 시스템입니다. 특정 도메인에 특화된 지식과 도구를 패키지로 제공합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Skills 시스템                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  Skill Loader                        │   │
│  │                                                      │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │   │
│  │  │ github  │ │ weather │ │  tmux   │ │summarize│   │   │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │   │
│  │                                                      │   │
│  │  ┌─────────┐ ┌─────────┐                            │   │
│  │  │ custom1 │ │ custom2 │  ...                       │   │
│  │  └─────────┘ └─────────┘                            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  스킬 구성요소:                                              │
│  • system.md  - 스킬 전용 시스템 프롬프트                    │
│  • tools/     - 스킬 전용 도구들                             │
│  • config     - 스킬 설정                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 내장 스킬

Nanobot에 번들로 포함된 스킬들입니다.

### 1. GitHub 스킬

GitHub API와 연동하는 스킬입니다.

```
skills/github/
├── system.md    # GitHub 작업 가이드
├── tools/
│   ├── list_repos.py
│   ├── create_issue.py
│   ├── create_pr.py
│   └── search_code.py
└── config.yaml
```

**system.md:**

```markdown
# GitHub Skill

You have access to GitHub tools. Use them to:
- List and search repositories
- Create and manage issues
- Create pull requests
- Search code across repositories

When the user asks about GitHub-related tasks, use these tools.
```

**사용 예시:**

```
User: "내 레포지토리 목록 보여줘"
Agent: [list_repos 도구 호출]
```

### 2. Weather 스킬

날씨 정보를 제공하는 스킬입니다.

```
skills/weather/
├── system.md
└── tools/
    └── get_weather.py
```

**tools/get_weather.py:**

```python
class GetWeatherTool(Tool):
    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "Get current weather for a location."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                }
            },
            "required": ["location"]
        }

    async def execute(self, location: str) -> str:
        # OpenWeatherMap API 호출
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"
                }
            )
            data = response.json()

            return (
                f"Weather in {location}:\n"
                f"Temperature: {data['main']['temp']}°C\n"
                f"Condition: {data['weather'][0]['description']}\n"
                f"Humidity: {data['main']['humidity']}%"
            )
```

### 3. Tmux 스킬

터미널 세션 관리를 위한 스킬입니다.

```
skills/tmux/
├── system.md
└── tools/
    ├── list_sessions.py
    ├── create_session.py
    ├── send_keys.py
    └── capture_pane.py
```

**사용 예시:**

```
User: "백그라운드에서 서버 실행해줘"
Agent: [tmux create_session → send_keys 도구 호출]
```

### 4. Summarize 스킬

텍스트 요약을 위한 스킬입니다.

```
skills/summarize/
├── system.md
└── tools/
    └── summarize.py
```

---

## Skill Loader

스킬을 로드하고 관리하는 클래스입니다.

```python
# agent/skills.py

class SkillLoader:
    """스킬 로더"""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.skills_dir = workspace / "skills"
        self.bundled_skills = Path(__file__).parent.parent / "skills"

    def load_all(self) -> list[Skill]:
        """모든 활성화된 스킬 로드"""
        skills = []

        # 번들 스킬 로드
        for skill_dir in self.bundled_skills.iterdir():
            if skill_dir.is_dir() and self._is_enabled(skill_dir.name):
                skill = self._load_skill(skill_dir)
                if skill:
                    skills.append(skill)

        # 워크스페이스 커스텀 스킬 로드
        if self.skills_dir.exists():
            for skill_dir in self.skills_dir.iterdir():
                if skill_dir.is_dir():
                    skill = self._load_skill(skill_dir)
                    if skill:
                        skills.append(skill)

        return skills

    def _load_skill(self, skill_dir: Path) -> Skill | None:
        """단일 스킬 로드"""
        try:
            # system.md 로드
            system_path = skill_dir / "system.md"
            system_prompt = system_path.read_text() if system_path.exists() else ""

            # 도구 로드
            tools = self._load_tools(skill_dir / "tools")

            # 설정 로드
            config = self._load_config(skill_dir)

            return Skill(
                name=skill_dir.name,
                system_prompt=system_prompt,
                tools=tools,
                config=config,
            )
        except Exception as e:
            logger.error(f"Failed to load skill {skill_dir.name}: {e}")
            return None

    def _load_tools(self, tools_dir: Path) -> list[Tool]:
        """스킬 도구 로드"""
        tools = []

        if not tools_dir.exists():
            return tools

        for tool_file in tools_dir.glob("*.py"):
            # 동적 모듈 로드
            spec = importlib.util.spec_from_file_location(
                tool_file.stem,
                tool_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Tool 클래스 찾기
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Tool) and obj is not Tool:
                    tools.append(obj())

        return tools

    def _is_enabled(self, skill_name: str) -> bool:
        """스킬 활성화 여부 확인"""
        # AGENTS.md에서 스킬 활성화 설정 확인
        agents_md = self.workspace / "AGENTS.md"
        if agents_md.exists():
            content = agents_md.read_text()
            # "skills: github, weather" 형태로 파싱
            # ...
        return True  # 기본값: 모두 활성화

    def get_combined_system_prompt(self, skills: list[Skill]) -> str:
        """모든 스킬의 시스템 프롬프트 결합"""
        prompts = []
        for skill in skills:
            if skill.system_prompt:
                prompts.append(f"## {skill.name.title()} Skill\n\n{skill.system_prompt}")
        return "\n\n---\n\n".join(prompts)
```

---

## Skill 데이터 클래스

```python
# agent/skills.py

@dataclass
class Skill:
    """스킬 정의"""
    name: str
    system_prompt: str
    tools: list[Tool]
    config: dict = field(default_factory=dict)

    def get_tool_definitions(self) -> list[dict]:
        """LLM용 도구 정의 반환"""
        return [tool.to_definition() for tool in self.tools]
```

---

## 커스텀 스킬 개발

### 1. 스킬 디렉토리 생성

```bash
mkdir -p ~/.nanobot/workspace/skills/my_skill/tools
```

### 2. system.md 작성

```markdown
# My Custom Skill

You have access to custom tools for [specific task].

## When to use
- Use `my_tool` when the user asks about [specific scenario]

## Guidelines
- Always [specific guideline]
- Remember to [important note]
```

### 3. 도구 구현

**tools/my_tool.py:**

```python
from agent.tools.base import Tool

class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "Description of what this tool does."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input description"
                },
                "option": {
                    "type": "boolean",
                    "description": "Optional flag",
                    "default": False
                }
            },
            "required": ["input"]
        }

    async def execute(self, input: str, option: bool = False) -> str:
        # 도구 로직 구현
        result = f"Processed: {input}"

        if option:
            result += " (with option)"

        return result
```

### 4. 설정 파일 (선택)

**config.yaml:**

```yaml
name: my_skill
version: 1.0.0
description: My custom skill for specific tasks

settings:
  api_key: ${MY_SKILL_API_KEY}
  timeout: 30

dependencies:
  - httpx
  - beautifulsoup4
```

---

## 스킬 활성화/비활성화

### AGENTS.md에서 설정

```markdown
# Agent Configuration

## Skills

Enable only specific skills:

```yaml
skills:
  - github
  - weather
```

Or disable specific skills:

```yaml
skills_disabled:
  - tmux
```
```

### 프로그래밍 방식

```python
# 특정 스킬만 로드
loader = SkillLoader(workspace)
skills = loader.load_all()

# 필터링
enabled_skills = [s for s in skills if s.name in ["github", "weather"]]
```

---

## 스킬 컨텍스트 통합

Agent Loop에서 스킬을 통합하는 방법입니다.

```python
# agent/loop.py

class AgentLoop:
    def __init__(self, ...):
        # ...
        self.skill_loader = SkillLoader(workspace)
        self.skills = self.skill_loader.load_all()

        # 스킬 도구 등록
        for skill in self.skills:
            for tool in skill.tools:
                self.tools.register(tool)

    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 구성"""
        parts = []

        # 기본 컨텍스트
        base_prompt = self.context.build_system_prompt()
        parts.append(base_prompt)

        # 스킬 프롬프트
        skill_prompt = self.skill_loader.get_combined_system_prompt(self.skills)
        if skill_prompt:
            parts.append(f"# Skills\n\n{skill_prompt}")

        return "\n\n---\n\n".join(parts)
```

---

## 실용적인 스킬 예시

### 1. 뉴스 스킬

```python
# skills/news/tools/get_news.py

class GetNewsTool(Tool):
    @property
    def name(self) -> str:
        return "get_news"

    async def execute(self, topic: str, count: int = 5) -> str:
        # NewsAPI 호출
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": topic,
                    "pageSize": count,
                    "apiKey": self.api_key
                }
            )

            articles = response.json()["articles"]

            result = []
            for article in articles:
                result.append(
                    f"**{article['title']}**\n"
                    f"{article['description']}\n"
                    f"[Read more]({article['url']})"
                )

            return "\n\n---\n\n".join(result)
```

### 2. 번역 스킬

```python
# skills/translate/tools/translate.py

class TranslateTool(Tool):
    @property
    def name(self) -> str:
        return "translate"

    async def execute(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto"
    ) -> str:
        # DeepL API 호출
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api-free.deepl.com/v2/translate",
                data={
                    "auth_key": self.api_key,
                    "text": text,
                    "target_lang": target_lang.upper(),
                    "source_lang": source_lang.upper() if source_lang != "auto" else None
                }
            )

            result = response.json()
            translated = result["translations"][0]["text"]

            return f"**Translation ({target_lang}):**\n{translated}"
```

### 3. 캘린더 스킬

```python
# skills/calendar/tools/add_event.py

class AddEventTool(Tool):
    @property
    def name(self) -> str:
        return "add_calendar_event"

    async def execute(
        self,
        title: str,
        start_time: str,
        end_time: str,
        description: str = ""
    ) -> str:
        # Google Calendar API 호출
        event = {
            'summary': title,
            'description': description,
            'start': {'dateTime': start_time, 'timeZone': 'Asia/Seoul'},
            'end': {'dateTime': end_time, 'timeZone': 'Asia/Seoul'},
        }

        # ... Google Calendar API 호출 로직

        return f"Event '{title}' added to calendar."
```

---

## 베스트 프랙티스

### 1. 명확한 system.md

```markdown
# Skill Name

## Purpose
Clearly describe what this skill does.

## Available Tools
- `tool1`: Description
- `tool2`: Description

## Usage Guidelines
1. When to use each tool
2. Expected input formats
3. Error handling notes

## Examples
- "User asks X" → Use `tool1`
- "User asks Y" → Use `tool2`
```

### 2. 에러 처리

```python
async def execute(self, **kwargs) -> str:
    try:
        # 도구 로직
        result = await self._do_work(**kwargs)
        return result
    except httpx.HTTPError as e:
        return f"Network error: {str(e)}"
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Tool error: {e}")
        return f"An error occurred: {str(e)}"
```

### 3. 타임아웃 설정

```python
async def execute(self, **kwargs) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # ...
```

---

*다음 글에서는 Cron & Heartbeat 시스템을 분석합니다.*
