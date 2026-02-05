---
layout: post
title: "Nanobot 완벽 가이드 (5) - Tools 시스템"
date: 2025-02-05
permalink: /nanobot-guide-05-tools/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Tools, Function Calling, Filesystem, Shell]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 도구 시스템과 내장 도구들을 분석합니다."
---

## Tools 시스템 개요

Nanobot의 도구 시스템은 에이전트가 외부 세계와 상호작용할 수 있게 합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Tools 시스템 구조                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  Tool Registry                       │    │
│  │                                                      │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │read_file│ │write_   │ │edit_file│ │list_dir │   │    │
│  │  └─────────┘ │file     │ └─────────┘ └─────────┘   │    │
│  │              └─────────┘                            │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │  exec   │ │web_     │ │web_fetch│ │ message │   │    │
│  │  └─────────┘ │search   │ └─────────┘ └─────────┘   │    │
│  │              └─────────┘                            │    │
│  │  ┌─────────┐                                        │    │
│  │  │  spawn  │                                        │    │
│  │  └─────────┘                                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Tool 베이스 클래스

```python
# agent/tools/base.py

from abc import ABC, abstractmethod

class Tool(ABC):
    """도구 베이스 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름 (LLM이 호출할 때 사용)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """도구 설명 (LLM이 선택할 때 참고)"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 형식의 파라미터 정의"""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """도구 실행 및 결과 반환"""
        pass

    def to_definition(self) -> dict:
        """LLM용 도구 정의 반환"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
```

---

## Tool Registry

```python
# agent/tools/registry.py

class ToolRegistry:
    """도구 레지스트리"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """도구 등록"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """이름으로 도구 조회"""
        return self._tools.get(name)

    def get_definitions(self) -> list[dict]:
        """LLM용 모든 도구 정의 반환"""
        return [tool.to_definition() for tool in self._tools.values()]

    def list_names(self) -> list[str]:
        """등록된 도구 이름 목록"""
        return list(self._tools.keys())
```

---

## 내장 도구

### 1. 파일 도구 (Filesystem)

#### read_file

```python
class ReadFileTool(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        }

    async def execute(self, path: str) -> str:
        try:
            return Path(path).read_text()
        except Exception as e:
            return f"Error reading file: {e}"
```

#### write_file

```python
class WriteFileTool(Tool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file (creates parent directories if needed)."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write"
                }
            },
            "required": ["path", "content"]
        }

    async def execute(self, path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
```

#### edit_file

```python
class EditFileTool(Tool):
    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "Edit a file by replacing specific text."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"}
            },
            "required": ["path", "old_text", "new_text"]
        }

    async def execute(self, path: str, old_text: str, new_text: str) -> str:
        try:
            p = Path(path)
            content = p.read_text()

            if old_text not in content:
                return f"Text not found in {path}"

            new_content = content.replace(old_text, new_text, 1)
            p.write_text(new_content)

            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing file: {e}"
```

#### list_dir

```python
class ListDirTool(Tool):
    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List contents of a directory."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path",
                    "default": "."
                }
            }
        }

    async def execute(self, path: str = ".") -> str:
        try:
            p = Path(path)
            items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))

            result = []
            for item in items:
                prefix = "📁 " if item.is_dir() else "📄 "
                result.append(f"{prefix}{item.name}")

            return "\n".join(result) or "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {e}"
```

---

### 2. 셸 도구 (Shell)

#### exec

```python
class ExecTool(Tool):
    # 위험한 명령어 차단
    DANGEROUS_COMMANDS = [
        "rm -rf /",
        "rm -rf ~",
        "format",
        "mkfs",
        "dd if=",
        "shutdown",
        "reboot",
        "> /dev/sda",
    ]

    def __init__(
        self,
        working_dir: str = ".",
        timeout: int = 60,
        restrict_to_workspace: bool = False,
    ):
        self.working_dir = working_dir
        self.timeout = timeout
        self.restrict_to_workspace = restrict_to_workspace

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return output."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory (optional)"
                }
            },
            "required": ["command"]
        }

    async def execute(self, command: str, working_dir: str | None = None) -> str:
        # 위험한 명령어 체크
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command.lower():
                return f"Blocked: dangerous command detected"

        cwd = working_dir or self.working_dir

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode() + stderr.decode()

            # 출력 길이 제한
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"

            return output or "(no output)"

        except asyncio.TimeoutError:
            return f"Command timed out after {self.timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"
```

---

### 3. 웹 도구 (Web)

#### web_search

```python
class WebSearchTool(Tool):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using Brave Search API."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }

    async def execute(self, query: str, count: int = 5) -> str:
        if not self.api_key:
            return "Web search not configured (missing API key)"

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": self.api_key}
            )

            data = response.json()
            results = []

            for item in data.get("web", {}).get("results", []):
                results.append(
                    f"**{item['title']}**\n"
                    f"{item['url']}\n"
                    f"{item.get('description', '')}\n"
                )

            return "\n---\n".join(results) or "No results found"
```

#### web_fetch

```python
class WebFetchTool(Tool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch and extract main content from a URL."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extractMode": {
                    "type": "string",
                    "enum": ["markdown", "text"],
                    "default": "markdown"
                },
                "maxChars": {"type": "integer", "default": 50000}
            },
            "required": ["url"]
        }

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int = 50000
    ) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                html = response.text

            # readability로 본문 추출
            from readability import Document
            doc = Document(html)
            content = doc.summary()

            # HTML → 텍스트 변환
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            if len(text) > maxChars:
                text = text[:maxChars] + "\n... (truncated)"

            return text

        except Exception as e:
            return f"Error fetching URL: {e}"
```

---

### 4. 메시지 도구

#### message

```python
class MessageTool(Tool):
    def __init__(self, send_callback):
        self.send_callback = send_callback

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "channel": {"type": "string"},
                "chat_id": {"type": "string"}
            },
            "required": ["content"]
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None
    ) -> str:
        await self.send_callback(OutboundMessage(
            channel=channel or "default",
            chat_id=chat_id or "default",
            content=content
        ))
        return "Message sent"
```

---

### 5. 서브에이전트 도구

#### spawn

```python
class SpawnTool(Tool):
    def __init__(self, manager: SubagentManager):
        self.manager = manager

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a subagent to handle a task in the background."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task description for the subagent"
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the task"
                }
            },
            "required": ["task"]
        }

    async def execute(self, task: str, label: str | None = None) -> str:
        return await self.manager.spawn(task, label)
```

---

## 커스텀 도구 추가

```python
# 1. Tool 상속
class MyCustomTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"

    @property
    def description(self) -> str:
        return "My custom tool description"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }

    async def execute(self, input: str) -> str:
        # 도구 로직
        return f"Result: {input}"

# 2. AgentLoop에 등록
loop.tools.register(MyCustomTool())
```

---

*다음 글에서는 Channels 시스템을 분석합니다.*
