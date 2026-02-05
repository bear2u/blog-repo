---
layout: post
title: "Nanobot 완벽 가이드 (4) - Agent Loop"
date: 2026-02-05
permalink: /nanobot-guide-04-agent-loop/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Agent Loop, LLM, Context, Iteration]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 핵심 처리 엔진인 Agent Loop를 분석합니다."
---

## Agent Loop 개요

**Agent Loop**은 Nanobot의 핵심 처리 엔진입니다. 메시지 수신부터 도구 실행, 응답 전송까지 전체 흐름을 관리합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Loop 역할                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. 메시지 버스에서 메시지 수신                             │
│  2. 히스토리, 메모리, 스킬로 컨텍스트 구성                  │
│  3. LLM 호출                                                │
│  4. 도구 호출 실행                                          │
│  5. 응답 전송                                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## AgentLoop 클래스

```python
# agent/loop.py

class AgentLoop:
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: ExecToolConfig | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations

        # 서브 컴포넌트 초기화
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(...)

        self._register_default_tools()
```

**주요 속성:**

| 속성 | 설명 |
|------|------|
| `bus` | 메시지 버스 (인바운드/아웃바운드) |
| `provider` | LLM 프로바이더 |
| `workspace` | 워크스페이스 경로 |
| `model` | 사용할 LLM 모델 |
| `max_iterations` | 최대 반복 횟수 (기본 20) |
| `context` | 컨텍스트 빌더 |
| `sessions` | 세션 매니저 |
| `tools` | 도구 레지스트리 |
| `subagents` | 서브에이전트 매니저 |

---

## 기본 도구 등록

```python
def _register_default_tools(self) -> None:
    """Register the default set of tools."""

    # 파일 도구
    self.tools.register(ReadFileTool())
    self.tools.register(WriteFileTool())
    self.tools.register(EditFileTool())
    self.tools.register(ListDirTool())

    # 셸 도구
    self.tools.register(ExecTool(
        working_dir=str(self.workspace),
        timeout=self.exec_config.timeout,
        restrict_to_workspace=self.exec_config.restrict_to_workspace,
    ))

    # 웹 도구
    self.tools.register(WebSearchTool(api_key=self.brave_api_key))
    self.tools.register(WebFetchTool())

    # 메시지 도구
    self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))

    # 서브에이전트 도구
    self.tools.register(SpawnTool(manager=self.subagents))
```

**등록되는 도구:**

| 도구 | 설명 |
|------|------|
| `read_file` | 파일 읽기 |
| `write_file` | 파일 쓰기 |
| `edit_file` | 파일 편집 |
| `list_dir` | 디렉토리 목록 |
| `exec` | 셸 명령 실행 |
| `web_search` | 웹 검색 |
| `web_fetch` | 웹 페이지 가져오기 |
| `message` | 사용자에게 메시지 전송 |
| `spawn` | 서브에이전트 생성 |

---

## 메인 루프

```python
async def run(self) -> None:
    """Run the agent loop, processing messages from the bus."""
    self._running = True
    logger.info("Agent loop started")

    async for message in self.bus.subscribe_inbound():
        if not self._running:
            break

        try:
            await self._process_message(message)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # 에러 응답 전송
            await self.bus.publish_outbound(OutboundMessage(
                channel=message.channel,
                chat_id=message.chat_id,
                content=f"Sorry, an error occurred: {str(e)}"
            ))
```

---

## 메시지 처리

```python
async def _process_message(self, message: InboundMessage) -> None:
    """Process a single inbound message."""

    session_id = f"{message.channel}:{message.chat_id}"

    # 1. 세션에서 히스토리 로드
    history = self.sessions.get_history(session_id)

    # 2. 시스템 프롬프트 구성
    system_prompt = self.context.build_system_prompt()

    # 3. 도구 정의 가져오기
    tool_definitions = self.tools.get_definitions()

    # 4. 메시지 히스토리 구성
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": message.content}
    ]

    # 5. 에이전트 루프 실행
    response = await self._run_agent_loop(
        messages=messages,
        tools=tool_definitions,
        session_id=session_id,
    )

    # 6. 응답 전송
    await self.bus.publish_outbound(OutboundMessage(
        channel=message.channel,
        chat_id=message.chat_id,
        content=response,
    ))

    # 7. 세션에 대화 저장
    self.sessions.add_message(session_id, "user", message.content)
    self.sessions.add_message(session_id, "assistant", response)
```

---

## 에이전트 루프 실행

```python
async def _run_agent_loop(
    self,
    messages: list[dict],
    tools: list[dict],
    session_id: str,
) -> str:
    """Run the agent loop until completion or max iterations."""

    iteration = 0

    while iteration < self.max_iterations:
        iteration += 1
        logger.debug(f"Agent loop iteration {iteration}")

        # LLM 호출
        response = await self.provider.complete(
            model=self.model,
            messages=messages,
            tools=tools,
        )

        # 도구 호출이 있는지 확인
        if response.tool_calls:
            # 도구 실행
            tool_results = await self._execute_tools(response.tool_calls)

            # 어시스턴트 메시지 추가
            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": response.tool_calls,
            })

            # 도구 결과 추가
            for result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.content,
                })

            # 다음 반복으로
            continue

        # 도구 호출이 없으면 완료
        return response.content

    # max_iterations 초과
    return "I've been running for too long. Let me stop here."
```

**에이전트 루프 흐름:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Loop 흐름                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  while iteration < max_iterations:                          │
│      │                                                       │
│      ├── LLM 호출                                           │
│      │   └── response = provider.complete(messages, tools)  │
│      │                                                       │
│      ├── if response.tool_calls:                            │
│      │   ├── 도구 실행                                      │
│      │   ├── 어시스턴트 메시지 추가                         │
│      │   ├── 도구 결과 추가                                 │
│      │   └── continue (다음 반복)                           │
│      │                                                       │
│      └── else:                                              │
│          └── return response.content (완료)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 도구 실행

```python
async def _execute_tools(
    self,
    tool_calls: list[ToolCall],
) -> list[ToolResult]:
    """Execute tool calls and return results."""

    results = []

    for call in tool_calls:
        tool = self.tools.get(call.name)

        if not tool:
            results.append(ToolResult(
                tool_call_id=call.id,
                content=f"Unknown tool: {call.name}",
            ))
            continue

        try:
            # 도구 실행
            output = await tool.execute(**call.arguments)

            results.append(ToolResult(
                tool_call_id=call.id,
                content=output,
            ))

        except Exception as e:
            logger.error(f"Tool {call.name} failed: {e}")
            results.append(ToolResult(
                tool_call_id=call.id,
                content=f"Tool execution failed: {str(e)}",
            ))

    return results
```

---

## Context Builder

시스템 프롬프트를 구성하는 컨텍스트 빌더입니다.

```python
# agent/context.py

class ContextBuilder:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def build_system_prompt(self) -> str:
        """Build the system prompt from workspace files."""

        parts = []

        # SOUL.md - 성격/가치관
        soul = self._read_file("SOUL.md")
        if soul:
            parts.append(f"# Soul\n\n{soul}")

        # AGENTS.md - 지침
        agents = self._read_file("AGENTS.md")
        if agents:
            parts.append(f"# Instructions\n\n{agents}")

        # TOOLS.md - 도구 문서
        tools = self._read_file("TOOLS.md")
        if tools:
            parts.append(f"# Tools\n\n{tools}")

        # USER.md - 사용자 정보
        user = self._read_file("USER.md")
        if user:
            parts.append(f"# User\n\n{user}")

        # MEMORY.md - 장기 메모리
        memory = self._read_file("MEMORY.md")
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # 스킬 설명
        skills = self._load_skills()
        if skills:
            parts.append(f"# Skills\n\n{skills}")

        return "\n\n---\n\n".join(parts)

    def _read_file(self, filename: str) -> str | None:
        """Read a file from workspace."""
        path = self.workspace / filename
        if path.exists():
            return path.read_text()
        return None
```

---

## Session Manager

대화 히스토리를 관리합니다.

```python
# session/manager.py

class SessionManager:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions: dict[str, list[dict]] = {}

    def get_history(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        return self.sessions.get(session_id, [])

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a message to session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append({
            "role": role,
            "content": content,
        })

        # 히스토리 제한 (최근 N개만 유지)
        max_history = 50
        if len(self.sessions[session_id]) > max_history:
            self.sessions[session_id] = self.sessions[session_id][-max_history:]

    def clear(self, session_id: str) -> None:
        """Clear session history."""
        if session_id in self.sessions:
            del self.sessions[session_id]
```

---

## 서브에이전트

백그라운드에서 작업을 실행하는 서브에이전트입니다.

```python
# agent/subagent.py

class SubagentManager:
    """Manage background subagents."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str,
        **kwargs,
    ):
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model
        self.running_tasks: dict[str, asyncio.Task] = {}

    async def spawn(self, task: str, label: str | None = None) -> str:
        """Spawn a new subagent to handle a task."""
        task_id = str(uuid.uuid4())[:8]
        label = label or task[:30]

        # 백그라운드에서 실행
        async_task = asyncio.create_task(
            self._run_subagent(task_id, task)
        )
        self.running_tasks[task_id] = async_task

        return f"Spawned subagent {task_id}: {label}"

    async def _run_subagent(self, task_id: str, task: str) -> None:
        """Run a subagent to completion."""
        try:
            # 간소화된 에이전트 루프 실행
            # ...
            pass
        finally:
            del self.running_tasks[task_id]
```

---

## 최적화 팁

### 1. max_iterations 조정

```json
{
  "agents": {
    "defaults": {
      "maxIterations": 10
    }
  }
}
```

복잡한 작업: 20+
간단한 작업: 5-10

### 2. 컨텍스트 크기 관리

- MEMORY.md를 작게 유지
- 불필요한 스킬 로드 방지
- 세션 히스토리 제한

### 3. 도구 선택적 등록

```python
# 필요한 도구만 등록
loop.tools.register(ReadFileTool())
loop.tools.register(ExecTool())
# web 도구는 제외
```

---

*다음 글에서는 Tools 시스템을 분석합니다.*
