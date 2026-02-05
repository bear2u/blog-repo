---
layout: post
title: "Nanobot 완벽 가이드 (3) - 아키텍처 분석"
date: 2025-02-05
permalink: /nanobot-guide-03-architecture/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Architecture, Module, Design]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 모듈 구조와 설계 원칙을 분석합니다."
---

## 아키텍처 개요

Nanobot은 **모듈화된 이벤트 기반** 아키텍처를 사용합니다. 메시지 버스를 통해 컴포넌트들이 느슨하게 연결되어 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                       Nanobot 아키텍처                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Channels                           │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐   │   │
│  │   │Telegram │  │WhatsApp │  │ Feishu  │  │  CLI  │   │   │
│  │   └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘   │   │
│  └────────┼────────────┼────────────┼───────────┼───────┘   │
│           │            │            │           │            │
│           ▼            ▼            ▼           ▼            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Message Bus                        │   │
│  │           InboundMessage ←→ OutboundMessage           │   │
│  └─────────────────────────┬────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Agent Loop                         │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │              Context Builder                     │ │   │
│  │  │    System Prompt + Memory + Skills + Tools       │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                         │                             │   │
│  │                         ▼                             │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │               LLM Provider                       │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                         │                             │   │
│  │                         ▼                             │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │              Tool Registry                       │ │   │
│  │  │   read_file | write_file | exec | web_search    │ │   │
│  │  │   web_fetch | message | spawn                    │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Supporting Modules                   │   │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐   │   │
│  │   │ Session │  │  Cron   │  │Heartbeat│  │Skills │   │   │
│  │   │ Manager │  │ Manager │  │  Check  │  │Loader │   │   │
│  │   └─────────┘  └─────────┘  └─────────┘  └───────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 프로젝트 구조

```
nanobot/
├── __init__.py         # 패키지 초기화
├── __main__.py         # python -m nanobot 진입점
│
├── agent/              # 🧠 핵심 에이전트 로직
│   ├── loop.py         #    에이전트 루프 (LLM ↔ 도구 실행)
│   ├── context.py      #    프롬프트 빌더
│   ├── memory.py       #    영구 메모리
│   ├── skills.py       #    스킬 로더
│   ├── subagent.py     #    백그라운드 작업 실행
│   └── tools/          #    내장 도구
│       ├── base.py     #    Tool 베이스 클래스
│       ├── registry.py #    ToolRegistry
│       ├── filesystem.py   # 파일 작업
│       ├── shell.py    #    exec 도구
│       ├── web.py      #    웹 검색/페치
│       ├── message.py  #    메시지 전송
│       └── spawn.py    #    서브에이전트 생성
│
├── skills/             # 🎯 번들 스킬
│   ├── github/         #    GitHub 통합
│   ├── weather/        #    날씨 정보
│   ├── tmux/           #    tmux 관리
│   ├── summarize/      #    텍스트 요약
│   └── skill-creator/  #    스킬 생성 도구
│
├── channels/           # 📱 메시징 채널
│   ├── base.py         #    BaseChannel 추상 클래스
│   ├── manager.py      #    ChannelManager
│   ├── telegram.py     #    Telegram 채널
│   ├── whatsapp.py     #    WhatsApp 채널
│   └── feishu.py       #    Feishu 채널
│
├── bus/                # 🚌 메시지 라우팅
│   ├── events.py       #    InboundMessage, OutboundMessage
│   └── queue.py        #    MessageBus
│
├── cron/               # ⏰ 스케줄된 작업
│   ├── scheduler.py    #    CronScheduler
│   └── jobs.py         #    Job 클래스
│
├── heartbeat/          # 💓 주기적 웨이크업
│   └── checker.py      #    HeartbeatChecker
│
├── providers/          # 🤖 LLM 프로바이더
│   ├── base.py         #    LLMProvider 베이스
│   └── litellm.py      #    LiteLLM 어댑터
│
├── session/            # 💬 대화 세션
│   └── manager.py      #    SessionManager
│
├── config/             # ⚙️ 설정
│   ├── schema.py       #    Pydantic 스키마
│   └── loader.py       #    설정 로더
│
├── cli/                # 🖥️ CLI 명령어
│   └── commands.py     #    Typer 앱
│
└── utils/              # 🔧 유틸리티
    └── helpers.py      #    헬퍼 함수
```

---

## 핵심 모듈

### 1. Agent Loop (`agent/loop.py`)

에이전트 루프는 Nanobot의 **핵심 처리 엔진**입니다.

```python
class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """
```

**처리 흐름:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Loop 처리 흐름                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. InboundMessage 수신                                     │
│         ↓                                                    │
│  2. Session에서 대화 히스토리 로드                          │
│         ↓                                                    │
│  3. ContextBuilder로 시스템 프롬프트 구성                   │
│     - SOUL.md (성격)                                         │
│     - AGENTS.md (지침)                                       │
│     - TOOLS.md (도구 문서)                                   │
│     - Skills (로드된 스킬)                                   │
│     - Memory (컨텍스트 메모리)                               │
│         ↓                                                    │
│  4. LLM 호출 (tool definitions 포함)                        │
│         ↓                                                    │
│  5. Tool Call이 있으면 실행                                 │
│     - 결과를 히스토리에 추가                                │
│     - max_iterations까지 반복                               │
│         ↓                                                    │
│  6. 최종 응답을 OutboundMessage로 전송                      │
│         ↓                                                    │
│  7. Session에 대화 저장                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Message Bus (`bus/`)

채널과 에이전트 사이의 메시지 라우팅을 담당합니다.

```python
# bus/events.py
@dataclass
class InboundMessage:
    """사용자로부터 들어오는 메시지"""
    channel: str          # "telegram", "whatsapp", "cli"
    chat_id: str          # 채팅 ID
    content: str          # 메시지 내용
    user_id: str          # 사용자 ID
    timestamp: datetime

@dataclass
class OutboundMessage:
    """에이전트가 보내는 메시지"""
    channel: str
    chat_id: str
    content: str
```

```python
# bus/queue.py
class MessageBus:
    """메시지 라우터"""

    async def publish_inbound(self, message: InboundMessage):
        """채널 → 에이전트"""

    async def publish_outbound(self, message: OutboundMessage):
        """에이전트 → 채널"""

    async def subscribe_inbound(self) -> AsyncIterator[InboundMessage]:
        """에이전트가 인바운드 구독"""

    async def subscribe_outbound(self) -> AsyncIterator[OutboundMessage]:
        """채널이 아웃바운드 구독"""
```

### 3. Channels (`channels/`)

플러그인 아키텍처로 다양한 메시징 채널을 지원합니다.

```python
# channels/base.py
class BaseChannel(ABC):
    """채널 베이스 클래스"""

    @abstractmethod
    async def start(self) -> None:
        """채널 시작"""

    @abstractmethod
    async def stop(self) -> None:
        """채널 종료"""

    @abstractmethod
    async def send(self, chat_id: str, content: str) -> None:
        """메시지 전송"""
```

```python
# channels/manager.py
class ChannelManager:
    """채널 관리자"""

    def __init__(self, config: ChannelsConfig, bus: MessageBus):
        self.channels = {}

        if config.telegram.enabled:
            self.channels["telegram"] = TelegramChannel(...)
        if config.whatsapp.enabled:
            self.channels["whatsapp"] = WhatsAppChannel(...)
        if config.feishu.enabled:
            self.channels["feishu"] = FeishuChannel(...)
```

### 4. Tools (`agent/tools/`)

에이전트가 사용할 수 있는 도구들을 제공합니다.

```python
# agent/tools/base.py
class Tool(ABC):
    """도구 베이스 클래스"""

    @property
    @abstractmethod
    def name(self) -> str:
        """도구 이름"""

    @property
    @abstractmethod
    def description(self) -> str:
        """도구 설명"""

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema 형식의 파라미터"""

    @abstractmethod
    async def execute(self, **kwargs) -> str:
        """도구 실행"""
```

```python
# agent/tools/registry.py
class ToolRegistry:
    """도구 레지스트리"""

    def register(self, tool: Tool) -> None:
        """도구 등록"""

    def get(self, name: str) -> Tool | None:
        """이름으로 도구 조회"""

    def get_definitions(self) -> list[dict]:
        """LLM용 도구 정의 반환"""
```

---

## 데이터 흐름

### 메시지 처리 흐름

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Channel │ →  │  Bus    │ →  │  Agent  │ →  │   LLM   │
│         │    │(Inbound)│    │  Loop   │    │Provider │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                   │
                                   ↓
                              ┌─────────┐
                              │  Tools  │
                              │ Execute │
                              └─────────┘
                                   │
┌─────────┐    ┌─────────┐         ↓
│ Channel │ ←  │  Bus    │ ←  ┌─────────┐
│         │    │(Outbound│    │ Response│
└─────────┘    └─────────┘    └─────────┘
```

### 컨텍스트 구성

```
┌─────────────────────────────────────────────────────────────┐
│                    Context Builder                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  System Prompt:                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. SOUL.md (성격, 가치관)                           │    │
│  │  2. AGENTS.md (에이전트 지침)                        │    │
│  │  3. TOOLS.md (도구 사용법)                           │    │
│  │  4. USER.md (사용자 정보)                            │    │
│  │  5. Skills 설명 (로드된 스킬들)                      │    │
│  │  6. Memory Context (관련 메모리)                     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Messages:                                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  1. 이전 대화 히스토리 (Session)                     │    │
│  │  2. 현재 사용자 메시지                               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  Tools:                                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Tool definitions (JSON Schema)                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 설계 원칙

### 1. 단순성 우선

```
• ~4,000줄 코드로 핵심 기능 구현
• 불필요한 추상화 최소화
• 읽기 쉬운 직관적인 코드
```

### 2. 모듈 독립성

```
• 각 모듈은 명확한 책임을 가짐
• 모듈 간 의존성 최소화
• Message Bus를 통한 느슨한 결합
```

### 3. 확장 가능성

```
• 새 채널 추가: BaseChannel 상속
• 새 도구 추가: Tool 상속
• 새 스킬 추가: skills/ 디렉토리에 폴더 생성
```

### 4. 설정 우선

```
• 모든 설정은 config.json에 집중
• 하드코딩 최소화
• 환경별 설정 분리
```

---

## 코드 통계

```
모듈별 코드 라인 수 (대략):

agent/loop.py      ~350줄   # 핵심 루프
agent/context.py   ~200줄   # 컨텍스트 빌더
agent/tools/*      ~600줄   # 도구들
channels/*         ~600줄   # 채널들
bus/*              ~150줄   # 메시지 버스
providers/*        ~200줄   # LLM 프로바이더
session/*          ~200줄   # 세션 관리
config/*           ~300줄   # 설정
cli/*              ~400줄   # CLI
기타               ~500줄   # 유틸리티 등
───────────────────────────
총계              ~4,000줄
```

---

*다음 글에서는 Agent Loop의 상세 구현을 분석합니다.*
