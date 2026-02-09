---
layout: page
title: effGen 가이드
permalink: /effgen-guide/
icon: fas fa-robot
---

# effGen 완벽 가이드

> **Small Language Models을 강력한 AI 에이전트로 변환**

**effGen**은 Small Language Models(SLM)을 강력한 자율 AI 에이전트로 변환하는 Python 프레임워크입니다. 대규모 LLM 없이도 빠르고 효율적인 에이전트를 구축할 수 있습니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개 및 개요](/blog-repo/effgen-guide-01-intro/) | effGen이란?, 주요 특징, 내장 도구 7가지 |
| 02 | [설치 및 빠른 시작](/blog-repo/effgen-guide-02-quick-start/) | 설치 방법, CLI/API 사용, 첫 에이전트 만들기 |
| 03 | [핵심 아키텍처](/blog-repo/effgen-guide-03-architecture/) | 아키텍처, 7가지 핵심 컴포넌트, 실행 플로우 |
| 04 | [모델 및 백엔드](/blog-repo/effgen-guide-04-models/) | 5가지 백엔드, 성능 비교, 모델 선택 가이드 |
| 05 | [도구 시스템 및 프로토콜](/blog-repo/effgen-guide-05-tools/) | 7가지 내장 도구, 커스텀 도구, MCP/A2A/ACP |
| 06 | [멀티에이전트 및 태스크 분해](/blog-repo/effgen-guide-06-multi-agent/) | 복잡도 분석, 조율 전략, 메모리 시스템 |
| 07 | [고급 활용 및 프로덕션](/blog-repo/effgen-guide-07-advanced/) | API 서버, 보안, 성능 튜닝, 배포 가이드 |

---

## 주요 특징

- **🧠 SLM 최적화** - Small Language Models에 특화된 설계
- **🔄 멀티모델 지원** - Transformers, vLLM, OpenAI, Anthropic, Gemini
- **🔧 도구 통합** - 7가지 내장 도구 + MCP/A2A/ACP 프로토콜
- **🧩 태스크 분해** - 복잡한 작업을 자동으로 분해
- **👥 멀티에이전트** - 서브에이전트 조율 및 병렬 실행
- **💾 메모리 시스템** - 단기/장기/벡터 메모리 지원
- **🔒 샌드박스 보안** - Docker 기반 안전한 코드 실행

---

## 빠른 시작

### 설치

```bash
# PyPI에서 설치
pip install effgen

# vLLM 백엔드 포함 (5-10x 더 빠름)
pip install effgen[vllm]
```

### 첫 에이전트 만들기

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

# Small Language Model 로드
model = load_model("Qwen/Qwen2.5-1.5B-Instruct", quantization="4bit")

# 에이전트 설정
config = AgentConfig(
    name="math_agent",
    model=model,
    tools=[Calculator()],
    system_prompt="You are a helpful math assistant."
)

# 에이전트 생성 및 실행
agent = Agent(config=config)
result = agent.run("What is 24344 * 334?")
print(f"Answer: {result.output}")
```

### CLI 사용

```bash
# 단일 작업 실행
effgen run "What is the capital of France?"

# 대화형 채팅
effgen chat

# API 서버 시작
effgen serve --port 8000
```

---

## 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                         effGen Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Agent (메인 실행 엔진)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                    │
│  ┌──────────────┬──────────────┬──────────────┬─────────────┐  │
│  │ Complexity   │ Decomposition│   Router     │ Orchestrator│  │
│  │  Analyzer    │    Engine    │  (도구선택)   │  (조율)     │  │
│  └──────────────┴──────────────┴──────────────┴─────────────┘  │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    도구 시스템                             │  │
│  │  Calculator │ WebSearch │ CodeExecutor │ PythonREPL     │  │
│  │  FileOps    │ Retrieval │ AgenticSearch                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    모델 백엔드                             │  │
│  │  Transformers │ vLLM │ OpenAI │ Anthropic │ Gemini      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 내장 도구

| 도구 | 설명 | 주요 기능 |
|------|------|----------|
| **Calculator** | 수학 계산 및 단위 변환 | 기본 연산, 고급 함수, 단위 변환 |
| **WebSearch** | DuckDuckGo 웹 검색 | 실시간 정보 검색 |
| **CodeExecutor** | Docker 샌드박스 실행 | 안전한 코드 실행 (Python, JavaScript, Bash) |
| **PythonREPL** | 대화형 Python 환경 | 상태 유지 Python 세션 |
| **FileOps** | 파일 읽기/쓰기 | 파일 시스템 작업 |
| **Retrieval** | RAG 기반 검색 | 지식 베이스 질의응답 |
| **AgenticSearch** | Grep 기반 정확한 검색 | 코드베이스 탐색 |

---

## 지원 모델

### Small Language Models (추천)

| 모델 | 파라미터 | VRAM | 특징 |
|------|---------|------|------|
| **Qwen2.5-1.5B** | 1.5B | ~2GB | 초고속, 저메모리 |
| **Qwen2.5-3B** | 3B | ~4GB | 균형잡힌 성능 |
| **Phi-3-Mini** | 3.8B | ~5GB | 추론 능력 우수 |
| **Gemma-2-2B** | 2B | ~3GB | 효율적인 구조 |

### 성능 비교

```
작업: 복잡한 수학 문제 해결 (10개 단계)

Backend        │ 처리 시간  │ 메모리 사용 │ GPU 활용률
──────────────┼───────────┼────────────┼──────────
Transformers  │ 23.4초    │ 3.2GB      │ 72%
vLLM          │ 2.1초     │ 2.8GB      │ 89%
속도 향상      │ 11.1배    │ -12.5%     │ +17%
```

---

## 프로토콜 지원

### MCP (Model Context Protocol)
- Anthropic의 표준 프로토콜
- 모델-도구 통신 표준화
- Claude, GPT 등과 호환

### A2A (Agent-to-Agent)
- 에이전트 간 직접 통신
- 분산 시스템 구축
- 메시지 큐 기반

### ACP (Agent Communication Protocol)
- 범용 에이전트 통신
- JSON-RPC 기반
- HTTP/WebSocket 지원

---

## 사용 사례

### 1. 개인 비서 에이전트
```python
from effgen import Agent, load_model
from effgen.tools.builtin import Calculator, WebSearch, FileOps

model = load_model("Qwen/Qwen2.5-3B-Instruct")
agent = Agent(
    model=model,
    tools=[Calculator(), WebSearch(), FileOps()],
    system_prompt="You are a helpful personal assistant."
)

result = agent.run("Search for Python tutorials and save the top 3 links to a file")
```

### 2. 데이터 분석 에이전트
```python
from effgen.tools.builtin import PythonREPL, FileOps

agent = Agent(
    model=model,
    tools=[PythonREPL(), FileOps()],
    system_prompt="You are a data analyst."
)

result = agent.run("Load data.csv, calculate statistics, and create a plot")
```

### 3. 연구 보조 에이전트
```python
from effgen.tools.builtin import WebSearch, Retrieval

retrieval = Retrieval(knowledge_base_path="./papers")
agent = Agent(
    model=model,
    tools=[WebSearch(), retrieval],
    system_prompt="You are a research assistant."
)

result = agent.run("Find recent papers on reinforcement learning and summarize key findings")
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| **PyTorch** | 딥러닝 프레임워크 |
| **Transformers** | 모델 로딩 및 추론 |
| **vLLM** | 고속 추론 엔진 |
| **Docker** | 샌드박스 실행 환경 |
| **FastAPI** | API 서버 |
| **Pydantic** | 데이터 검증 |
| **Rich** | CLI UI |

---

## 성능 최적화

### 모델 양자화
```python
# 4-bit 양자화 (메모리 75% 절감)
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# 8-bit 양자화 (메모리 50% 절감)
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="8bit")
```

### vLLM 백엔드
```python
# vLLM으로 5-10배 속도 향상
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    backend="vllm",
    tensor_parallel_size=2  # GPU 2개 사용
)
```

### 배치 처리
```python
# 여러 작업 동시 처리
results = agent.run_batch([
    "Calculate 123 * 456",
    "Search for weather in Tokyo",
    "Translate 'hello' to French"
])
```

---

## 보안 기능

### Docker 샌드박스
```python
executor = CodeExecutor(
    sandbox_config={
        "memory_limit": "512m",
        "cpu_quota": 50000,
        "timeout": 30,
        "network": "none"  # 네트워크 차단
    }
)
```

### 입력 검증
- 자동 입력 새니타이제이션
- SQL 인젝션 방지
- 명령어 인젝션 방지

### 속도 제한
```python
config = AgentConfig(
    rate_limit={"requests_per_minute": 60}
)
```

---

## 라이선스 및 인용

**라이선스**: MIT License

**논문 인용**:
```bibtex
@software{srivastava2026effgen,
    title={effGen: Enabling Small Language Models as Capable Autonomous Agents},
    author={Gaurav Srivastava and Aafiya Hussain and Chi Wang and Yingyan Celine Lin and Xuan Wang},
    year={2026},
    eprint={2602.00887},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2602.00887}
}
```

---

## 관련 링크

- [GitHub 저장소](https://github.com/ctrl-gaurav/effGen)
- [arXiv 논문](https://arxiv.org/abs/2602.00887)
- [공식 웹사이트](https://effgen.org/)
- [공식 문서](https://effgen.org/docs/)
- [PyPI 패키지](https://pypi.org/project/effgen/)
- [이슈 트래커](https://github.com/ctrl-gaurav/effGen/issues)

---

*작성일: 2026년 2월 9일*
*저자: Gaurav Srivastava*
