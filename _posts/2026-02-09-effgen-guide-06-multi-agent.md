---
layout: post
title: "effGen 완벽 가이드 (06) - 멀티에이전트 및 태스크 분해"
date: 2026-02-09
permalink: /effgen-guide-06-multi-agent/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen의 지능형 태스크 분해 엔진과 멀티에이전트 조율 시스템으로 복잡한 문제를 해결하는 방법"
---

# effGen 완벽 가이드 (06) - 멀티에이전트 및 태스크 분해

## 목차
1. [태스크 분해 개요](#태스크-분해-개요)
2. [복잡도 분석](#복잡도-분석)
3. [DecompositionEngine](#decompositionengine)
4. [멀티에이전트 조율](#멀티에이전트-조율)
5. [서브에이전트 관리](#서브에이전트-관리)
6. [병렬 실행 및 동기화](#병렬-실행-및-동기화)
7. [메모리 시스템](#메모리-시스템)
8. [실전 예제](#실전-예제)

---

## 태스크 분해 개요

effGen의 핵심 혁신 중 하나는 복잡한 태스크를 자동으로 분해하여 SLM이 처리할 수 있게 만드는 것입니다.

### 왜 태스크 분해가 필요한가?

Small Language Models는 단순한 작업은 잘 수행하지만, 복잡한 멀티스텝 작업에서는 어려움을 겪습니다.

```python
# 복잡한 태스크 예시
query = """
Research the top 3 AI agent frameworks released in 2026,
compare their key features and performance benchmarks,
implement a simple example using each framework,
test the implementations,
and create a comprehensive comparison report with visualizations.
"""

# 이것은 다음과 같은 여러 서브태스크로 구성됨:
# 1. 웹 검색 (정보 수집)
# 2. 데이터 추출 및 구조화
# 3. 비교 분석
# 4. 코드 작성 (×3)
# 5. 코드 테스트 (×3)
# 6. 데이터 시각화
# 7. 보고서 작성
```

### 태스크 분해 플로우

```
입력 쿼리
    ↓
[복잡도 분석]
    ↓
간단? ──Yes──> 직접 실행
    ↓ No
[태스크 분해]
    ↓
[의존성 그래프 생성]
    ↓
[서브에이전트 할당]
    ↓
[병렬/순차 실행]
    ↓
[결과 통합]
    ↓
최종 응답
```

### 기본 사용

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import WebSearch, Calculator, FileOps

model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator(), FileOps()],
    enable_decomposition=True,  # 태스크 분해 활성화
    max_subtasks=10,            # 최대 서브태스크 수
    system_prompt="You are a capable assistant."
))

# 복잡한 태스크 자동 분해
result = agent.run(
    "Find the GDP of the top 5 economies, calculate their total, "
    "and save the results to a formatted report"
)

# 내부적으로:
# 1. WebSearch: "top 5 economies by GDP 2026"
# 2. WebSearch: GDP data for each (병렬 실행)
# 3. Calculator: sum of GDPs
# 4. FileOps: create formatted report
```

---

## 복잡도 분석

effGen은 `ComplexityAnalyzer`를 사용하여 태스크의 복잡도를 5가지 기준으로 평가합니다.

### ComplexityAnalyzer의 5가지 기준

```python
from effgen.core.complexity import ComplexityAnalyzer

analyzer = ComplexityAnalyzer()

query = "Find the current Bitcoin price and calculate 15% of it"

complexity_score = analyzer.analyze(query)

print(complexity_score)
# {
#   "total_score": 3.5,
#   "dimensions": {
#     "step_count": 2,          # 필요한 단계 수
#     "tool_diversity": 2,      # 필요한 도구 종류 수
#     "information_density": 1, # 처리할 정보량
#     "reasoning_depth": 2,     # 추론 깊이
#     "domain_complexity": 1    # 도메인 전문성 요구
#   },
#   "is_complex": False,
#   "recommendation": "single_agent"
# }
```

### 1. Step Count (단계 수)

태스크를 완료하는 데 필요한 명확한 단계의 수입니다.

```python
# 점수 1: 단일 단계
"What is 2 + 2?"

# 점수 2: 2-3 단계
"Search for Python tutorials and summarize the top result"

# 점수 4: 4-6 단계
"Research AI frameworks, compare their features, and create a table"

# 점수 5: 7+ 단계
"Research papers, implement algorithms, run experiments, analyze results, write report"
```

### 2. Tool Diversity (도구 다양성)

사용해야 하는 서로 다른 도구의 수입니다.

```python
# 점수 1: 단일 도구
"Calculate the square root of 144"  # Calculator만

# 점수 2: 2 도구
"Search for Bitcoin price and calculate 15%"  # WebSearch + Calculator

# 점수 4: 4+ 도구
"Search papers, extract data, visualize, save report"
# WebSearch + FileOps + CodeExecutor + PythonREPL
```

### 3. Information Density (정보 밀도)

처리하고 추적해야 하는 정보의 양입니다.

```python
# 점수 1: 최소 정보
"What is the capital of France?"

# 점수 3: 중간 정보
"Compare the populations of Tokyo, New York, and London"

# 점수 5: 높은 정보
"Analyze quarterly sales data for 50 products across 10 regions"
```

### 4. Reasoning Depth (추론 깊이)

필요한 논리적 추론 및 추상화 수준입니다.

```python
# 점수 1: 직접적 답변
"What is 5 + 3?"

# 점수 2: 단순 추론
"If I have $100 and spend $35, how much is left?"

# 점수 4: 복잡한 추론
"Given sales trends, predict next quarter revenue and recommend strategies"

# 점수 5: 다단계 추상적 추론
"Analyze market dynamics, competitor strategies, and economic indicators to develop a business plan"
```

### 5. Domain Complexity (도메인 복잡도)

특수한 도메인 지식이나 전문성 요구 수준입니다.

```python
# 점수 1: 일반 지식
"What is the weather in Seoul?"

# 점수 2: 기본 전문 지식
"Explain how REST APIs work"

# 점수 4: 고급 전문 지식
"Implement a transformer attention mechanism"

# 점수 5: 최첨단 전문 지식
"Design a novel architecture for multi-agent reinforcement learning"
```

### 복잡도 임계값

```python
from effgen.core.complexity import ComplexityAnalyzer

analyzer = ComplexityAnalyzer(
    decomposition_threshold=3.0  # 이 점수 이상이면 분해
)

# 간단한 태스크 (점수: 2.0)
simple_task = "Calculate 25% of 80"
result = analyzer.analyze(simple_task)
print(result["recommendation"])  # "single_agent"

# 복잡한 태스크 (점수: 4.5)
complex_task = "Research AI papers, implement key ideas, and write comparison"
result = analyzer.analyze(complex_task)
print(result["recommendation"])  # "decompose"
```

---

## DecompositionEngine

`DecompositionEngine`은 복잡한 태스크를 서브태스크로 자동 분해합니다.

### 기본 사용

```python
from effgen.core.decomposition import DecompositionEngine
from effgen.tools.builtin import WebSearch, Calculator, FileOps

engine = DecompositionEngine(
    tools=[WebSearch(), Calculator(), FileOps()],
    max_subtasks=10,
    enable_parallel=True
)

query = "Find the top 3 programming languages by popularity and calculate their average GitHub stars"

plan = engine.decompose(query)

print(plan)
# DecompositionPlan(
#   subtasks=[
#     SubTask(
#       id="task_1",
#       description="Search for top programming languages by popularity",
#       tool="web_search",
#       parameters={"query": "top programming languages 2026"},
#       dependencies=[]
#     ),
#     SubTask(
#       id="task_2",
#       description="Extract top 3 languages from search results",
#       tool=None,  # LLM 직접 처리
#       dependencies=["task_1"]
#     ),
#     SubTask(
#       id="task_3a",
#       description="Get GitHub stars for language 1",
#       tool="web_search",
#       dependencies=["task_2"]
#     ),
#     SubTask(
#       id="task_3b",
#       description="Get GitHub stars for language 2",
#       tool="web_search",
#       dependencies=["task_2"]
#     ),
#     SubTask(
#       id="task_3c",
#       description="Get GitHub stars for language 3",
#       tool="web_search",
#       dependencies=["task_2"]
#     ),
#     SubTask(
#       id="task_4",
#       description="Calculate average stars",
#       tool="calculator",
#       dependencies=["task_3a", "task_3b", "task_3c"]
#     )
#   ],
#   execution_graph=<DAG>,
#   estimated_time=45.0  # seconds
# )
```

### 의존성 그래프

DecompositionEngine은 DAG(Directed Acyclic Graph)를 생성하여 실행 순서를 결정합니다.

```python
from effgen.core.decomposition import DecompositionEngine

engine = DecompositionEngine(tools=[...])

query = "Research topic A, then use findings to explore topic B, and finally synthesize both"

plan = engine.decompose(query)

# 의존성 그래프 시각화
print(plan.execution_graph.to_ascii())

# 출력:
#     task_1 (Research A)
#         ↓
#     task_2 (Research B, depends on A)
#         ↓
#     task_3 (Synthesize, depends on A & B)

# 병렬 실행 가능한 태스크 식별
parallel_groups = plan.get_parallel_groups()
print(parallel_groups)
# [
#   ["task_1"],           # Group 1: 병렬 실행 가능
#   ["task_2"],           # Group 2: task_1 완료 후
#   ["task_3"]            # Group 3: task_2 완료 후
# ]
```

### 고급 분해 전략

```python
from effgen.core.decomposition import DecompositionEngine, DecompositionStrategy

# 전략 1: 최소 분해 (빠른 실행)
minimal_engine = DecompositionEngine(
    tools=[...],
    strategy=DecompositionStrategy.MINIMAL,
    max_subtasks=5
)

# 전략 2: 균형 분해 (기본)
balanced_engine = DecompositionEngine(
    tools=[...],
    strategy=DecompositionStrategy.BALANCED,
    max_subtasks=10
)

# 전략 3: 최대 분해 (높은 품질)
maximal_engine = DecompositionEngine(
    tools=[...],
    strategy=DecompositionStrategy.MAXIMAL,
    max_subtasks=20,
    enable_parallel=True
)

# 전략 4: 도메인 특화
from effgen.core.decomposition import DomainDecomposer

research_engine = DomainDecomposer(
    domain="research",
    tools=[WebSearch(), Retrieval(), FileOps()],
    decomposition_templates={
        "literature_review": [
            "Search for papers on {topic}",
            "Extract key findings from each paper",
            "Identify common themes",
            "Synthesize into coherent review"
        ]
    }
)
```

### 분해 결과 검증

```python
from effgen.core.decomposition import DecompositionEngine

engine = DecompositionEngine(tools=[...])

plan = engine.decompose(query)

# 분해 품질 검증
validation = plan.validate()

print(validation)
# {
#   "is_valid": True,
#   "has_cycles": False,              # 순환 의존성 없음
#   "all_dependencies_met": True,     # 모든 의존성 해결 가능
#   "estimated_success_rate": 0.87,   # 예상 성공률
#   "warnings": [
#     "task_3 may timeout (estimated 120s)"
#   ]
# }

# 수동 조정
if not validation["is_valid"]:
    # 문제 있는 서브태스크 수정
    plan.remove_subtask("task_problematic")
    plan.add_subtask(SubTask(...))
```

---

## 멀티에이전트 조율

`Orchestrator`는 여러 전문화된 에이전트를 조율하여 복잡한 작업을 수행합니다.

### 기본 오케스트레이션

```python
from effgen import Agent, load_model
from effgen.core.orchestrator import Orchestrator
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import WebSearch, CodeExecutor, FileOps

# 전문화된 에이전트 생성
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

research_agent = Agent(config=AgentConfig(
    name="researcher",
    model=model,
    tools=[WebSearch()],
    system_prompt="You are a research specialist. Find and analyze information."
))

coding_agent = Agent(config=AgentConfig(
    name="coder",
    model=model,
    tools=[CodeExecutor()],
    system_prompt="You are a coding expert. Write and execute code."
))

writing_agent = Agent(config=AgentConfig(
    name="writer",
    model=model,
    tools=[FileOps()],
    system_prompt="You are a technical writer. Create clear documentation."
))

# 오케스트레이터 생성
orchestrator = Orchestrator(
    agents=[research_agent, coding_agent, writing_agent],
    coordination_strategy="hierarchical"  # 계층적 조율
)

# 복잡한 작업 실행
result = orchestrator.run(
    "Research transformer architectures, implement a simple attention mechanism, "
    "and write a tutorial document"
)

# 내부 실행 플로우:
# 1. research_agent: 트랜스포머 아키텍처 조사
# 2. coding_agent: 어텐션 메커니즘 구현
# 3. writing_agent: 튜토리얼 문서 작성
```

### 조율 전략

effGen은 3가지 조율 전략을 지원합니다.

#### 1. Hierarchical (계층적)

메인 에이전트가 서브에이전트에게 작업을 위임합니다.

```python
from effgen.core.orchestrator import Orchestrator, CoordinationStrategy

orchestrator = Orchestrator(
    agents=[...],
    coordination_strategy=CoordinationStrategy.HIERARCHICAL,
    main_agent=research_agent  # 메인 코디네이터
)

# 실행 플로우:
#     Main Agent
#        ↓
#   ┌────┴────┐
#   ↓    ↓    ↓
# Agent1 Agent2 Agent3
```

#### 2. Collaborative (협력적)

에이전트들이 동등하게 협력합니다.

```python
orchestrator = Orchestrator(
    agents=[...],
    coordination_strategy=CoordinationStrategy.COLLABORATIVE
)

# 실행 플로우:
# Agent1 ←→ Agent2 ←→ Agent3
#    ↓         ↓         ↓
#    └─────→ Result ←───┘
```

#### 3. Sequential (순차적)

에이전트들이 순서대로 작업을 처리합니다.

```python
orchestrator = Orchestrator(
    agents=[research_agent, coding_agent, writing_agent],
    coordination_strategy=CoordinationStrategy.SEQUENTIAL
)

# 실행 플로우:
# Input → Agent1 → Agent2 → Agent3 → Output
```

### 동적 에이전트 선택

```python
from effgen.core.orchestrator import Orchestrator

orchestrator = Orchestrator(
    agents=[research_agent, coding_agent, writing_agent, data_agent],
    enable_dynamic_selection=True  # 태스크에 따라 자동 선택
)

# 에이전트 능력 정의
research_agent.capabilities = ["web_search", "information_gathering", "analysis"]
coding_agent.capabilities = ["code_generation", "debugging", "testing"]
writing_agent.capabilities = ["documentation", "summarization"]
data_agent.capabilities = ["data_processing", "visualization", "statistics"]

# 오케스트레이터가 자동으로 적절한 에이전트 선택
result = orchestrator.run("Analyze sales data and create visualizations")
# → data_agent 자동 선택
```

---

## 서브에이전트 관리

`SubAgentManager`는 서브에이전트의 생명주기를 관리합니다.

### 기본 사용

```python
from effgen.core.subagent import SubAgentManager
from effgen import Agent

manager = SubAgentManager(
    max_subagents=5,           # 최대 동시 서브에이전트 수
    subagent_timeout=120,      # 서브에이전트 타임아웃 (초)
    enable_pooling=True        # 에이전트 풀링 활성화
)

# 서브에이전트 생성
subagent1 = manager.create_subagent(
    name="researcher_1",
    config=AgentConfig(
        model=model,
        tools=[WebSearch()],
        system_prompt="Research specialist"
    )
)

# 작업 할당
task_id = manager.assign_task(
    agent=subagent1,
    task="Search for AI agent frameworks"
)

# 결과 대기
result = manager.wait_for_completion(task_id, timeout=60)

# 서브에이전트 해제
manager.release_subagent(subagent1)
```

### 에이전트 풀링

```python
from effgen.core.subagent import SubAgentPool

# 에이전트 풀 생성
pool = SubAgentPool(
    agent_template=AgentConfig(
        model=model,
        tools=[WebSearch(), Calculator()],
        system_prompt="General assistant"
    ),
    pool_size=10,              # 풀 크기
    max_concurrent=5           # 최대 동시 실행
)

# 풀에서 에이전트 가져오기
with pool.acquire() as agent:
    result = agent.run("Some task")
    # 자동으로 풀에 반환됨

# 여러 태스크 병렬 실행
tasks = [
    "Calculate 2+2",
    "Search for Python",
    "Find weather in Seoul"
]

results = pool.map(tasks, max_workers=3)
# [4, [...search results...], [...weather info...]]
```

### 서브에이전트 모니터링

```python
from effgen.core.subagent import SubAgentManager

manager = SubAgentManager(enable_monitoring=True)

# 서브에이전트 생성 및 작업 할당
subagent = manager.create_subagent(...)
task_id = manager.assign_task(subagent, "Complex task")

# 실시간 모니터링
status = manager.get_status(task_id)
print(status)
# {
#   "task_id": "task_123",
#   "state": "running",
#   "progress": 0.45,
#   "elapsed_time": 12.3,
#   "current_step": "Executing web search",
#   "estimated_remaining": 15.2
# }

# 모든 서브에이전트 상태
all_status = manager.get_all_status()
# [
#   {"agent": "researcher_1", "state": "running", ...},
#   {"agent": "coder_1", "state": "idle", ...},
#   ...
# ]
```

---

## 병렬 실행 및 동기화

effGen은 병렬 실행과 동기화를 자동으로 관리합니다.

### 자동 병렬 실행

```python
from effgen import Agent
from effgen.core.agent import AgentConfig

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator()],
    enable_parallel_execution=True,  # 병렬 실행 활성화
    max_parallel_tasks=5             # 최대 병렬 태스크
))

# 독립적인 태스크는 자동으로 병렬 실행
result = agent.run(
    "Find the population of Tokyo, New York, London, Paris, and Berlin"
)

# 내부적으로 5개의 검색을 병렬 실행:
# WebSearch("Tokyo population") ||
# WebSearch("New York population") ||
# WebSearch("London population") ||
# WebSearch("Paris population") ||
# WebSearch("Berlin population")
```

### 수동 병렬 제어

```python
from effgen.core.parallel import ParallelExecutor
import asyncio

executor = ParallelExecutor(max_workers=4)

# 여러 태스크 정의
tasks = [
    {"agent": research_agent, "query": "Topic A"},
    {"agent": research_agent, "query": "Topic B"},
    {"agent": coding_agent, "query": "Implement X"},
    {"agent": writing_agent, "query": "Document Y"}
]

# 병렬 실행
results = executor.execute_parallel(tasks)

# 또는 async/await 사용
async def run_tasks():
    results = await asyncio.gather(
        research_agent.arun("Topic A"),
        research_agent.arun("Topic B"),
        coding_agent.arun("Implement X"),
        writing_agent.arun("Document Y")
    )
    return results

results = asyncio.run(run_tasks())
```

### 동기화 포인트

```python
from effgen.core.sync import SyncPoint

sync = SyncPoint()

# 병렬 작업 1
async def task1():
    result = await agent1.arun("Step 1")
    await sync.wait("checkpoint_1")  # 동기화 대기
    return result

# 병렬 작업 2
async def task2():
    result = await agent2.arun("Step 1")
    await sync.wait("checkpoint_1")  # 동기화 대기
    return result

# 모든 작업이 checkpoint_1에 도달할 때까지 대기
async def coordinator():
    results = await asyncio.gather(task1(), task2())
    sync.release("checkpoint_1")  # 다음 단계로 진행
    return results
```

### 배리어 패턴

```python
from effgen.core.sync import Barrier

# 3개 태스크가 모두 완료될 때까지 대기
barrier = Barrier(parties=3)

async def worker(agent, task_id):
    result = await agent.arun(f"Task {task_id}")
    await barrier.wait()  # 모든 워커 대기
    # 모두 완료되면 계속 진행
    return result

# 실행
results = await asyncio.gather(
    worker(agent1, 1),
    worker(agent2, 2),
    worker(agent3, 3)
)
```

---

## 메모리 시스템

effGen의 통합 메모리 시스템은 단기, 장기, 벡터 메모리를 제공합니다.

### UnifiedMemory

```python
from effgen.memory import UnifiedMemory
from effgen import Agent

# 메모리 시스템 생성
memory = UnifiedMemory(
    short_term_size=10,                    # 최근 10개 대화
    long_term_storage="./memory.db",       # SQLite 영구 저장
    vector_store="chromadb",               # 벡터 DB
    vector_dimension=384                   # 임베딩 차원
)

# 에이전트에 통합
agent = Agent(config=AgentConfig(
    model=model,
    tools=[...],
    memory=memory,
    enable_memory_retrieval=True
))

# 대화 히스토리 자동 관리
agent.run("My name is Alice")
# ... (여러 대화)
agent.run("What's my name?")  # "Your name is Alice"
```

### 단기 메모리 (Short-term)

최근 대화 컨텍스트를 유지합니다.

```python
from effgen.memory import ShortTermMemory

short_term = ShortTermMemory(max_size=10)

# 메시지 추가
short_term.add("user", "Hello")
short_term.add("assistant", "Hi there!")
short_term.add("user", "What's the weather?")

# 최근 대화 조회
recent = short_term.get_recent(n=5)
print(recent)
# [
#   {"role": "user", "content": "Hello"},
#   {"role": "assistant", "content": "Hi there!"},
#   ...
# ]

# 컨텍스트 윈도우 관리
context = short_term.get_context(max_tokens=2048)
# 토큰 제한에 맞게 자동 트리밍
```

### 장기 메모리 (Long-term)

영구 저장 및 검색 가능한 메모리입니다.

```python
from effgen.memory import LongTermMemory

long_term = LongTermMemory(storage_path="./memory.db")

# 중요한 정보 저장
long_term.store(
    content="User's birthday is March 15",
    metadata={
        "type": "personal_info",
        "importance": "high",
        "timestamp": "2026-02-09"
    }
)

# 나중에 검색
results = long_term.search(
    query="user birthday",
    filter={"type": "personal_info"}
)
print(results)
# [{"content": "User's birthday is March 15", ...}]

# 시간 기반 검색
recent_memories = long_term.get_recent(
    since="2026-02-01",
    limit=10
)
```

### 벡터 메모리 (Semantic)

의미 기반 검색을 위한 벡터 임베딩 메모리입니다.

```python
from effgen.memory import VectorMemory

vector_mem = VectorMemory(
    vector_store="chromadb",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    collection_name="agent_memories"
)

# 메모리 추가 (자동 임베딩)
vector_mem.add(
    content="Paris is the capital of France",
    metadata={"category": "geography"}
)

vector_mem.add(
    content="Python is a programming language",
    metadata={"category": "technology"}
)

# 의미 기반 검색
results = vector_mem.search(
    query="What is the capital city of France?",
    top_k=3
)
print(results)
# [
#   {
#     "content": "Paris is the capital of France",
#     "score": 0.92,
#     "metadata": {"category": "geography"}
#   }
# ]
```

### 메모리 통합 사용

```python
from effgen.memory import UnifiedMemory
from effgen import Agent

# 모든 메모리 타입 통합
memory = UnifiedMemory(
    short_term_size=20,
    long_term_storage="./memory.db",
    vector_store="chromadb"
)

agent = Agent(config=AgentConfig(
    model=model,
    memory=memory,
    memory_retrieval_threshold=0.7  # 관련성 임계값
))

# 세션 1
agent.run("I prefer Python over JavaScript")
agent.run("My favorite color is blue")

# 세션 2 (나중에)
agent.run("What programming language do I prefer?")
# 메모리에서 자동 검색: "You prefer Python over JavaScript"

# 세션 3
agent.run("Recommend a color for my website")
# 메모리 활용: "Since your favorite color is blue, I recommend a blue theme..."
```

---

## 실전 예제

복잡한 실제 시나리오를 통해 멀티에이전트 시스템을 구축합니다.

### 예제 1: 연구 프로젝트 자동화

```python
from effgen import Agent, load_model
from effgen.core.orchestrator import Orchestrator
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import WebSearch, Retrieval, FileOps, PythonREPL

model = load_model("Qwen/Qwen2.5-7B-Instruct", quantization="4bit")

# 전문화된 에이전트들
literature_agent = Agent(config=AgentConfig(
    name="literature_reviewer",
    model=model,
    tools=[WebSearch(), Retrieval(index_path="./papers")],
    system_prompt="""
    You are a research literature specialist.
    Find relevant papers, extract key findings, and identify research gaps.
    """
))

experiment_agent = Agent(config=AgentConfig(
    name="experimenter",
    model=model,
    tools=[PythonREPL(), FileOps()],
    system_prompt="""
    You are an experimental scientist.
    Design experiments, run simulations, and analyze results.
    """
))

writer_agent = Agent(config=AgentConfig(
    name="scientific_writer",
    model=model,
    tools=[FileOps()],
    system_prompt="""
    You are a scientific writer.
    Create well-structured research papers with clear methodology and results.
    """
))

# 오케스트레이터
orchestrator = Orchestrator(
    agents=[literature_agent, experiment_agent, writer_agent],
    coordination_strategy="sequential",
    enable_memory_sharing=True  # 에이전트 간 메모리 공유
)

# 연구 프로젝트 실행
result = orchestrator.run("""
Conduct a research project on attention mechanisms in transformers:
1. Review recent literature (2024-2026)
2. Implement a novel attention variant
3. Run comparative experiments
4. Write a research paper with results
""")

print(result)
# 완성된 연구 논문: "./research_paper.pdf"
```

### 예제 2: 소프트웨어 개발 파이프라인

```python
from effgen import Agent
from effgen.core.orchestrator import Orchestrator
from effgen.tools.builtin import CodeExecutor, FileOps, WebSearch

# 전문 에이전트들
architect_agent = Agent(config=AgentConfig(
    name="architect",
    model=model,
    tools=[FileOps(), WebSearch()],
    system_prompt="You design software architecture and create specifications."
))

developer_agent = Agent(config=AgentConfig(
    name="developer",
    model=model,
    tools=[CodeExecutor(), FileOps()],
    system_prompt="You implement code based on specifications."
))

tester_agent = Agent(config=AgentConfig(
    name="tester",
    model=model,
    tools=[CodeExecutor(), FileOps()],
    system_prompt="You write and run tests to ensure code quality."
))

reviewer_agent = Agent(config=AgentConfig(
    name="reviewer",
    model=model,
    tools=[FileOps()],
    system_prompt="You review code for bugs, security issues, and best practices."
))

# 개발 파이프라인
pipeline = Orchestrator(
    agents=[architect_agent, developer_agent, tester_agent, reviewer_agent],
    coordination_strategy="sequential"
)

# 프로젝트 실행
result = pipeline.run("""
Create a REST API for a todo list application:
1. Design the architecture and API spec
2. Implement the backend with FastAPI
3. Write comprehensive tests
4. Review and optimize the code
""")

print("Project completed!")
print("Files created:", result["files"])
print("Test coverage:", result["coverage"])
```

### 예제 3: 데이터 분석 워크플로우

```python
from effgen import Agent
from effgen.core.orchestrator import Orchestrator
from effgen.tools.builtin import FileOps, PythonREPL, WebSearch

# 데이터 파이프라인 에이전트
collector_agent = Agent(config=AgentConfig(
    name="data_collector",
    model=model,
    tools=[WebSearch(), FileOps()],
    system_prompt="You collect data from various sources."
))

cleaner_agent = Agent(config=AgentConfig(
    name="data_cleaner",
    model=model,
    tools=[PythonREPL(), FileOps()],
    system_prompt="You clean and preprocess data."
))

analyst_agent = Agent(config=AgentConfig(
    name="analyst",
    model=model,
    tools=[PythonREPL()],
    system_prompt="You perform statistical analysis and create visualizations."
))

report_agent = Agent(config=AgentConfig(
    name="reporter",
    model=model,
    tools=[FileOps()],
    system_prompt="You create comprehensive data reports."
))

# 분석 워크플로우
workflow = Orchestrator(
    agents=[collector_agent, cleaner_agent, analyst_agent, report_agent],
    coordination_strategy="sequential"
)

# 데이터 분석 프로젝트
result = workflow.run("""
Analyze e-commerce sales trends:
1. Collect sales data from the database
2. Clean and prepare the data
3. Perform trend analysis and create visualizations
4. Generate an executive summary report
""")

print("Analysis complete!")
print("Report:", result["report_path"])
print("Key insights:", result["insights"])
```

### 예제 4: 커스텀 분해 전략

```python
from effgen.core.decomposition import DecompositionEngine, SubTask

class ResearchDecomposer:
    """연구 작업 특화 분해기"""

    def __init__(self, tools):
        self.engine = DecompositionEngine(tools=tools)

    def decompose_literature_review(self, topic: str, num_papers: int = 10):
        """문헌 리뷰 태스크 분해"""
        subtasks = []

        # 1. 초기 검색
        subtasks.append(SubTask(
            id="search",
            description=f"Search for recent papers on {topic}",
            tool="web_search",
            parameters={"query": f"{topic} papers 2024-2026", "max_results": num_papers},
            dependencies=[]
        ))

        # 2. 각 논문 분석 (병렬)
        for i in range(num_papers):
            subtasks.append(SubTask(
                id=f"analyze_{i}",
                description=f"Analyze paper {i+1}",
                tool="retrieval",
                dependencies=["search"]
            ))

        # 3. 통합
        subtasks.append(SubTask(
            id="synthesize",
            description="Synthesize findings into coherent review",
            tool=None,
            dependencies=[f"analyze_{i}" for i in range(num_papers)]
        ))

        return subtasks

# 사용
decomposer = ResearchDecomposer(tools=[WebSearch(), Retrieval()])
plan = decomposer.decompose_literature_review("transformer attention mechanisms", num_papers=5)

# 실행
agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Retrieval()],
    custom_decomposer=decomposer
))

result = agent.run("Create a literature review on transformer attention mechanisms")
```

---

## 다음 단계

이제 effGen의 강력한 멀티에이전트 시스템과 태스크 분해 엔진을 이해했습니다. 다음 챕터에서는 프로덕션 배포와 고급 최적화 기법을 다룹니다.

**[다음: 챕터 07 - 고급 활용 및 프로덕션 →](/effgen-guide-07-advanced/)**

---

## 참고 자료

1. Srivastava, G. et al. (2026). "EffGen: Task Decomposition for Small Language Models". arXiv:2602.00887
2. Multi-Agent Systems: A Survey. https://arxiv.org/abs/2401.00001
3. DAG-based Task Scheduling. https://en.wikipedia.org/wiki/Directed_acyclic_graph

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/)
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/)
- [04장: 모델 및 백엔드](/effgen-guide-04-models/)
- [05장: 도구 시스템 및 프로토콜](/effgen-guide-05-tools/)
- [06장: 멀티에이전트 및 태스크 분해](/effgen-guide-06-multi-agent/) ← 현재 문서
- [07장: 고급 활용 및 프로덕션](/effgen-guide-07-advanced/)
