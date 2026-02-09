---
layout: post
title: "effGen 완벽 가이드 (03) - 핵심 아키텍처"
date: 2026-02-09
permalink: /effgen-guide-03-architecture/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen의 내부 아키텍처와 핵심 컴포넌트를 심층 분석하여 SLM 기반 에이전트 시스템의 작동 원리 이해"
---

# effGen 완벽 가이드 (03) - 핵심 아키텍처

## 목차
1. [전체 아키텍처 개요](#전체-아키텍처-개요)
2. [핵심 컴포넌트](#핵심-컴포넌트)
3. [모델 어댑터 시스템](#모델-어댑터-시스템)
4. [도구 시스템](#도구-시스템)
5. [실행 흐름](#실행-흐름)
6. [최적화 기법](#최적화-기법)

---

## 전체 아키텍처 개요

effGen은 모듈화된 계층 구조로 설계되어 있어 각 컴포넌트가 독립적으로 작동하면서도 긴밀하게 협력합니다.

### 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface Layer                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   CLI    │  │ Python   │  │  REST    │  │   WebUI      │   │
│  │  effgen  │  │   API    │  │   API    │  │  (planned)   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘   │
└───────┼─────────────┼─────────────┼────────────────┼───────────┘
        │             │             │                │
        └─────────────┴─────────────┴────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                    Agent Orchestration Layer                   │
│  ┌──────────────────────────▼───────────────────────────────┐ │
│  │                    Agent (core/agent.py)                  │ │
│  │  - Main execution loop                                    │ │
│  │  - Tool calling logic                                     │ │
│  │  - State management                                       │ │
│  └────────────┬──────────────────────────┬───────────────────┘ │
│               │                          │                     │
│  ┌────────────▼──────────┐  ┌───────────▼─────────────┐      │
│  │  ComplexityAnalyzer   │  │  DecompositionEngine    │      │
│  │  - Analyze task       │  │  - Break down tasks     │      │
│  │  - Route decisions    │  │  - Dependency graph     │      │
│  └───────────────────────┘  └─────────────────────────┘      │
│                                                                │
│  ┌───────────────────────┐  ┌─────────────────────────────┐  │
│  │   ExecutionTracker    │  │      Orchestrator           │  │
│  │   - Track progress    │  │      - Coordinate agents    │  │
│  │   - Logging           │  │      - Parallel execution   │  │
│  └───────────────────────┘  └─────────────────────────────┘  │
│                                                                │
│  ┌───────────────────────┐  ┌─────────────────────────────┐  │
│  │    Router             │  │   SubAgentManager           │  │
│  │    - Tool selection   │  │   - Spawn sub-agents        │  │
│  │    - Path planning    │  │   - Result aggregation      │  │
│  └───────────────────────┘  └─────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                      Model Adapter Layer                       │
│  ┌────────────────┬─────────┴────────┬──────────────────────┐ │
│  │                │                  │                      │ │
│  │  Transformers  │      vLLM        │   API Adapters       │ │
│  │  Adapter       │      Adapter     │   - OpenAI           │ │
│  │  - Local SLM   │      - Fast      │   - Anthropic        │ │
│  │  - Quantize    │      inference   │   - Gemini           │ │
│  │                │                  │                      │ │
│  └────────────────┴──────────────────┴──────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                        Tool System Layer                       │
│  ┌─────────────────────────▼───────────────────────────────┐  │
│  │                    Tool Registry                         │  │
│  │    - Tool discovery and registration                     │  │
│  └────────┬──────────────────────────┬─────────────────────┘  │
│           │                          │                        │
│  ┌────────▼──────────┐  ┌────────────▼───────────────────┐   │
│  │  Built-in Tools   │  │   Protocol Adapters            │   │
│  │  - Calculator     │  │   ┌─────────────────────────┐  │   │
│  │  - WebSearch      │  │   │  MCP (Model Context     │  │   │
│  │  - CodeExecutor   │  │   │  Protocol - Anthropic)  │  │   │
│  │  - PythonREPL     │  │   ├─────────────────────────┤  │   │
│  │  - FileOps        │  │   │  A2A (Agent-to-Agent    │  │   │
│  │  - Retrieval      │  │   │  Protocol - OpenAI)     │  │   │
│  │  - AgenticSearch  │  │   ├─────────────────────────┤  │   │
│  └───────────────────┘  │   │  ACP (Agent Comm.       │  │   │
│                         │   │  Protocol)              │  │   │
│                         │   └─────────────────────────┘  │   │
│                         └────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                      Memory & Storage Layer                    │
│  ┌──────────────────────────▼───────────────────────────────┐ │
│  │                  UnifiedMemory                            │ │
│  │  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │ │
│  │  │ Short-term  │ │  Long-term   │ │  Vector Store    │  │ │
│  │  │ (in-memory) │ │  (SQLite)    │ │  (ChromaDB/FAISS)│  │ │
│  │  │ - Recent    │ │  - Persist   │ │  - Semantic      │  │ │
│  │  │   context   │ │  - Sessions  │ │    search        │  │ │
│  │  └─────────────┘ └──────────────┘ └──────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┼─────────────────────────────────┐
│                    Execution Environment Layer                 │
│  ┌──────────────────────────▼───────────────────────────────┐ │
│  │                 Sandbox Manager                           │ │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐   │ │
│  │  │ Docker Sandbox  │  │  Security & Isolation        │   │ │
│  │  │ - Containers    │  │  - Resource limits           │   │ │
│  │  │ - Images        │  │  - Network control           │   │ │
│  │  │ - Volumes       │  │  - Input validation          │   │ │
│  │  └─────────────────┘  └──────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름

```
User Query
    │
    ├──> [ComplexityAnalyzer] ──> Complexity Score
    │                                  │
    │                                  ├──> Simple? ──> Direct Execution
    │                                  │
    │                                  └──> Complex? ──> [DecompositionEngine]
    │                                                         │
    │                                                         ├──> SubTask 1 ─┐
    │                                                         ├──> SubTask 2 ─┼──> [Orchestrator]
    │                                                         └──> SubTask 3 ─┘
    │
    ├──> [Router] ──> Tool Selection
    │                      │
    │                      ├──> Calculator
    │                      ├──> WebSearch
    │                      └──> Custom Tool
    │
    └──> [Agent] ──> Execution Loop
              │
              ├──> [Model] ──> Generate Action
              │        │
              │        └──> [PromptOptimizer] ──> Compressed Prompt
              │
              ├──> [Tool] ──> Execute
              │        │
              │        └──> [Sandbox] ──> Safe Execution
              │
              ├──> [Memory] ──> Store Context
              │
              └──> Final Result
```

---

## 핵심 컴포넌트

각 컴포넌트의 역할과 구현 세부사항입니다.

### 1. Agent (core/agent.py)

에이전트의 핵심 실행 엔진입니다.

**주요 책임**:
- 실행 루프 관리
- 도구 호출 조율
- 상태 추적
- 에러 핸들링

**클래스 구조**:

```python
class Agent:
    """Main Agent class"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = config.model
        self.tools = {tool.name: tool for tool in config.tools}

        # 컴포넌트 초기화
        self.complexity_analyzer = ComplexityAnalyzer()
        self.decomposer = DecompositionEngine()
        self.executor = ExecutionTracker()
        self.router = Router(self.tools)
        self.memory = config.memory or UnifiedMemory()

        # 서브에이전트 매니저
        if config.enable_sub_agents:
            self.sub_agent_manager = SubAgentManager(config)

        # 상태
        self.state = AgentState()

    def run(self, query: str) -> str:
        """메인 실행 메서드"""
        try:
            # 1. 복잡도 분석
            complexity = self.complexity_analyzer.analyze(query, self.tools)

            # 2. 분해 여부 결정
            if complexity.should_decompose:
                return self._run_decomposed(query, complexity)
            else:
                return self._run_direct(query)

        except Exception as e:
            return self._handle_error(e)

    def _run_direct(self, query: str) -> str:
        """직접 실행 (단순 작업)"""
        self.state.reset()

        for iteration in range(self.config.max_iterations):
            # 프롬프트 구성
            prompt = self._build_prompt(query)

            # 모델 호출
            response = self.model.generate(prompt)

            # 응답 파싱
            action = self._parse_response(response)

            if action.is_final_answer:
                return action.content

            # 도구 실행
            tool_result = self._execute_tool(action)

            # 메모리 업데이트
            self.memory.add(action, tool_result)

            # 상태 업데이트
            self.state.update(iteration, action, tool_result)

        raise MaxIterationsError()

    def _run_decomposed(self, query: str, complexity: Complexity) -> str:
        """분해된 작업 실행 (복잡한 작업)"""
        # 태스크 분해
        subtasks = self.decomposer.decompose(query, complexity)

        # 실행 계획 생성
        plan = ExecutionPlan(subtasks)

        # 오케스트레이터로 실행
        results = self.orchestrator.execute(plan)

        # 결과 통합
        return self._aggregate_results(results)

    def _execute_tool(self, action: Action) -> ToolResult:
        """도구 실행"""
        tool = self.router.select_tool(action)

        if tool.requires_sandbox:
            return self._execute_sandboxed(tool, action.input)
        else:
            return tool.run(action.input)

    def _build_prompt(self, query: str) -> str:
        """프롬프트 구성"""
        # 시스템 프롬프트
        system = self.config.system_prompt

        # 도구 설명
        tool_desc = self._get_tool_descriptions()

        # 대화 히스토리
        history = self.memory.get_recent(n=5)

        # 프롬프트 최적화 (SLM용)
        optimizer = PromptOptimizer()
        optimized = optimizer.compress(
            system=system,
            tools=tool_desc,
            history=history,
            query=query
        )

        return optimized
```

**실행 루프 상세**:

```python
class ExecutionLoop:
    """에이전트 실행 루프"""

    def __init__(self, agent: Agent):
        self.agent = agent

    def run(self, query: str) -> str:
        """
        ReAct 패턴 구현:
        Thought -> Action -> Observation -> Repeat
        """
        state = {
            "query": query,
            "thoughts": [],
            "actions": [],
            "observations": []
        }

        while not self._is_complete(state):
            # Step 1: Think
            thought = self._generate_thought(state)
            state["thoughts"].append(thought)

            # Step 2: Act
            action = self._decide_action(thought, state)
            state["actions"].append(action)

            # 종료 조건 확인
            if action.type == ActionType.FINAL_ANSWER:
                return action.content

            # Step 3: Observe
            observation = self._execute_action(action)
            state["observations"].append(observation)

            # Step 4: Update
            self._update_state(state, observation)

        return self._extract_answer(state)

    def _generate_thought(self, state: dict) -> str:
        """사고 과정 생성"""
        prompt = f"""
Given the query: {state['query']}

Previous actions: {state['actions'][-3:]}
Previous observations: {state['observations'][-3:]}

What should I do next? Think step by step.
"""
        return self.agent.model.generate(prompt)

    def _decide_action(self, thought: str, state: dict) -> Action:
        """행동 결정"""
        available_tools = self.agent.router.get_available_tools(state)

        prompt = f"""
Thought: {thought}

Available tools:
{self._format_tools(available_tools)}

Decide the next action in this format:
Action: [tool name]
Action Input: [input to the tool]

Or if you have the final answer:
Final Answer: [answer]
"""
        response = self.agent.model.generate(prompt)
        return self._parse_action(response)
```

### 2. ComplexityAnalyzer

작업의 복잡도를 분석하여 실행 전략을 결정합니다.

**분석 요소**:

```python
class ComplexityAnalyzer:
    """작업 복잡도 분석기"""

    def analyze(self, query: str, tools: List[Tool]) -> Complexity:
        """
        5가지 요소로 복잡도 분석:
        1. 쿼리 길이 및 구조
        2. 필요한 도구 수
        3. 단계 수 추정
        4. 도메인 난이도
        5. 의존성 복잡도
        """
        scores = {
            "length": self._analyze_length(query),
            "tools": self._analyze_tools(query, tools),
            "steps": self._analyze_steps(query),
            "domain": self._analyze_domain(query),
            "dependency": self._analyze_dependencies(query)
        }

        # 가중 평균
        total_score = (
            scores["length"] * 0.15 +
            scores["tools"] * 0.25 +
            scores["steps"] * 0.30 +
            scores["domain"] * 0.15 +
            scores["dependency"] * 0.15
        )

        return Complexity(
            score=total_score,
            breakdown=scores,
            should_decompose=total_score > 0.7,
            estimated_steps=scores["steps"]
        )

    def _analyze_length(self, query: str) -> float:
        """쿼리 길이 분석"""
        words = query.split()
        # 간단: < 10 단어, 복잡: > 50 단어
        if len(words) < 10:
            return 0.2
        elif len(words) < 30:
            return 0.5
        else:
            return 0.9

    def _analyze_tools(self, query: str, tools: List[Tool]) -> float:
        """필요한 도구 수 분석"""
        # 각 도구의 트리거 키워드 확인
        needed_tools = []

        for tool in tools:
            if tool.matches(query):
                needed_tools.append(tool)

        # 0개: 0.0, 1개: 0.3, 2개: 0.6, 3+개: 1.0
        tool_count = len(needed_tools)
        if tool_count == 0:
            return 0.0
        elif tool_count == 1:
            return 0.3
        elif tool_count == 2:
            return 0.6
        else:
            return 1.0

    def _analyze_steps(self, query: str) -> float:
        """필요한 단계 수 추정"""
        # 키워드 기반 단계 분석
        sequential_keywords = [
            "then", "after", "next", "finally",
            "first", "second", "third"
        ]
        parallel_keywords = [
            "and", "also", "both", "all"
        ]

        query_lower = query.lower()

        sequential_count = sum(
            1 for kw in sequential_keywords
            if kw in query_lower
        )
        parallel_count = sum(
            1 for kw in parallel_keywords
            if kw in query_lower
        )

        estimated_steps = 1 + sequential_count + (parallel_count * 0.5)

        # 1 step: 0.1, 2-3 steps: 0.5, 4+ steps: 0.9
        if estimated_steps <= 1:
            return 0.1
        elif estimated_steps <= 3:
            return 0.5
        else:
            return 0.9

    def _analyze_domain(self, query: str) -> float:
        """도메인 난이도 분석"""
        # 도메인별 난이도 매핑
        domain_difficulty = {
            "math": 0.3,
            "search": 0.2,
            "coding": 0.7,
            "analysis": 0.8,
            "research": 0.9,
            "general": 0.4
        }

        detected_domain = self._detect_domain(query)
        return domain_difficulty.get(detected_domain, 0.5)

    def _analyze_dependencies(self, query: str) -> float:
        """의존성 복잡도 분석"""
        # "use the result of X to do Y" 패턴 감지
        dependency_patterns = [
            r"use .+ to",
            r"based on .+ calculate",
            r"with the result",
            r"from .+ find"
        ]

        import re
        dependency_count = sum(
            1 for pattern in dependency_patterns
            if re.search(pattern, query.lower())
        )

        return min(dependency_count * 0.3, 1.0)
```

**복잡도 기반 라우팅**:

```python
class ComplexityRouter:
    """복잡도에 따른 실행 경로 결정"""

    def route(self, complexity: Complexity) -> ExecutionStrategy:
        """
        복잡도에 따른 전략 선택:
        - Simple (< 0.3): Direct execution
        - Medium (0.3-0.7): Optimized execution
        - Complex (> 0.7): Decomposition + multi-agent
        """
        if complexity.score < 0.3:
            return DirectStrategy()
        elif complexity.score < 0.7:
            return OptimizedStrategy(
                enable_caching=True,
                enable_parallel_tools=True
            )
        else:
            return DecompositionStrategy(
                max_subtasks=5,
                enable_sub_agents=True,
                parallel_execution=True
            )
```

### 3. DecompositionEngine

복잡한 작업을 서브태스크로 분해합니다.

**분해 알고리즘**:

```python
class DecompositionEngine:
    """태스크 분해 엔진"""

    def __init__(self, model):
        self.model = model
        self.dependency_analyzer = DependencyAnalyzer()

    def decompose(self, query: str, complexity: Complexity) -> List[SubTask]:
        """
        작업 분해 프로세스:
        1. 논리적 단계 식별
        2. 의존성 분석
        3. 병렬 가능성 판단
        4. 서브태스크 생성
        """
        # Step 1: 모델을 사용하여 단계 식별
        steps = self._identify_steps(query)

        # Step 2: 의존성 그래프 구축
        dependency_graph = self.dependency_analyzer.build_graph(steps)

        # Step 3: 토폴로지 정렬로 실행 순서 결정
        execution_order = self._topological_sort(dependency_graph)

        # Step 4: 서브태스크 객체 생성
        subtasks = []
        for step_id in execution_order:
            step = steps[step_id]
            subtask = SubTask(
                id=step_id,
                description=step.description,
                dependencies=[
                    dep_id for dep_id in dependency_graph[step_id]
                ],
                tools=self._select_tools_for_step(step),
                parallel_group=self._assign_parallel_group(
                    step_id, dependency_graph
                )
            )
            subtasks.append(subtask)

        return subtasks

    def _identify_steps(self, query: str) -> Dict[int, Step]:
        """모델을 사용하여 논리적 단계 식별"""
        prompt = f"""
Break down this complex task into logical steps:

Task: {query}

Identify each step in this format:
Step 1: [description]
Step 2: [description]
...

Be specific about what each step accomplishes.
"""
        response = self.model.generate(prompt)
        return self._parse_steps(response)

    def _parse_steps(self, response: str) -> Dict[int, Step]:
        """모델 응답에서 단계 파싱"""
        import re

        steps = {}
        pattern = r"Step (\d+): (.+)"

        for match in re.finditer(pattern, response):
            step_id = int(match.group(1))
            description = match.group(2).strip()

            steps[step_id] = Step(
                id=step_id,
                description=description,
                keywords=self._extract_keywords(description)
            )

        return steps

    def _topological_sort(
        self,
        graph: Dict[int, List[int]]
    ) -> List[int]:
        """토폴로지 정렬로 실행 순서 결정"""
        from collections import deque

        # 진입 차수 계산
        in_degree = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1

        # 큐 초기화 (진입 차수 0인 노드들)
        queue = deque([
            node for node in in_degree
            if in_degree[node] == 0
        ])

        sorted_order = []

        while queue:
            node = queue.popleft()
            sorted_order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # 사이클 감지
        if len(sorted_order) != len(graph):
            raise CyclicDependencyError()

        return sorted_order

    def _assign_parallel_group(
        self,
        step_id: int,
        graph: Dict[int, List[int]]
    ) -> int:
        """병렬 실행 그룹 할당"""
        # 같은 깊이 레벨의 독립적인 태스크는 같은 그룹
        # BFS로 레벨 계산
        levels = {}
        queue = deque([(node, 0) for node in graph if not graph[node]])

        while queue:
            node, level = queue.popleft()
            levels[node] = level

            for neighbor in graph:
                if node in graph[neighbor]:
                    queue.append((neighbor, level + 1))

        return levels.get(step_id, 0)
```

**의존성 분석**:

```python
class DependencyAnalyzer:
    """태스크 간 의존성 분석"""

    def build_graph(self, steps: Dict[int, Step]) -> Dict[int, List[int]]:
        """의존성 그래프 구축"""
        graph = {step_id: [] for step_id in steps}

        for step_id, step in steps.items():
            # 다른 단계들과의 의존성 확인
            for other_id, other_step in steps.items():
                if other_id >= step_id:
                    continue  # 이후 단계만 의존할 수 있음

                if self._has_dependency(step, other_step):
                    graph[other_id].append(step_id)

        return graph

    def _has_dependency(self, step: Step, potential_dep: Step) -> bool:
        """두 단계 간 의존성 여부 확인"""
        # 휴리스틱:
        # 1. 결과 참조 ("use the result", "based on")
        # 2. 데이터 흐름 (변수명, 엔티티 참조)
        # 3. 순차적 키워드 ("then", "after")

        step_desc = step.description.lower()

        # 명시적 참조
        if f"step {potential_dep.id}" in step_desc:
            return True

        # 데이터 흐름 분석
        potential_dep_outputs = potential_dep.keywords
        for output in potential_dep_outputs:
            if output.lower() in step_desc:
                return True

        # 순차 키워드
        sequential_markers = ["then", "after", "using", "with"]
        if any(marker in step_desc for marker in sequential_markers):
            # 간단한 휴리스틱: 이전 단계에 의존
            return True

        return False
```

### 4. ExecutionTracker

실행 과정을 추적하고 로깅합니다.

```python
class ExecutionTracker:
    """실행 추적 및 모니터링"""

    def __init__(self):
        self.executions = []
        self.current_execution = None

    def start(self, query: str):
        """실행 시작"""
        self.current_execution = Execution(
            id=str(uuid.uuid4()),
            query=query,
            start_time=time.time(),
            steps=[]
        )

    def log_step(
        self,
        step_type: str,
        content: str,
        metadata: dict = None
    ):
        """단계 로깅"""
        if not self.current_execution:
            raise NoActiveExecutionError()

        step = ExecutionStep(
            type=step_type,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        )

        self.current_execution.steps.append(step)

    def finish(self, result: str):
        """실행 종료"""
        if not self.current_execution:
            return

        self.current_execution.end_time = time.time()
        self.current_execution.result = result
        self.current_execution.status = "success"

        self.executions.append(self.current_execution)
        self.current_execution = None

    def get_statistics(self) -> dict:
        """실행 통계"""
        return {
            "total_executions": len(self.executions),
            "average_duration": np.mean([
                e.end_time - e.start_time
                for e in self.executions
            ]),
            "success_rate": sum(
                1 for e in self.executions
                if e.status == "success"
            ) / len(self.executions),
            "tool_usage": self._analyze_tool_usage()
        }

    def _analyze_tool_usage(self) -> Dict[str, int]:
        """도구 사용 빈도 분석"""
        tool_counts = {}

        for execution in self.executions:
            for step in execution.steps:
                if step.type == "tool_call":
                    tool_name = step.metadata.get("tool_name")
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return tool_counts
```

### 5. Router

적절한 도구를 선택하고 라우팅합니다.

```python
class Router:
    """도구 라우팅"""

    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.matcher = ToolMatcher()

    def select_tool(self, action: Action) -> Tool:
        """행동에 맞는 도구 선택"""
        tool_name = action.tool_name

        if tool_name in self.tools:
            return self.tools[tool_name]

        # 퍼지 매칭
        matched_tool = self.matcher.fuzzy_match(
            tool_name,
            list(self.tools.keys())
        )

        if matched_tool:
            return self.tools[matched_tool]

        raise ToolNotFoundError(f"Tool '{tool_name}' not found")

    def get_tool_for_query(self, query: str) -> List[Tool]:
        """쿼리에 적합한 도구들 추천"""
        recommendations = []

        for tool_name, tool in self.tools.items():
            relevance = self._calculate_relevance(query, tool)
            if relevance > 0.5:
                recommendations.append((tool, relevance))

        # 관련도 순 정렬
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return [tool for tool, _ in recommendations]

    def _calculate_relevance(self, query: str, tool: Tool) -> float:
        """쿼리와 도구의 관련도 계산"""
        # 키워드 매칭
        query_lower = query.lower()
        keyword_matches = sum(
            1 for keyword in tool.keywords
            if keyword.lower() in query_lower
        )

        # 예제 매칭
        example_similarity = max(
            (self._similarity(query, example)
             for example in tool.examples),
            default=0.0
        )

        # 가중 평균
        relevance = (
            keyword_matches * 0.4 +
            example_similarity * 0.6
        )

        return min(relevance, 1.0)
```

### 6. Orchestrator

여러 에이전트와 서브태스크를 조율합니다.

```python
class Orchestrator:
    """멀티에이전트 조율"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.sub_agent_manager = SubAgentManager(config)
        self.thread_pool = ThreadPoolExecutor(max_workers=config.max_sub_agents)

    def execute(self, plan: ExecutionPlan) -> Dict[int, Any]:
        """실행 계획 실행"""
        results = {}

        # 병렬 그룹별로 실행
        for group_id in sorted(set(
            task.parallel_group for task in plan.tasks
        )):
            group_tasks = [
                task for task in plan.tasks
                if task.parallel_group == group_id
            ]

            # 같은 그룹은 병렬 실행
            group_results = self._execute_parallel(group_tasks, results)
            results.update(group_results)

        return results

    def _execute_parallel(
        self,
        tasks: List[SubTask],
        previous_results: Dict[int, Any]
    ) -> Dict[int, Any]:
        """병렬 실행"""
        futures = {}

        for task in tasks:
            # 의존성 결과 수집
            dep_results = {
                dep_id: previous_results[dep_id]
                for dep_id in task.dependencies
            }

            # 서브에이전트 생성 및 실행
            future = self.thread_pool.submit(
                self._execute_task,
                task,
                dep_results
            )
            futures[task.id] = future

        # 결과 수집
        results = {}
        for task_id, future in futures.items():
            results[task_id] = future.result()

        return results

    def _execute_task(
        self,
        task: SubTask,
        dependencies: Dict[int, Any]
    ) -> Any:
        """단일 태스크 실행"""
        # 서브에이전트 생성
        sub_agent = self.sub_agent_manager.create_agent(
            task=task,
            dependencies=dependencies
        )

        # 실행
        result = sub_agent.run(task.description)

        return result
```

### 7. SubAgentManager

서브에이전트를 생성하고 관리합니다.

```python
class SubAgentManager:
    """서브에이전트 관리"""

    def __init__(self, parent_config: AgentConfig):
        self.parent_config = parent_config
        self.active_agents = {}

    def create_agent(
        self,
        task: SubTask,
        dependencies: Dict[int, Any]
    ) -> Agent:
        """태스크 전용 서브에이전트 생성"""
        # 부모 설정 상속
        sub_config = AgentConfig(
            name=f"sub_agent_{task.id}",
            model=self.parent_config.model,
            tools=task.tools,  # 태스크별 도구만
            system_prompt=self._build_sub_prompt(task, dependencies),
            max_iterations=5,  # 서브에이전트는 짧게
            verbose=False
        )

        agent = Agent(config=sub_config)
        self.active_agents[task.id] = agent

        return agent

    def _build_sub_prompt(
        self,
        task: SubTask,
        dependencies: Dict[int, Any]
    ) -> str:
        """서브에이전트용 프롬프트 구성"""
        prompt = f"""You are a specialized sub-agent for this specific task:

Task: {task.description}

"""
        if dependencies:
            prompt += "You have access to results from previous steps:\n"
            for dep_id, result in dependencies.items():
                prompt += f"- Step {dep_id}: {result}\n"

            prompt += "\nUse these results to complete your task.\n"

        prompt += "\nFocus only on this specific task. Be concise and precise."

        return prompt

    def aggregate_results(
        self,
        results: Dict[int, Any]
    ) -> str:
        """서브에이전트 결과 통합"""
        # 최종 결과 조합
        aggregation_prompt = f"""
Combine these sub-task results into a coherent final answer:

{self._format_results(results)}

Provide a comprehensive answer that addresses the original query.
"""
        # 부모 모델로 통합
        final_result = self.parent_config.model.generate(aggregation_prompt)

        return final_result
```

---

## 모델 어댑터 시스템

다양한 모델 백엔드를 지원하는 어댑터 패턴입니다.

### BaseModel 인터페이스

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """모든 모델 어댑터의 베이스 클래스"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """텍스트 생성"""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        **kwargs
    ) -> dict:
        """구조화된 출력 생성"""
        pass

    @abstractmethod
    def get_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """텍스트 임베딩"""
        pass
```

### 1. TransformersAdapter

Hugging Face Transformers 백엔드:

```python
class TransformersAdapter(BaseModel):
    """Hugging Face Transformers 어댑터"""

    def __init__(
        self,
        model_name: str,
        quantization: str = None,
        device: str = "auto"
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # 양자화 설정
        quant_config = self._get_quant_config(quantization)

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 생성 설정
        self.generation_config = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        # 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **{**self.generation_config, **kwargs}
            )

        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def _get_quant_config(self, quantization: str):
        """양자화 설정"""
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            return None
```

### 2. vLLMAdapter

vLLM 고속 추론 백엔드:

```python
class vLLMAdapter(BaseModel):
    """vLLM 어댑터 (고속 추론)"""

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        **kwargs
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            **kwargs
        )

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        # vLLM은 배치 처리에 최적화
        outputs = self.llm.generate(
            [prompt],
            self.sampling_params
        )

        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """배치 생성 (vLLM의 강점)"""
        outputs = self.llm.generate(
            prompts,
            self.sampling_params
        )

        return [output.outputs[0].text for output in outputs]
```

### 3. OpenAIAdapter

OpenAI API 백엔드:

```python
class OpenAIAdapter(BaseModel):
    """OpenAI API 어댑터"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: str = None
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        return response.choices[0].message.content

    def generate_structured(
        self,
        prompt: str,
        schema: dict,
        **kwargs
    ) -> dict:
        """구조화된 출력 (Structured Outputs)"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema
            },
            **kwargs
        )

        import json
        return json.loads(response.choices[0].message.content)
```

### 4. AnthropicAdapter

Anthropic Claude API 백엔드:

```python
class AnthropicAdapter(BaseModel):
    """Anthropic Claude API 어댑터"""

    def __init__(
        self,
        model_name: str = "claude-3-haiku-20240307",
        api_key: str = None
    ):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        return message.content[0].text
```

### 5. GeminiAdapter

Google Gemini API 백엔드:

```python
class GeminiAdapter(BaseModel):
    """Google Gemini API 어댑터"""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str = None
    ):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, **kwargs) -> str:
        """텍스트 생성"""
        response = self.model.generate_content(prompt)
        return response.text
```

---

## 도구 시스템

도구 시스템은 Built-in Tools와 Protocol Adapters로 구성됩니다.

### BaseTool 인터페이스

```python
class BaseTool(ABC):
    """모든 도구의 베이스 클래스"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.description = ""
        self.parameters = {}
        self.keywords = []
        self.examples = []

    @abstractmethod
    def run(self, input: str) -> Any:
        """도구 실행"""
        pass

    def matches(self, query: str) -> bool:
        """쿼리가 이 도구와 관련있는지"""
        query_lower = query.lower()
        return any(
            keyword.lower() in query_lower
            for keyword in self.keywords
        )

    def get_schema(self) -> dict:
        """도구 스키마 (JSON Schema)"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }
```

### Protocol Adapters

#### MCP (Model Context Protocol)

Anthropic의 Model Context Protocol 지원:

```python
class MCPTool(BaseTool):
    """MCP 프로토콜 도구"""

    @classmethod
    def from_server(cls, server_url: str) -> List[BaseTool]:
        """MCP 서버에서 도구 로드"""
        import requests

        # MCP 서버에서 도구 목록 가져오기
        response = requests.get(f"{server_url}/tools")
        tools_schema = response.json()

        # 각 도구를 effGen Tool로 변환
        tools = []
        for tool_def in tools_schema:
            tool = cls._create_tool_from_schema(tool_def, server_url)
            tools.append(tool)

        return tools

    @classmethod
    def _create_tool_from_schema(
        cls,
        schema: dict,
        server_url: str
    ) -> BaseTool:
        """스키마에서 도구 생성"""
        class DynamicMCPTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.name = schema["name"]
                self.description = schema["description"]
                self.parameters = schema["parameters"]
                self.server_url = server_url

            def run(self, input: str) -> Any:
                # MCP 서버로 요청
                import requests
                response = requests.post(
                    f"{self.server_url}/execute/{self.name}",
                    json={"input": input}
                )
                return response.json()["result"]

        return DynamicMCPTool()
```

#### A2A (Agent-to-Agent Protocol)

OpenAI의 Agent-to-Agent Protocol 지원:

```python
class A2ATool(BaseTool):
    """A2A 프로토콜 도구"""

    @classmethod
    def from_endpoint(cls, endpoint_url: str) -> List[BaseTool]:
        """A2A 엔드포인트에서 도구 로드"""
        # OpenAI Agent Protocol 구현
        # ... (유사한 패턴)
        pass
```

---

## 실행 흐름

전체 실행 흐름을 순서도로 정리합니다.

```
User Query
    │
    ▼
┌─────────────────────┐
│ Agent.run(query)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ ComplexityAnalyzer  │
│  .analyze()         │
└──────────┬──────────┘
           │
           ├─── complexity < 0.7 ───┐
           │                        │
           ▼                        ▼
    ┌─────────────┐        ┌───────────────┐
    │ Decompose   │        │ Direct Exec   │
    └──────┬──────┘        └───────┬───────┘
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │ SubTasks    │                │
    │ [1,2,3...]  │                │
    └──────┬──────┘                │
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │Orchestrator │                │
    │ .execute()  │                │
    └──────┬──────┘                │
           │                       │
           ├─ Parallel Group 0     │
           ├─ Parallel Group 1     │
           └─ ...                  │
                                   │
           ┌───────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Execution   │
    │ Loop        │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Iteration 1 │
    └──────┬──────┘
           │
           ├─ Thought ──────────┐
           │                    │
           ├─ Action ────────┐  │
           │                 │  │
           ├─ Tool Exec ──┐  │  │
           │              │  │  │
           │              ▼  ▼  ▼
           │         ┌──────────────┐
           │         │PromptBuilder │
           │         └──────┬───────┘
           │                │
           │                ▼
           │         ┌──────────────┐
           │         │Prompt        │
           │         │Optimizer     │
           │         └──────┬───────┘
           │                │
           │                ▼
           │         ┌──────────────┐
           │         │Model         │
           │         │.generate()   │
           │         └──────┬───────┘
           │                │
           │                ▼
           │         ┌──────────────┐
           │         │Parse Response│
           │         └──────┬───────┘
           │                │
           │         ┌──────┴───────┐
           │         │              │
           │         ▼              ▼
           │    ┌────────┐    ┌─────────┐
           │    │ Tool   │    │ Final   │
           │    │ Call   │    │ Answer  │
           │    └───┬────┘    └────┬────┘
           │        │              │
           │        ▼              │
           │    ┌────────┐         │
           │    │Router  │         │
           │    └───┬────┘         │
           │        │              │
           │        ▼              │
           │    ┌────────┐         │
           │    │Execute │         │
           │    │Tool    │         │
           │    └───┬────┘         │
           │        │              │
           │        ▼              │
           │    ┌────────┐         │
           │    │Result  │         │
           │    └───┬────┘         │
           │        │              │
           ├────────┘              │
           │                       │
           ▼                       │
    ┌─────────────┐                │
    │ Iteration 2 │                │
    │    ...      │                │
    └──────┬──────┘                │
           │                       │
           ├───────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ Final Result│
    └──────┬──────┘
           │
           ▼
        Return
```

---

## 최적화 기법

effGen이 SLM에서 성능을 끌어내기 위해 사용하는 최적화 기법들입니다.

### 1. 프롬프트 압축

**목표**: 컨텍스트를 70-80% 압축하면서 의미 보존

```python
class PromptOptimizer:
    """SLM용 프롬프트 최적화"""

    def compress(
        self,
        system: str,
        tools: str,
        history: List[dict],
        query: str
    ) -> str:
        """프롬프트 압축"""
        # 1. 시스템 프롬프트 간소화
        compressed_system = self._simplify_system(system)

        # 2. 도구 설명 압축
        compressed_tools = self._compress_tools(tools)

        # 3. 히스토리 요약
        compressed_history = self._summarize_history(history)

        # 4. 쿼리는 그대로
        compressed_query = query

        # 조합
        prompt = f"""{compressed_system}

Tools: {compressed_tools}

{compressed_history}

User: {compressed_query}
Assistant:"""

        return prompt

    def _simplify_system(self, system: str) -> str:
        """시스템 프롬프트 간소화"""
        # 핵심 지침만 유지
        essential_parts = []

        # "You are X" 패턴 유지
        if "You are" in system:
            role = self._extract_role(system)
            essential_parts.append(f"You are {role}.")

        # 필수 규칙만 추출
        rules = self._extract_essential_rules(system)
        essential_parts.extend(rules)

        return " ".join(essential_parts)

    def _compress_tools(self, tools_desc: str) -> str:
        """도구 설명 압축"""
        # 긴 설명 대신 간결한 형식
        # Before:
        # Calculator - A tool for performing mathematical calculations.
        #   It can handle basic arithmetic, advanced functions, ...
        #
        # After:
        # Calculator: math operations

        compressed = []
        for tool in self._parse_tools(tools_desc):
            compressed.append(
                f"{tool.name}: {tool.short_desc}"
            )

        return ", ".join(compressed)

    def _summarize_history(self, history: List[dict]) -> str:
        """대화 히스토리 요약"""
        if len(history) <= 2:
            # 짧으면 그대로
            return self._format_history(history)

        # 긴 경우 최근 N개 + 요약
        recent = history[-2:]
        older = history[:-2]

        summary = self._create_summary(older)

        return f"[Previous: {summary}]\n{self._format_history(recent)}"
```

### 2. 토큰 예산 관리

```python
class TokenBudgetManager:
    """토큰 예산 관리"""

    def __init__(self, max_tokens: int = 2048):
        self.max_tokens = max_tokens
        self.allocations = {
            "system": 0.15,    # 15%
            "tools": 0.20,     # 20%
            "history": 0.30,   # 30%
            "query": 0.15,     # 15%
            "generation": 0.20 # 20%
        }

    def allocate(self, components: dict) -> dict:
        """각 컴포넌트에 토큰 할당"""
        allocated = {}

        for key, ratio in self.allocations.items():
            if key == "generation":
                continue  # 생성은 남은 것 사용

            budget = int(self.max_tokens * ratio)
            content = components.get(key, "")

            # 예산 내로 자르기
            allocated[key] = self._truncate_to_budget(
                content,
                budget
            )

        return allocated

    def _truncate_to_budget(
        self,
        text: str,
        budget: int
    ) -> str:
        """예산 내로 텍스트 자르기"""
        tokens = self._tokenize(text)

        if len(tokens) <= budget:
            return text

        # 뒤에서부터 자르기 (최근 정보 유지)
        truncated_tokens = tokens[-budget:]
        return self._detokenize(truncated_tokens)
```

### 3. 캐싱 전략

```python
class CacheManager:
    """중간 결과 캐싱"""

    def __init__(self):
        self.tool_cache = LRUCache(maxsize=100)
        self.embedding_cache = LRUCache(maxsize=1000)

    def get_tool_result(
        self,
        tool_name: str,
        input_hash: str
    ) -> Optional[Any]:
        """캐시된 도구 결과 가져오기"""
        cache_key = f"{tool_name}:{input_hash}"
        return self.tool_cache.get(cache_key)

    def cache_tool_result(
        self,
        tool_name: str,
        input_hash: str,
        result: Any
    ):
        """도구 결과 캐시"""
        cache_key = f"{tool_name}:{input_hash}"
        self.tool_cache.put(cache_key, result)
```

---

## 다음 단계

이제 effGen의 내부 아키텍처를 깊이 이해했습니다. 다음 챕터에서는 실전 프로젝트를 통해 effGen을 활용하는 방법을 배우겠습니다.

**[다음: 챕터 04 - 고급 기능 및 커스터마이징 (예정)](/effgen-guide-04-advanced/)**

---

## 참고 자료

1. Srivastava, G. et al. (2026). "EffGen: Small Language Models as Autonomous Agents". arXiv:2602.00887
2. ReAct Pattern: Yao et al. (2022). "ReAct: Synergizing Reasoning and Acting in Language Models"
3. Tool Use: Schick et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools"
4. vLLM Architecture: https://docs.vllm.ai/en/latest/architecture.html

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/)
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/) ← 현재 문서