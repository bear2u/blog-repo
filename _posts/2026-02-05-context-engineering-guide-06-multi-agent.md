---
layout: post
title: "Context Engineering 완벽 가이드 (6) - Multi-Agent Systems"
date: 2026-02-05
permalink: /context-engineering-guide-06-multi-agent/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, Multi-Agent, Orchestration, Coordination, Collaboration]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "멀티에이전트 시스템의 통신, 조정, 협업 메커니즘을 알아봅니다."
---

## Multi-Agent Systems 개요

복잡한 작업을 해결하기 위해 여러 전문화된 에이전트가 협업합니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────────┐                          │
│                    │ Orchestrator│                          │
│                    │   Agent     │                          │
│                    └──────┬──────┘                          │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           │               │               │                 │
│           ▼               ▼               ▼                 │
│    ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│    │  Research  │  │   Coder    │  │  Reviewer  │          │
│    │   Agent    │  │   Agent    │  │   Agent    │          │
│    └────────────┘  └────────────┘  └────────────┘          │
│           │               │               │                 │
│           └───────────────┼───────────────┘                 │
│                           │                                  │
│                    ┌──────▼──────┐                          │
│                    │   Shared    │                          │
│                    │   Memory    │                          │
│                    └─────────────┘                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 통신 프로토콜

### 메시지 기반 통신

```python
@dataclass
class AgentMessage:
    """에이전트 간 메시지"""
    sender: str
    recipient: str
    type: MessageType  # REQUEST, RESPONSE, BROADCAST, ACK
    content: dict
    correlation_id: str
    timestamp: float

class MessageType(Enum):
    REQUEST = "request"      # 작업 요청
    RESPONSE = "response"    # 작업 응답
    BROADCAST = "broadcast"  # 전체 공지
    ACK = "ack"              # 수신 확인
    STATUS = "status"        # 상태 업데이트
    DELEGATE = "delegate"    # 위임

class MessageBus:
    """에이전트 간 메시지 버스"""

    def __init__(self):
        self.subscribers = {}
        self.message_queue = asyncio.Queue()

    async def publish(self, message: AgentMessage):
        """메시지 발행"""
        await self.message_queue.put(message)

        # 브로드캐스트 처리
        if message.type == MessageType.BROADCAST:
            for agent_id in self.subscribers:
                if agent_id != message.sender:
                    await self.subscribers[agent_id].put(message)
        else:
            # 직접 전송
            if message.recipient in self.subscribers:
                await self.subscribers[message.recipient].put(message)

    async def subscribe(self, agent_id: str) -> asyncio.Queue:
        """구독"""
        queue = asyncio.Queue()
        self.subscribers[agent_id] = queue
        return queue
```

### 통신 패턴

```
┌─────────────────────────────────────────────────────────────┐
│                    Communication Patterns                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Request-Response (요청-응답)                             │
│     Agent A ──REQUEST──> Agent B                            │
│     Agent A <─RESPONSE── Agent B                            │
│                                                              │
│  2. Broadcast (브로드캐스트)                                 │
│     Orchestrator ──BROADCAST──> All Agents                  │
│                                                              │
│  3. Delegation Chain (위임 체인)                             │
│     A ──DELEGATE──> B ──DELEGATE──> C                       │
│     A <──RESULT─── B <──RESULT─── C                         │
│                                                              │
│  4. Peer-to-Peer (P2P)                                      │
│     Agent A <────────> Agent B                              │
│         ↕                  ↕                                 │
│     Agent C <────────> Agent D                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 오케스트레이션 메커니즘

### 중앙집중식 오케스트레이터

```python
class Orchestrator:
    """중앙 오케스트레이터"""

    def __init__(self, agents: dict[str, Agent]):
        self.agents = agents
        self.task_queue = asyncio.Queue()
        self.results = {}

    async def execute_complex_task(self, task: str) -> str:
        # 1. 작업 분해
        subtasks = await self._decompose_task(task)

        # 2. 에이전트 할당
        assignments = self._assign_agents(subtasks)

        # 3. 실행 계획 생성
        execution_plan = self._create_execution_plan(assignments)

        # 4. 실행
        for batch in execution_plan:
            batch_results = await asyncio.gather(*[
                self._execute_with_agent(assignment)
                for assignment in batch
            ])
            self._update_shared_context(batch_results)

        # 5. 결과 통합
        return await self._synthesize_results()

    def _assign_agents(self, subtasks: list) -> list:
        """최적의 에이전트 할당"""
        assignments = []

        for subtask in subtasks:
            # 능력 기반 매칭
            best_agent = max(
                self.agents.items(),
                key=lambda x: x[1].capability_score(subtask)
            )
            assignments.append({
                "subtask": subtask,
                "agent": best_agent[0],
                "priority": subtask.get("priority", 0)
            })

        return assignments

    def _create_execution_plan(self, assignments: list) -> list:
        """의존성 기반 실행 계획"""
        # 토폴로지 정렬로 의존성 해결
        graph = self._build_dependency_graph(assignments)
        return topological_sort(graph)
```

### 분산형 조정

```python
class DistributedAgent:
    """분산형 에이전트"""

    def __init__(self, agent_id: str, peers: list[str]):
        self.id = agent_id
        self.peers = peers
        self.state = AgentState.IDLE

    async def propose(self, task: str) -> str:
        """작업 제안 및 합의"""
        # 1. 제안 브로드캐스트
        proposal = {
            "id": str(uuid.uuid4()),
            "task": task,
            "proposer": self.id,
            "timestamp": time.time()
        }

        votes = await self._collect_votes(proposal)

        # 2. 합의 확인
        if self._has_consensus(votes):
            result = await self._execute(task)
            await self._broadcast_result(result)
            return result
        else:
            # 합의 실패 시 재조정
            return await self._renegotiate(proposal, votes)

    async def _collect_votes(self, proposal: dict) -> list:
        """피어들의 투표 수집"""
        responses = await asyncio.gather(*[
            self._request_vote(peer, proposal)
            for peer in self.peers
        ])
        return responses

    def _has_consensus(self, votes: list) -> bool:
        """과반수 합의 확인"""
        approvals = sum(1 for v in votes if v.get("approve"))
        return approvals > len(votes) / 2
```

---

## 협업 전략

### 1. 분업 (Division of Labor)

```python
class SpecializedAgentTeam:
    """전문화된 에이전트 팀"""

    def __init__(self):
        self.agents = {
            "researcher": ResearchAgent(),
            "planner": PlannerAgent(),
            "coder": CoderAgent(),
            "reviewer": ReviewerAgent(),
            "tester": TesterAgent()
        }

    async def solve_problem(self, problem: str) -> str:
        # 순차적 파이프라인
        research = await self.agents["researcher"].research(problem)
        plan = await self.agents["planner"].plan(research)
        code = await self.agents["coder"].implement(plan)
        review = await self.agents["reviewer"].review(code)

        if review["approved"]:
            tests = await self.agents["tester"].test(code)
            return {"code": code, "tests": tests}
        else:
            # 피드백 루프
            return await self._iterate(code, review["feedback"])
```

### 2. 토론 (Debate)

```
┌─────────────────────────────────────────────────────────────┐
│                    Debate Pattern                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Round 1: Initial Proposals                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                     │
│  │Agent A  │  │Agent B  │  │Agent C  │                     │
│  │Proposal │  │Proposal │  │Proposal │                     │
│  └────┬────┘  └────┬────┘  └────┬────┘                     │
│       │            │            │                           │
│       └────────────┼────────────┘                           │
│                    ▼                                         │
│  Round 2: Critique & Counter                                 │
│  ┌─────────────────────────────────────┐                    │
│  │  A critiques B, C                   │                    │
│  │  B critiques A, C                   │                    │
│  │  C critiques A, B                   │                    │
│  └─────────────────────────────────────┘                    │
│                    │                                         │
│                    ▼                                         │
│  Round 3: Synthesis                                          │
│  ┌─────────────────────────────────────┐                    │
│  │  Judge Agent: Best aspects of each  │                    │
│  │  → Final synthesized solution       │                    │
│  └─────────────────────────────────────┘                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

```python
class DebateSystem:
    """토론 기반 의사결정"""

    async def debate(self, topic: str, agents: list[Agent]) -> str:
        # Round 1: 초기 제안
        proposals = await asyncio.gather(*[
            agent.propose(topic) for agent in agents
        ])

        # Round 2: 비평
        critiques = []
        for i, agent in enumerate(agents):
            other_proposals = [p for j, p in enumerate(proposals) if j != i]
            critique = await agent.critique(other_proposals)
            critiques.append(critique)

        # Round 3: 반론
        rebuttals = await asyncio.gather(*[
            agent.rebut(critiques)
            for agent in agents
        ])

        # 합의 도출
        return await self._synthesize(proposals, critiques, rebuttals)
```

### 3. 계층적 위임 (Hierarchical Delegation)

```python
class HierarchicalTeam:
    """계층적 에이전트 팀"""

    def __init__(self):
        self.manager = ManagerAgent()
        self.leads = {
            "frontend": FrontendLeadAgent(),
            "backend": BackendLeadAgent(),
            "devops": DevOpsLeadAgent()
        }
        self.workers = {
            "frontend": [FrontendAgent() for _ in range(3)],
            "backend": [BackendAgent() for _ in range(3)],
            "devops": [DevOpsAgent() for _ in range(2)]
        }

    async def execute_project(self, project: str) -> dict:
        # 매니저가 큰 방향 결정
        strategy = await self.manager.strategize(project)

        # 리드들에게 분야별 계획 위임
        plans = {}
        for domain, lead in self.leads.items():
            domain_strategy = strategy.get(domain)
            plans[domain] = await lead.plan(domain_strategy)

        # 워커들에게 실행 위임
        results = {}
        for domain, plan in plans.items():
            lead = self.leads[domain]
            workers = self.workers[domain]

            # 리드가 작업 분배
            assignments = await lead.distribute(plan, len(workers))

            # 워커들 병렬 실행
            domain_results = await asyncio.gather(*[
                worker.execute(assignment)
                for worker, assignment in zip(workers, assignments)
            ])

            # 리드가 통합
            results[domain] = await lead.integrate(domain_results)

        # 매니저가 최종 통합
        return await self.manager.finalize(results)
```

---

## 공유 컨텍스트 관리

```python
class SharedContext:
    """에이전트 간 공유 컨텍스트"""

    def __init__(self):
        self.state = {}
        self.history = []
        self.lock = asyncio.Lock()

    async def update(self, agent_id: str, key: str, value: Any):
        """상태 업데이트"""
        async with self.lock:
            self.state[key] = value
            self.history.append({
                "agent": agent_id,
                "action": "update",
                "key": key,
                "value": value,
                "timestamp": time.time()
            })

    async def read(self, key: str) -> Any:
        """상태 읽기"""
        return self.state.get(key)

    def get_context_for_agent(self, agent_id: str) -> dict:
        """에이전트용 컨텍스트 구성"""
        return {
            "current_state": self.state,
            "recent_updates": [
                h for h in self.history[-10:]
                if h["agent"] != agent_id
            ],
            "my_updates": [
                h for h in self.history
                if h["agent"] == agent_id
            ][-5:]
        }
```

---

## 창발적 행동

멀티에이전트 시스템에서 예상치 못한 협업 패턴이 나타날 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Emergent Behaviors                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Self-Organization (자기 조직화)                          │
│     - 에이전트들이 자발적으로 역할 분담                      │
│     - 병목 발견 시 자동 조정                                 │
│                                                              │
│  2. Collective Intelligence (집단 지성)                      │
│     - 개별 에이전트보다 뛰어난 팀 성능                       │
│     - 다양한 관점의 통합                                     │
│                                                              │
│  3. Adaptive Specialization (적응적 전문화)                  │
│     - 경험 축적에 따른 역할 진화                             │
│     - 팀 구성 자동 최적화                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*다음 글에서는 Emergent Symbolic Mechanisms를 살펴봅니다.*
