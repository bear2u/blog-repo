---
layout: post
title: "Context Engineering 완벽 가이드 (5) - Memory Systems"
date: 2026-02-05
permalink: /context-engineering-guide-05-memory/
author: davidkimai
categories: [AI 에이전트, Context Engineering]
tags: [Context Engineering, Memory, MEM1, RAG, Long-term Memory]
original_url: "https://github.com/davidkimai/Context-Engineering"
excerpt: "Singapore-MIT MEM1 연구 기반의 메모리 시스템 설계를 알아봅니다."
---

## Memory Systems 개요

> **"MEM1 trains AI agents to keep only what matters—merging memory and reasoning at every step."**
> — Singapore-MIT, 2025

효과적인 메모리 시스템은 장기 에이전트의 핵심입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               Sensory Buffer (즉시)                  │    │
│  │  • 현재 입력                                         │    │
│  │  • 도구 출력                                         │    │
│  │  • 최근 응답                                         │    │
│  └───────────────────────────┬─────────────────────────┘    │
│                              │                              │
│                              ▼                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Working Memory (단기)                     │    │
│  │  • 현재 작업 컨텍스트                                │    │
│  │  • 활성화된 계획                                     │    │
│  │  • 임시 변수                                         │    │
│  └───────────────────────────┬─────────────────────────┘    │
│                              │                              │
│           ┌──────────────────┼──────────────────┐          │
│           ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │  Episodic   │    │  Semantic   │    │ Procedural  │    │
│  │  Memory     │    │  Memory     │    │  Memory     │    │
│  │  (이벤트)   │    │  (지식)     │    │  (기술)     │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## MEM1 연구 핵심

### 추론 기반 메모리 통합

```
┌─────────────────────────────────────────────────────────────┐
│                    MEM1 Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Traditional Approach:                                       │
│  [Context 1][Context 2][Context 3]...[Context N]            │
│  → 컨텍스트가 계속 쌓여서 폭발                              │
│                                                              │
│  MEM1 Approach:                                              │
│  [Internal State] ← 매 단계마다 압축/갱신                   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Step 1: Observe → Reason → Compress → Update State    │ │
│  │  Step 2: Observe → Reason → Compress → Update State    │ │
│  │  Step 3: Observe → Reason → Compress → Update State    │ │
│  │  ...                                                   │ │
│  │  State = 항상 고정된 크기 유지                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 핵심 원칙

1. **Selective Retention**: 중요한 것만 유지
2. **Reasoning-Driven**: 추론을 통해 무엇이 중요한지 판단
3. **Fixed Capacity**: 고정된 메모리 크기 유지
4. **Continuous Consolidation**: 지속적인 통합

---

## 메모리 유형별 구현

### 1. Working Memory (작업 메모리)

```python
class WorkingMemory:
    """현재 작업을 위한 단기 메모리"""

    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.items = []
        self.focus = None

    def add(self, item: dict):
        """새 항목 추가 (용량 초과시 오래된 것 제거)"""
        self.items.append({
            **item,
            "timestamp": time.time(),
            "relevance": 1.0
        })

        if len(self.items) > self.capacity:
            self._evict_least_relevant()

    def _evict_least_relevant(self):
        """가장 관련성 낮은 항목 제거"""
        # 시간 경과에 따라 관련성 감소
        for item in self.items:
            age = time.time() - item["timestamp"]
            item["relevance"] *= 0.95 ** (age / 60)  # 분당 5% 감소

        # 가장 낮은 관련성 항목 제거
        self.items.sort(key=lambda x: x["relevance"], reverse=True)
        self.items = self.items[:self.capacity]

    def get_context(self) -> str:
        """현재 작업 컨텍스트 반환"""
        return "\n".join([
            f"[{item['type']}] {item['content']}"
            for item in self.items
        ])
```

### 2. Episodic Memory (일화 메모리)

```python
class EpisodicMemory:
    """이벤트 기반 장기 메모리"""

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.episodes = []

    def record_episode(self, episode: dict):
        """에피소드 기록"""
        self.episodes.append({
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event": episode["event"],
            "context": episode["context"],
            "outcome": episode["outcome"],
            "importance": self._calculate_importance(episode)
        })
        self._persist()

    def _calculate_importance(self, episode: dict) -> float:
        """에피소드 중요도 계산"""
        importance = 0.5  # 기본값

        # 성공/실패 여부
        if episode.get("outcome", {}).get("success"):
            importance += 0.2

        # 사용자 피드백
        if episode.get("user_feedback"):
            importance += 0.3

        # 참조 횟수
        importance += min(0.3, episode.get("reference_count", 0) * 0.05)

        return min(1.0, importance)

    def recall(self, query: str, k: int = 5) -> list:
        """관련 에피소드 회상"""
        # 의미적 유사도로 검색
        query_embedding = embed(query)

        scored = []
        for episode in self.episodes:
            episode_embedding = embed(episode["event"])
            similarity = cosine_similarity(query_embedding, episode_embedding)

            # 중요도와 최근성 가중치
            recency = self._recency_weight(episode["timestamp"])
            score = similarity * episode["importance"] * recency

            scored.append((score, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:k]]

    def _recency_weight(self, timestamp: str) -> float:
        """최근성 가중치 계산"""
        age_days = (datetime.now() - datetime.fromisoformat(timestamp)).days
        return 0.5 ** (age_days / 30)  # 30일마다 반감
```

### 3. Semantic Memory (의미 메모리)

```python
class SemanticMemory:
    """지식 기반 장기 메모리"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.knowledge_graph = {}

    def store_fact(self, fact: dict):
        """사실 저장"""
        embedding = embed(fact["content"])

        self.vector_store.add(
            id=fact["id"],
            vector=embedding,
            metadata={
                "content": fact["content"],
                "source": fact["source"],
                "confidence": fact["confidence"],
                "category": fact["category"]
            }
        )

        # 지식 그래프 연결
        self._add_to_graph(fact)

    def query(self, query: str, k: int = 10) -> list:
        """지식 검색"""
        results = self.vector_store.search(
            query=embed(query),
            k=k
        )

        # 지식 그래프로 관련 사실 확장
        expanded = self._expand_with_graph(results)

        return expanded

    def _expand_with_graph(self, results: list) -> list:
        """지식 그래프로 결과 확장"""
        expanded = list(results)

        for result in results:
            category = result["metadata"]["category"]
            related = self.knowledge_graph.get(category, [])
            expanded.extend(related[:3])  # 카테고리당 최대 3개

        return expanded
```

---

## RAG (Retrieval-Augmented Generation)

### 기본 RAG 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Pipeline                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  User Query                                                  │
│      │                                                       │
│      ▼                                                       │
│  ┌───────────────┐                                          │
│  │   Embedding   │ → Query를 벡터로 변환                    │
│  └───────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐     ┌─────────────────┐                  │
│  │   Retrieval   │────►│  Vector Store   │                  │
│  │               │◄────│  (문서 DB)      │                  │
│  └───────┬───────┘     └─────────────────┘                  │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐                                          │
│  │   Reranking   │ → 검색 결과 재정렬                       │
│  └───────┬───────┘                                          │
│          │                                                   │
│          ▼                                                   │
│  ┌───────────────┐                                          │
│  │  Generation   │ → LLM이 컨텍스트와 함께 응답             │
│  └───────────────┘                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 고급 RAG 전략

```python
class AdvancedRAG:
    """고급 RAG 구현"""

    def __init__(self, vector_store, reranker, llm):
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm

    async def query(self, question: str) -> str:
        # 1. 쿼리 확장
        expanded_queries = await self._expand_query(question)

        # 2. 멀티 쿼리 검색
        all_docs = []
        for query in expanded_queries:
            docs = await self.vector_store.search(query, k=10)
            all_docs.extend(docs)

        # 3. 중복 제거 및 재정렬
        unique_docs = self._deduplicate(all_docs)
        reranked = await self.reranker.rerank(question, unique_docs)

        # 4. 컨텍스트 구성
        context = self._format_context(reranked[:5])

        # 5. 생성
        response = await self.llm.generate(
            system="Answer based on the provided context.",
            context=context,
            question=question
        )

        return response

    async def _expand_query(self, question: str) -> list:
        """쿼리 확장"""
        prompt = f"""
        Generate 3 alternative versions of this question
        to improve search coverage:

        Original: {question}

        Alternatives:
        """
        response = await self.llm.generate(prompt)
        alternatives = parse_alternatives(response)
        return [question] + alternatives
```

---

## 메모리 통합 예시

```python
class UnifiedMemorySystem:
    """통합 메모리 시스템"""

    def __init__(self):
        self.working = WorkingMemory(capacity=10)
        self.episodic = EpisodicMemory("./episodic_memory")
        self.semantic = SemanticMemory(VectorStore())

    async def process_interaction(
        self,
        user_input: str,
        agent_response: str,
        tool_results: list = None
    ):
        # 1. 작업 메모리 업데이트
        self.working.add({
            "type": "user",
            "content": user_input
        })
        self.working.add({
            "type": "assistant",
            "content": agent_response
        })

        # 2. 에피소드 기록
        self.episodic.record_episode({
            "event": f"User: {user_input}\nAssistant: {agent_response}",
            "context": self.working.get_context(),
            "outcome": {"success": True}
        })

        # 3. 새로운 사실 추출 및 저장
        facts = await self._extract_facts(user_input, agent_response)
        for fact in facts:
            self.semantic.store_fact(fact)

    def build_context(self, query: str) -> str:
        """쿼리에 대한 전체 컨텍스트 구성"""
        context_parts = []

        # 작업 메모리
        context_parts.append("## Current Context")
        context_parts.append(self.working.get_context())

        # 관련 에피소드
        episodes = self.episodic.recall(query, k=3)
        if episodes:
            context_parts.append("\n## Relevant Past Interactions")
            for ep in episodes:
                context_parts.append(f"- {ep['event'][:200]}...")

        # 관련 지식
        knowledge = self.semantic.query(query, k=5)
        if knowledge:
            context_parts.append("\n## Relevant Knowledge")
            for k in knowledge:
                context_parts.append(f"- {k['content']}")

        return "\n".join(context_parts)
```

---

## 메모리 최적화 전략

| 전략 | 설명 | 사용 시점 |
|------|------|-----------|
| **Summarization** | 오래된 대화 요약 | 컨텍스트 윈도우 초과 시 |
| **Pruning** | 관련 없는 정보 제거 | 정기적으로 |
| **Compression** | 중복 정보 압축 | 저장 시 |
| **Hierarchical** | 계층적 저장 | 대규모 지식 베이스 |
| **Forgetting** | 의도적 망각 | 오래되고 사용 안 되는 정보 |

---

*다음 글에서는 Multi-Agent Systems를 살펴봅니다.*
