---
layout: post
title: "UltraRAG 완벽 가이드 (07) - 평가(Evaluation) 시스템"
date: 2026-02-15
permalink: /ultrarag-guide-07-evaluation/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, Evaluation, Benchmark,评测]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "UltraRAG의 통합 평가 시스템: 벤치마크 데이터셋, 평가 지표, 분석 도구까지详细介绍합니다."
---

## 평가 시스템 개요

UltraRAG는 **통합 평가 시스템**을 제공하여 연구 효율성을 극대화합니다:

- **표준화된 평가 워크플로우**: 일관된 평가 프로세스
- **주요 벤치마크 데이터셋**: 즉시 사용 가능한 연구 데이터
- **통합 지표 관리**: 다양한 평가 지표 지원
- **베이스라인 통합**: 재현 가능한 실험 비교
- **시각적 분석 도구**: Case Study 인터페이스

---

## 벤치마크 데이터셋

UltraRAG는 RAG 분야에서 널리 사용되는 평가 데이터셋을 제공합니다:

### 사용 가능한 데이터셋

| 데이터셋 | 설명 | 링크 |
|----------|------|------|
| **TREC-CAR** | 자동 요약 및 검색 | [ModelScope](https://modelscope.cn/datasets/UltraRAG/UltraRAG_Benchmark) |
| **NQ** | Natural Questions | Google |
| **HotpotQA** | 다중 문서 질문 응답 | Stanford |
| **2WikiMultiHopQA** | 다단계 추론 QA | 2Wiki |
| **PopQA** |/entity 질문 응답 | PopQA |

### 데이터셋 다운로드

```yaml
pipeline:
  - benchmark.get_data:
      dataset: "trec-car"
      split: "test"
      output_dir: "./data/benchmarks/trec-car/"
```

---

## 평가 지표

### 검색 평가 지표

| 지표 | 설명 | 범위 |
|------|------|------|
| **Precision@K** | 상위 K개 중 관련 문서 비율 | 0-1 |
| **Recall@K** | 관련 문서 중 상위 K개 비율 | 0-1 |
| **MRR** | 첫 번째 관련 문서 순위의 역수 | 0-1 |
| **NDCG** | 정규화된 누적 이득 | 0-1 |

### 생성 평가 지표

| 지표 | 설명 |
|------|------|
| **Rouge-L** | 생성 텍스트와 참조 텍스트의最长 공통 부분 시퀀스 |
| **BLEU** | n-gram 중복 기반 점수 |
| **Faithfulness** | 생성 결과가 참조 문서와 일치하는 정도 |
| **Answer Relevance** | 답변이 질문과 관련 있는 정도 |

---

## 평가 파이프라인

### 기본 평가 실행

```yaml
# examples/eval_trec.yaml
pipeline:
  - benchmark.get_data:
      dataset: "trec-car"
      split: "test"

  - retriever.retriever_init
  - generation.generation_init

  - retriever.retriever_search
  - generation.generate

  - evaluation.evaluate:
      metrics: ["precision", "recall", "mrr", "ndcg"]
      k_values: [1, 3, 5, 10]
```

### 생성 결과 평가

```yaml
# examples/evaluate_results.yaml
pipeline:
  - evaluation.evaluate_generation:
      reference_path: "./data/golden_answers.json"
      prediction_path: "./outputs/predictions.json"
      metrics: ["rouge", "bleu", "faithfulness"]
```

---

## 평가 결과 분석

### Case Study 인터페이스

UltraRAG는 시각적인 **Case Study 인터페이스**를 제공합니다:

```
Workflow:
Query → Retrieve → Generate → Evaluate
                ↓
         Intermediate Outputs
                ↓
         Visual Analysis
```

각 워크플로우의 중간 출력을 시각적으로 추적할 수 있습니다:

```yaml
pipeline:
  - benchmark.get_data

  - retriever.retriever_search:
      return_intermediate: true  # 중간 결과 반환

  - evaluation.case_study:
      output_dir: "./analysis/cases/"
      visualize: true
```

---

## 커스텀 평가

### 커스텀 지표 정의

```python
# custom_evaluator.py
from ultrarag.evaluation import BaseEvaluator

class MyCustomMetric(BaseEvaluator):
    name = "custom_f1"

    def compute(self, predictions, references):
        # 커스텀 F1 점수 계산
        ...
```

### 커스텀 평가 실행

```yaml
pipeline:
  - evaluation.evaluate:
      metrics: ["custom_f1", "rouge", "precision"]
      custom_evaluators:
        - "./custom_evaluator.py"
```

---

## 베이스라인 비교

### 베이스라인 설정

```yaml
pipeline:
  # UltraRAG 구성
  - config:
      name: "UltraRAG"
      pipeline: "./configs/ultrarag.yaml"

  # 기본 구성 (비교용)
  - config:
      name: "Baseline-Dense"
      pipeline: "./configs/baseline_dense.yaml"

  - config:
      name: "Baseline-BM25"
      pipeline: "./configs/baseline_bm25.yaml"

  # 평가 실행
  - evaluation.compare:
      output: "./results/comparison.csv"
      visualize: true
```

---

## 평가 결과 형식

평가 결과는 CSV 또는 JSON 형식으로 저장됩니다:

```json
{
  "experiment": "ultrarag_v3_baseline",
  "timestamp": "2026-01-23T10:30:00",
  "results": {
    "precision@5": 0.85,
    "recall@5": 0.72,
    "mrr": 0.88,
    "ndcg@10": 0.82,
    "rouge_l": 0.45
  },
  "per_case": [
    {
      "query_id": "q001",
      "precision@5": 0.9,
      "recall@5": 0.8,
      "answer": "..."
    }
  ]
}
```

---

## 팁 및 베스트 프랙티스

1. **일관된 평가 프로토콜**: 같은 조건에서 여러 번 실험하여 평균 사용
2. **다양한 지표 조합**:单一 지표에 의존하지 말고 여러 지표 함께 사용
3. **Error Analysis**: 오류 사례를 철저히 분석하여 개선점 파악
4. **Human Evaluation**: 자동 평가만 신뢰하지 않고 human 평가도 병행

---

*다음 글에서는 UltraRAG UI 활용 방법에 대해 살펴보겠습니다.*
