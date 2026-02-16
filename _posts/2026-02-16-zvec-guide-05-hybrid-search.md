---
layout: post
title: "zvec 완벽 가이드 (05) - 하이브리드 검색"
date: 2026-02-16
permalink: /zvec-guide-05-hybrid-search/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, Hybrid Search, Dense, Sparse]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec의 하이브리드 검색: Dense 벡터와 Sparse 벡터를 결합한 고급 검색 기능을 알아봅니다."
---

## 하이브리드 검색 개요

zvec는 **Dense(밀도)** 벡터와 **Sparse(희소)** 벡터를 모두 지원하며, 이를 결합한 하이브리드 검색을 제공합니다.

### Dense vs Sparse

| 타입 | 설명 | 사용 예시 |
|------|------|----------|
| **Dense** | 대부분의 차원이 0이 아닌 값 | BERT, Sentence-BERT 임베딩 |
| **Sparse** | 대부분의 차원이 0, 일부만 값 | BM25, SPLADE |

---

## 스키마에서 하이브리드 구성

```python
import zvec

schema = zvec.CollectionSchema(
    name="hybrid_example",
    vectors={
        # Dense 벡터 (의미적 검색용)
        "dense_embedding": zvec.VectorSchema(
            name="dense_embedding",
            dtype=zvec.DataType.VECTOR_FP32,
            dimension=768,
            metric_type=zvec.MetricType.COSINE
        ),
    },
    sparse_vectors={
        # Sparse 벡터 (키워드 검색용)
        "sparse_embedding": zvec.VectorSchema(
            name="sparse_embedding",
            dtype=zvec.DataType.SPARSE_FP32,
        )
    }
)
```

---

## 데이터 삽입

```python
# 문서 삽입 (Dense + Sparse)
docs = [
    zvec.Doc(
        id="doc_1",
        vectors={
            "dense_embedding": [0.1, 0.2, 0.3, ...]  # 768차원
        },
        sparse_vectors={
            "sparse_embedding": {
                0: 0.5,    # 단어 ID: TF-IDF 점수
                15: 0.3,
                42: 0.8,
            }
        },
        payload={
            "text": "Python 프로그래밍 가이드",
            "category": "book"
        }
    ),
]

collection.insert(docs)
collection.commit()
```

---

## 하이브리드 검색 실행

### 가중치 기반 결합

```python
# 각 벡터 타입에 가중치 부여
results = collection.query(
    hybrid={
        "dense": {
            "field": "dense_embedding",
            "vector": query_dense_vector,
            "weight": 0.7  # 70% 가중치
        },
        "sparse": {
            "field": "sparse_embedding",
            "vector": query_sparse_vector,
            "weight": 0.3  # 30% 가중치
        }
    },
    topk=10
)
```

### RRf (Reciprocal Rank Fusion)

순위 기반fusion으로 결합:

```python
results = collection.query(
    hybrid={
        "method": "rrf",  # Reciprocal Rank Fusion
        "queries": [
            {
                "field": "dense_embedding",
                "vector": query_dense,
                "topk": 100  # 더 넓은 범위에서 검색
            },
            {
                "field": "sparse_embedding",
                "vector": query_sparse,
                "topk": 100
            }
        ],
        "k": 60  # RRF 파라미터
    },
    topk=10
)
```

---

## 필터와 하이브리드 결합

하이브리드 검색에 필터 조건 추가:

```python
results = collection.query(
    hybrid={
        "dense": {
            "field": "dense_embedding",
            "vector": query_dense
        },
        "sparse": {
            "field": "sparse_embedding",
            "vector": query_sparse
        }
    },
    filter_expr='payload.category == "book" AND payload.price > 10000',
    topk=10
)
```

---

## 실전 예제: 문서 검색

```python
import zvec
from sentence_transformers import SentenceTransformer

# 1. 스키마 정의
schema = zvec.CollectionSchema(
    name="documents",
    vectors={
        "dense": zvec.VectorSchema("dense", zvec.DataType.VECTOR_FP32, 384)
    },
    sparse_vectors={
        "sparse": zvec.VectorSchema("sparse", zvec.DataType.SPARSE_FP32)
    }
)

# 2. 컬렉션 생성
collection = zvec.create_and_open("./data/docs", schema=schema)

# 3. 문서 삽입 (Dense + Sparse)
# 실제로는 전처리 파이프라인에서 변환
docs = [
    zvec.Doc(
        id="doc_1",
        vectors={"dense": dense_embedding_1},
        sparse_vectors={"sparse": sparse_embedding_1},
        payload={"title": "Python 기초", "content": "Python 프로그래밍..."}
    ),
]
collection.insert(docs)
collection.build_index(["dense"])
collection.commit()

# 4. 하이브리드 검색
query = "Python 프로그래밍 배우기"

# Dense 임베딩 생성
dense_q = model.encode(query)

# Sparse 임베딩 생성 (예: BM25)
sparse_q = bm25.encode(query)

# 하이브리드 검색
results = collection.query(
    hybrid={
        "dense": {"field": "dense", "vector": dense_q, "weight": 0.6},
        "sparse": {"field": "sparse", "vector": sparse_q, "weight": 0.4}
    },
    topk=5
)

for r in results:
    print(f"{r['id']}: {r['payload']['title']} (score: {r['score']:.3f})")
```

---

##何时使用 Hybrid Search

| 시나리오 | 권장 방법 |
|----------|----------|
| 의미적 검색 중심 | Dense Only (가중치 1.0) |
| 키워드 중심 | Sparse Only |
| 정확한 키워드 + 범주적 의미 모두 필요 | Hybrid (가중치 or RRF) |
| 긴 문서 검색 | Hybrid + 필터링 |

---

## 성능 팁

1. **인덱스 빌드**: Dense 벡터에만 인덱스 빌드 (Sparse는本身就 인덱스)
2. **topk**: 하이브리드 검색 시 각 서브쿼리의 topk를 더 크게 설정
3. **캐싱**: 반복되는 쿼리는 캐싱 활용

---

*다음 글에서는 성능 최적화에 대해 살펴보겠습니다.*
