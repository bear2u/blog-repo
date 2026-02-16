---
layout: post
title: "zvec 완벽 가이드 (04) - 검색 기능"
date: 2026-02-16
permalink: /zvec-guide-04-search/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, Search, Query, Similarity]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec의 벡터 검색 기능을 알아봅니다. 유사도 검색, 필터링, top-k 검색까지 상세 가이드입니다."
---

## 검색 개요

zvec의 핵심 기능인 벡터 검색을 살펴보겠습니다. 유사도 기반으로 가장 관련된 문서를 찾습니다.

---

## 기본 검색

### VectorQuery 생성

```python
import zvec

# 벡터 쿼리 생성
query = zvec.VectorQuery(
    field="embedding",              # 벡터 필드 이름
    vector=[0.1, 0.2, 0.3, 0.4],    # 검색할 벡터
    topk=10,                        # 반환할 결과 수
)
```

### 검색 실행

```python
# 컬렉션에서 검색
results = collection.query(
    vector=query,
    topk=10
)

# 결과 출력
for result in results:
    print(f"ID: {result['id']}, Score: {result['score']}")
```

### 결과 형식

```python
[
    {
        'id': 'doc_1',
        'score': 0.95,
        'vectors': {'embedding': [0.1, 0.2, 0.3, 0.4]},
        'payload': {'text': '문서 내용'}
    },
    {
        'id': 'doc_2',
        'score': 0.87,
        'vectors': {'embedding': [0.2, 0.3, 0.4, 0.5]},
        'payload': {'text': '다른 내용'}
    }
]
```

---

## 유사도 점수

zvec는 다음과 같은 유사도 메트릭을 지원합니다:

| 메트릭 | 설명 | 가장 큰 값이 |
|--------|------|-------------|
| `COSINE` | 코사인 유사도 | 유사함 |
| `EUCLIDEAN` | 유클리드 거리 | 유사함 |
| `DOT` | 내적 | 유사함 |

### 메트릭 지정

```python
# 스키마 생성 시 메트릭 지정
schema = zvec.CollectionSchema(
    name="example",
    vectors={
        "embedding": zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            128,
            metric_type=zvec.MetricType.COSINE  # 코사인 유사도
        )
    }
)
```

---

## 필터링

### 기본 필터링

payload를 기반으로 필터링:

```python
# 조건 필터
results = collection.query(
    vector=query,
    filter_expr='payload.category == "tech"'  # SQL 스타일 필터
)
```

### 필터 표현식

| 표현식 | 설명 |
|--------|------|
| `payload.age > 18` | 숫자 비교 |
| `payload.status == "active"` | 문자열 비교 |
| `payload.tags CONTAINS "AI"` | 배열 포함 |
| `payload.price BETWEEN 10000 AND 50000` | 범위 |

### 논리 연산자

```python
# AND 조건
filter_expr='payload.category == "tech" AND payload.price > 10000'

# OR 조건
filter_expr='payload.status == "published" OR payload.status == "draft"'

# NOT 조건
filter_expr='NOT payload.deleted'
```

---

## 멀티벡터 검색

여러 벡터 필드를 동시에 검색:

```python
# 멀티벡터 쿼리
query = zvec.VectorQuery(
    field=["embedding", "image_embedding"],  # 여러 필드
    vector={
        "embedding": [0.1, 0.2, ...],
        "image_embedding": [0.5, 0.6, ...]
    },
    topk=10
)

results = collection.query(vector=query)
```

---

## 희소 벡터 검색

희소 벡터로 검색:

```python
# 희소 벡터 쿼리
sparse_query = zvec.SparseVectorQuery(
    field="sparse_embedding",
    vector={
        0: 0.5,   # 인덱스: 값
        100: 0.3,
        500: 0.2
    },
    topk=10
)

results = collection.query(vector=sparse_query)
```

---

## 결과 페이징

```python
# 오프셋 기반 페이징
results = collection.query(
    vector=query,
    topk=10,
    offset=0   # 시작 위치
)

# 다음 페이지
next_results = collection.query(
    vector=query,
    topk=10,
    offset=10  # 10개 건너뛰기
)
```

---

## 검색 파라미터

### HNSW 파라미터

```python
results = collection.query(
    vector=query,
    topk=10,
    params={
        "ef": 50,    # 검색 확장 필드 (크면 더 정확하지만 느림)
        "M": 16      # 인덱스 빌드 시와 동일
    }
)
```

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `ef` | 검색 시 탐색 범위 | 50-200 |
| `M` | 인덱스 빌드 파라미터 | 16-64 |

---

## 전체 예제

```python
import zvec

# 컬렉션 열기
collection = zvec.open("./data/books")

# 쿼리 벡터 (실제 임베딩으로 교체)
query_vector = [0.15] * 128

# 검색 실행 (카테고리 필터 + top-k)
results = collection.query(
    vector=zvec.VectorQuery(
        field="embedding",
        vector=query_vector,
        topk=5
    ),
    filter_expr='payload.price > 0',
    params={"ef": 100}
)

# 결과 출력
print(f"검색 결과: {len(results)}개")
for i, result in enumerate(results, 1):
    print(f"{i}. {result['id']} (score: {result['score']:.3f})")
    print(f"   제목: {result['payload']['title']}")
    print(f"   가격: {result['payload']['price']}원")

collection.close()
```

---

*다음 글에서는 하이브리드 검색 기능에 대해 살펴보겠습니다.*
