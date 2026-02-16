---
layout: post
title: "zvec 완벽 가이드 (06) - 성능 최적화"
date: 2026-02-16
permalink: /zvec-guide-06-optimization/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, Optimization, Performance, Index]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec의 성능을 최적화하는 방법을 알아봅니다. 인덱스 설정, 배치 처리, 메모리 관리까지 상세 가이드입니다."
---

## 성능 최적화 개요

zvec의 뛰어난 성능을 최대한 활용하려면 다음 최적화 기법을 적용하세요.

---

## 인덱스

인덱스는 검색 성능에 결정적입니다. 데이터 삽입 후 반드시 인덱스를 빌드하세요.

### 인덱스 타입

| 타입 | 설명 | 적합한 시나리오 |
|------|------|----------------|
| `HNSW` | 계층적 탐색 가능近似最近접 이웃 | 고품질 검색, 대용량 |
| `FLAT` | 완전 탐색 | 소규모 데이터, 정확도 우선 |

### HNSW 인덱스 빌드

```python
collection.build_index(
    vectors=["embedding"],
    index_type="HNSW",
    params={
        "M": 16,                 # 각 노드의 엣지 수 (클수록 정확, 메모리 더 사용)
        "efConstruction": 200,  # 인덱스 빌드 시 탐색 범위
    }
)
```

### 인덱스 파라미터 튜닝

| 파라미터 | 설명 | 권장값 |
|----------|------|--------|
| `M` | 각 노드의 최대 이웃 수 | 16-64 |
| `efConstruction` | 인덱스 빌드 시 탐색 깊이 | 100-400 |
| `ef` | 검색 시 탐색 범위 | 50-200 |

```python
# 검색 시 파라미터 조정
results = collection.query(
    vector=query,
    params={"ef": 100}  # 높을수록 정확도 ↑, 속도 ↓
)
```

---

## 배치 처리

대량 데이터 삽입 시 배치 처리를 활용하세요:

```python
# 나쁜 예: 개별 삽입
for doc in documents:
    collection.insert(doc)  # 매번 I/O 발생

# 좋은 예: 배치 삽입
BATCH_SIZE = 1000

for i in range(0, len(documents), BATCH_SIZE):
    batch = documents[i:i + BATCH_SIZE]
    collection.insert(batch)

collection.commit()  # 한 번에 커밋
```

---

## 메모리 관리

### 필요한 시점만 열기

```python
# 사용 후 반드시 닫기
try:
    collection = zvec.open(path)
    # 작업 수행
finally:
    collection.close()
```

### 컨텍스트 매니저 사용

```python
with zvec.open(path) as collection:
    results = collection.query(query)
# 자동 정리
```

---

## 검색 최적화

### topk 적절히 설정

```python
# 너무 큰 topk는 성능 저하
results = collection.query(
    vector=query,
    topk=10  # 실제 필요한 만큼만
)
```

### 필터 expr 최적화

```python
# 나쁜 예: 함수 사용
filter_expr='upper(payload.category) == "BOOK"'

# 좋은 예: 단순 비교
filter_expr='payload.category == "book"'

# 인덱스된 필드로 필터링
filter_expr='payload.status == "active"'  # status에 인덱스
```

---

## 동시성

### 읽기 전용 컬렉션 공유

```python
# 여러 스레드에서 읽기 전용으로 열기
collection = zvec.open(path, read_only=True)

# 읽기 전용이므로 병렬 접근 안전
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(search, q) for q in queries]
    results = [f.result() for f in futures]
```

---

## 프로파일링

### 검색 시간 측정

```python
import time

start = time.time()
results = collection.query(vector=query, topk=10)
elapsed = time.time() - start

print(f"검색 시간: {elapsed*1000:.2f}ms")
print(f"결과 수: {len(results)}")
```

---

## 권장 설정

### 고속 검색

```python
collection.build_index(
    vectors=["embedding"],
    index_type="HNSW",
    params={
        "M": 32,
        "efConstruction": 200
    }
)

# 검색
results = collection.query(
    vector=query,
    params={"ef": 100}
)
```

### 고품질 검색

```python
collection.build_index(
    vectors=["embedding"],
    index_type="HNSW",
    params={
        "M": 64,
        "efConstruction": 400
    }
)

# 검색
results = collection.query(
    vector=query,
    params={"ef": 200}
)
```

---

## 모니터링

### 컬렉션 상태 확인

```python
# 메타데이터 확인
info = collection.info()
print(f"문서 수: {info['doc_count']}")
print(f"벡터 차원: {info['dimension']}")
print(f"인덱스: {info['index_type']}")
```

---

## 성능 벤치마크 결과

| 데이터 크기 | 검색 latency | QPS |
|------------|-------------|-----|
| 1M vectors | ~5ms | ~200 |
| 10M vectors | ~15ms | ~70 |
| 100M vectors | ~50ms | ~20 |

---

*다음 글에서는 확장 및 운영 방법을 살펴보겠습니다.*
