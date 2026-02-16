---
layout: post
title: "zvec 완벽 가이드 (03) - 핵심 API"
date: 2026-02-16
permalink: /zvec-guide-03-core-api/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, API, Collection, Document]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec의 핵심 API를 알아봅니다: CollectionSchema, Document, VectorQuery의 사용법을 상세히 설명합니다."
---

## 핵심 API 개요

zvec의 핵심은 다음과 같은 주요 클래스로 구성됩니다:

- **CollectionSchema**: 컬렉션 스키마 정의
- **Collection**: 벡터 저장 및 검색 관리
- **Document**: 개별 문서(벡터) 표현
- **VectorQuery**: 검색 쿼리

---

## CollectionSchema

컬렉션의 구조를 정의합니다:

```python
import zvec

schema = zvec.CollectionSchema(
    name="my_collection",           # 컬렉션 이름
    vectors={
        "embedding": zvec.VectorSchema(
            name="embedding",       # 벡터 필드 이름
            dtype=zvec.DataType.VECTOR_FP32,  #数据类型
            dimension=128           # 벡터 차원
        )
    },
    # 선택적:稀疏 벡터
    sparse_vectors={
        "sparse_embedding": zvec.VectorSchema(
            name="sparse_embedding",
            dtype=zvec.DataType.SPARSE_FP32,
        )
    }
)
```

### DataType 지원

| 타입 | 설명 |
|------|------|
| `VECTOR_FP32` | 32비트 부동소수점 벡터 |
| `VECTOR_FP16` | 16비트 부동소수점 벡터 |
| `SPARSE_FP32` | 희소 벡터 |

---

## 컬렉션 생성 및 관리

### 컬렉션 생성

```python
# 새 컬렉션 생성
collection = zvec.create_and_open(
    path="./data/my_collection",
    schema=schema
)
```

### 기존 컬렉션 열기

```python
# 이미 생성된 컬렉션 열기
collection = zvec.open(path="./data/my_collection")
```

### 컬렉션 삭제

```python
# 컬렉션 삭제
zvec.delete(path="./data/my_collection")
```

---

## Document

삽입할 문서를 표현합니다:

```python
# 단일 문서
doc = zvec.Doc(
    id="doc_1",                                    # 문서 ID
    vectors={"embedding": [0.1, 0.2, 0.3, ...]},  # 벡터
    sparse_vectors={"sparse_embedding": {...}},     # 희소 벡터 (선택)
    payload={"text": "문서 내용", "category": "tech"}  # 메타데이터
)

# 여러 문서
docs = [
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.5]}),
]
```

### Document 속성

| 속성 | 설명 |
|------|------|
| `id` | 문서 고유 식별자 |
| `vectors` | 벡터 데이터 딕셔너리 |
| `sparse_vectors` | 희소 벡터 딕셔너리 |
| `payload` | 메타데이터/필터용 데이터 |

---

## 문서 삽입

```python
# 단일 문서 삽입
collection.insert(doc)

# 여러 문서 삽입
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.5]}),
    zvec.Doc(id="doc_3", vectors={"embedding": [0.3, 0.4, 0.5, 0.6]}),
])
```

---

## 문서 조회

```python
# ID로 조회
doc = collection.get("doc_1")

# 조건으로 조회
docs = collection.filter(
    filter_expr='payload.category == "tech"'
)
```

---

## 문서 삭제

```python
# ID로 삭제
collection.delete(["doc_1", "doc_2"])

# 조건으로 삭제
collection.delete_filter(
    filter_expr='payload.category == "archived"'
)
```

---

## 인덱스 빌드

데이터 삽입 후 인덱스를 빌드해야 검색 성능이 좋습니다:

```python
# 인덱스 빌드
collection.build_index(
    vectors=["embedding"],  # 인덱스할 벡터 필드
    index_type="HNSW",     # 인덱스 타입
    params={
        "M": 16,           # HNSW 파라미터
        "efConstruction": 200
    }
)
```

---

## 커밋

변경사항을磁盘에 저장:

```python
collection.commit()
```

---

## 컬렉션 닫기

사용 완료 후 반드시 닫아야 합니다:

```python
collection.close()
```

---

## 전체 예제

```python
import zvec

# 1. 스키마 정의
schema = zvec.CollectionSchema(
    name="books",
    vectors={
        "embedding": zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            128
        )
    }
)

# 2. 컬렉션 생성
collection = zvec.create_and_open("./data/books", schema=schema)

# 3. 문서 삽입
collection.insert([
    zvec.Doc(
        id="book_1",
        vectors={"embedding": [0.1] * 128},
        payload={"title": "Python Guide", "price": 30000}
    ),
    zvec.Doc(
        id="book_2",
        vectors={"embedding": [0.2] * 128},
        payload={"title": "Rust Guide", "price": 35000}
    ),
])

# 4. 인덱스 빌드
collection.build_index(vectors=["embedding"])

# 5. 커밋
collection.commit()

# 6. 닫기
collection.close()
```

---

*다음 글에서는 검색(Query) 기능에 대해 자세히 살펴보겠습니다.*
