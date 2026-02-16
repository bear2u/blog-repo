---
layout: post
title: "zvec 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-16
permalink: /zvec-guide-02-installation/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, Installation, Quickstart]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec 설치 방법: Python pip, Node.js npm, 소스 빌드까지 단계별로 안내합니다."
---

## 설치 방법

zvec는 다양한 설치 방법을 제공합니다. 가장 빠른 방법은 pip 또는 npm을 사용하는 것입니다.

---

## Python 설치

### 요구사항

- Python 3.10 - 3.12

### pip 설치

```bash
pip install zvec
```

### 특정 버전 설치

```bash
pip install zvec==0.1.0
```

---

## Node.js 설치

### npm 설치

```bash
npm install @zvec/zvec
```

---

## 소스 코드에서 빌드

소스 코드에서 직접 빌드하려면 [공식 빌드 가이드](https://zvec.org/en/docs/build/)를 참고하세요.

### 주요 빌드 단계

```bash
# 레포지토리 클론
git clone https://github.com/alibaba/zvec.git
cd zvec

# CMake 빌드
mkdir build && cd build
cmake ..
make

# Python 바인딩 빌드
cd python
pip install -e .
```

---

## 1분 퀵스타트

설치가 완료되었다면, 다음 예제로 zvec를 즉시 사용해볼 수 있습니다:

```python
import zvec

# 컬렉션 스키마 정의
schema = zvec.CollectionSchema(
    name="example",
    vectors=zvec.VectorSchema("embedding", zvec.DataType.VECTOR_FP32, 4),
)

# 컬렉션 생성
collection = zvec.create_and_open(path="./zvec_example", schema=schema)

# 문서 삽입
collection.insert([
    zvec.Doc(id="doc_1", vectors={"embedding": [0.1, 0.2, 0.3, 0.4]}),
    zvec.Doc(id="doc_2", vectors={"embedding": [0.2, 0.3, 0.4, 0.1]}),
])

# 벡터 유사도로 검색
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

# 결과 출력
print(results)
```

### 코드 설명

| 단계 | 설명 |
|------|------|
| `CollectionSchema` | 컬렉션의 스키마 정의 |
| `create_and_open()` |磁盘에 컬렉션 생성 및 열기 |
| `insert()` | 문서(벡터) 삽입 |
| `query()` | 유사도 검색 수행 |

---

## 실행 결과 예시

```
[
    {'id': 'doc_2', 'score': 0.98, 'vectors': {'embedding': [0.2, 0.3, 0.4, 0.1]}},
    {'id': 'doc_1', 'score': 0.85, 'vectors': {'embedding': [0.1, 0.2, 0.3, 0.4]}}
]
```

---

## 프로젝트 구조

설치 후 생성되는 주요 파일과 디렉토리:

```
zvec_example/
├── data/           # 벡터 데이터
├── index/          # 인덱스 파일
└── meta/           # 메타데이터
```

---

## 다음 단계

이제 zvec의 기본 사용법을 알았습니다. 다음 글에서는 zvec의 핵심 API에 대해 더 자세히 살펴보겠습니다.

---

*다음 글에서는 zvec의 핵심 API (Collection, Vector, Document)를 살펴보겠습니다.*
