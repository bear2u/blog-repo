---
layout: page
title: zvec 가이드
permalink: /zvec-guide/
icon: fas fa-database
---

# zvec 완벽 가이드

> *Alibaba의 인프로세스 벡터 데이터베이스 - 경량, 초고속, 애플리케이션에 직접 임베드*

**zvec**는 Alibaba가 개발한 인프로세스(in-process) 벡터 데이터베이스입니다. Alibaba의 검증된 벡터 검색 엔진 Proxima를 기반으로 하며, 최소한의 설정으로 생산 수준의 low-latency, 확장 가능한 유사도 검색을 제공합니다.

---

## 목차

| # | 제목 | 내용 |
|---|------|------|
| 01 | [소개](/blog-repo/zvec-guide-01-intro/) | zvec란? 주요 특징, 아키텍처 |
| 02 | [설치](/blog-repo/zvec-guide-02-installation/) | pip, npm 설치, 빠른 시작 |
| 03 | [핵심 API](/blog-repo/zvec-guide-03-core-api/) | Collection, Document, Vector |
| 04 | [검색 기능](/blog-repo/zvec-guide-04-search/) | 유사도 검색, 필터링, top-k |
| 05 | [하이브리드 검색](/blog-repo/zvec-guide-05-hybrid-search/) | Dense + Sparse 벡터 결합 |
| 06 | [성능 최적화](/blog-repo/zvec-guide-06-optimization/) | 인덱스, 배치 처리, 메모리 |
| 07 | [확장 및 운영](/blog-repo/zvec-guide-07-advanced/) | 백업, 모니터링, 프로덕션 |

---

## 주요 특징

- **초고속 검색**: 수십억 개의 벡터를 밀리초 단위로 검색
- **간단한 사용**: 서버/설정 없이 pip 설치만으로 즉시 사용
- **Dense + Sparse**: 밀도와 희소 벡터 모두 지원
- **하이브리드 검색**: 의미적 유사성과 구조화된 필터 결합
- **경량**: 인프로세스 라이브러리로 어디서나 실행

---

## 빠른 시작

```bash
# Python 설치
pip install zvec

# Node.js 설치
npm install @zvec/zvec
```

### 1분 예제

```python
import zvec

# 스키마 정의
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

# 검색
results = collection.query(
    zvec.VectorQuery("embedding", vector=[0.4, 0.3, 0.3, 0.1]),
    topk=10
)

print(results)
```

---

## 기술 스택

| 기술 | 용도 |
|------|------|
| C++ | 핵심 엔진 |
| Python 3.10-3.12 | Python SDK |
| Node.js | Node.js SDK |
| Proxima | Alibaba 벡터 검색 엔진 |

---

## 지원 플랫폼

| 플랫폼 | 아키텍처 |
|--------|---------|
| Linux | x86_64, ARM64 |
| macOS | ARM64 |

---

## 관련 링크

- [공식 웹사이트](https://zvec.org/en/)
- [문서](https://zvec.org/en/docs/)
- [벤치마크](https://zvec.org/en/docs/benchmarks/)
- [GitHub](https://github.com/alibaba/zvec)
- [Discord](https://discord.gg/rKddFBBu9z)
- [X (Twitter)](https://x.com/zvec_ai)
