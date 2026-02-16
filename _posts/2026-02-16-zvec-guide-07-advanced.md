---
layout: post
title: "zvec 완벽 가이드 (07) - 확장 및 운영"
date: 2026-02-16
permalink: /zvec-guide-07-advanced/
author: Alibaba
categories: [개발 도구]
tags: [zvec, Vector Database, Production, Deployment, Scaling]
original_url: "https://github.com/alibaba/zvec"
excerpt: "zvec를 프로덕션 환경에서 운영할 때 필요한 확장, 백업, 모니터링 등의 고급 기능을 알아봅니다."
---

## 프로덕션 운영 개요

zvec를 프로덕션 환경에서 운영하려면 다음 사항들을 고려해야 합니다.

---

## 백업 및 복원

### 백업

```python
import shutil
import os

def backup_collection(source_path, backup_path):
    """컬렉션 백업"""
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(source_path, backup_path)
    print(f"백업 완료: {backup_path}")

# 백업 실행
backup_collection("./data/production", "./backup/production_20260216")
```

### 복원

```python
def restore_collection(backup_path, target_path):
    """백업에서 복원"""
    shutil.copytree(backup_path, target_path)
    collection = zvec.open(target_path)
    return collection

# 복원 실행
collection = restore_collection("./backup/production_20260216", "./data/production")
```

---

## 마이그레이션

### 버전 업그레이드

```python
# 새 버전으로 마이그레이션
def migrate_collection(old_path, new_path):
    """컬레전 마이그레이션"""
    # 1. 백업
    backup_collection(old_path, f"{old_path}_backup")

    # 2. 새 경로에 열기 (자동 마이그레이션)
    collection = zvec.create_and_open(
        path=new_path,
        schema=load_schema(old_path)  # 기존 스키마 로드
    )

    # 3. 데이터 확인
    info = collection.info()
    print(f"마이그레이션 완료: {info['doc_count']} 문서")

    return collection
```

---

## 모니터링

### 기본 메트릭

```python
def get_collection_stats(collection):
    """컬렉션 통계"""
    info = collection.info()

    return {
        "doc_count": info.get("doc_count", 0),
        "vector_dimension": info.get("dimension", 0),
        "index_type": info.get("index_type", "N/A"),
        "indexed": info.get("indexed", False),
        "size_mb": get_collection_size(collection.path)
    }

def get_collection_size(path):
    """디렉토리 크기 계산"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total / (1024 * 1024)  # MB

# 사용
stats = get_collection_stats(collection)
print(f"문서 수: {stats['doc_count']}")
print(f"크기: {stats['size_mb']:.2f}MB")
```

---

## 파티셔닝

대규모 데이터는 파티션으로 분할:

```python
# 파티션별 컬렉션
partitions = ["users_2024_q1", "users_2024_q2", "users_2024_q3", "users_2024_q4"]

def search_partitioned(query, partitions):
    """파티션 검색"""
    all_results = []

    for partition in partitions:
        with zvec.open(f"./data/{partition}") as collection:
            results = collection.query(vector=query, topk=10)
            all_results.extend(results)

    # 점수로 정렬
    all_results.sort(key=lambda x: x['score'], reverse=True)
    return all_results[:10]
```

---

## 클라이언트 라이브러리

### Python 고수준 API

```python
# 커넥션 풀 (필요시 구현)
class ZvecConnectionPool:
    def __init__(self, path, pool_size=4):
        self.path = path
        self.pool = queue.Queue(pool_size)

        for _ in range(pool_size):
            self.pool.put(zvec.open(path))

    def get(self):
        return self.pool.get()

    def release(self, conn):
        self.pool.put(conn)

# 사용
pool = ZvecConnectionPool("./data/production")
try:
    conn = pool.get()
    results = conn.query(vector=query)
finally:
    pool.release(conn)
```

---

## 에지 디바이스 운영

zvec는 경량이므로 에지 디바이스에서도 실행 가능:

```python
# 에지 디바이스용 최적화
schema = zvec.CollectionSchema(
    name="edge_data",
    vectors={
        "embedding": zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP16,  # FP16으로 메모리 절약
            128
        )
    }
)

# 빌드 시 메모리 최적화
collection.build_index(
    vectors=["embedding"],
    params={"M": 8}  # 작은 M으로 메모리 절약
)
```

---

## API 서버로 제공

REST API로 서빙:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
collection = None

@app.route('/init', methods=['POST'])
def init():
    global collection
    data = request.json
    collection = zvec.open(data['path'])
    return jsonify({"status": "ok"})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    results = collection.query(
        vector=zvec.VectorQuery(
            field="embedding",
            vector=data['vector'],
            topk=data.get('topk', 10)
        )
    )
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 보안

### 읽기 전용 모드

```python
# 읽기 전용으로 열기 (프로덕션 권장)
collection = zvec.open(path, read_only=True)

# 읽기 전용에서는 쓰기 작업 불가
# collection.insert(doc)  # 오류 발생
```

### 접근 제어

```python
import os

# 파일 시스템 레벨로 접근 제어
os.chmod("./data/collection", 0o600)  # 소유자만 읽기/쓰기
```

---

## 문제 해결

### 일반적인 오류

| 오류 | 원인 | 해결 |
|------|------|------|
| `Collection not found` | 경로 오류 | 경로 확인 |
| `Dimension mismatch` | 벡터 차원 불일치 | 스키마 확인 |
| `Index not built` | 인덱스 미빌드 | `build_index()` 실행 |
| `Out of memory` | 메모리 초과 | 배치 크기 축소, 인덱스 파라미터 조정 |

---

## 결론

zvec는:

- **간단**: 설치만 하면 즉시 사용 가능
- **빠른**: 수십억 벡서를 밀리초에 검색
- **유연**: 프로덕션부터 에지까지 다양한 환경 지원

더 자세한 내용은 공식 문서를 참고하세요:

- [공식 웹사이트](https://zvec.org/en/)
- [문서](https://zvec.org/en/docs/)
- [Discord](https://discord.gg/rKddFBBu9z)

---

*zvec 완벽 가이드 시리즈를 읽어주셔서 감사합니다!*
