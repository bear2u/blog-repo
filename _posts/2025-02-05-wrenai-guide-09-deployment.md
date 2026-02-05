---
layout: post
title: "WrenAI 완벽 가이드 (9) - 배포 가이드"
date: 2025-02-05
permalink: /wrenai-guide-09-deployment/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, Docker, Kubernetes, Deployment, DevOps]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI를 Docker Compose와 Kubernetes로 배포하는 방법을 안내합니다."
---

## 배포 방식 비교

| 방식 | 용도 | 확장성 | 복잡도 |
|------|------|--------|--------|
| **Docker Compose** | 로컬, 소규모 | 제한적 | 낮음 |
| **Kubernetes** | 프로덕션, 대규모 | 높음 | 높음 |
| **Wren Launcher** | 빠른 시작 | 제한적 | 매우 낮음 |

---

## Docker Compose 배포

### 기본 구성

```yaml
# docker/docker-compose.yaml

version: '3.8'

services:
  bootstrap:
    image: busybox
    volumes:
      - data:/app/data
    command: >
      sh -c "
        mkdir -p /app/data/mdl /app/data/db
        chmod -R 777 /app/data
      "

  wren-engine:
    image: ghcr.io/canner/wren-engine:latest
    ports:
      - "8080:8080"
    volumes:
      - data:/app/data
    networks:
      - wren
    depends_on:
      - bootstrap

  ibis-server:
    image: ghcr.io/canner/wren-ibis:latest
    ports:
      - "8000:8000"
    environment:
      - WREN_ENGINE_ENDPOINT=http://wren-engine:8080
    networks:
      - wren
    depends_on:
      - wren-engine

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - wren

  wren-ai-service:
    image: ghcr.io/canner/wren-ai-service:latest
    ports:
      - "5555:5555"
    volumes:
      - ./config.yaml:/app/config.yaml
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - WREN_AI_SERVICE_HOST=0.0.0.0
      - WREN_AI_SERVICE_PORT=5555
    networks:
      - wren
    depends_on:
      - qdrant
      - wren-engine

  wren-ui:
    image: ghcr.io/canner/wren-ui:latest
    ports:
      - "3000:3000"
    volumes:
      - data:/app/data
    environment:
      - WREN_ENGINE_ENDPOINT=http://wren-engine:8080
      - WREN_AI_ENDPOINT=http://wren-ai-service:5555
      - IBIS_SERVER_ENDPOINT=http://ibis-server:8000
      - DB_TYPE=sqlite
      - SQLITE_FILE=/app/data/db/db.sqlite3
    networks:
      - wren
    depends_on:
      - wren-ai-service

volumes:
  data:
  qdrant_data:

networks:
  wren:
    driver: bridge
```

### 배포 명령

```bash
cd docker

# 환경 파일 설정
cp .env.example .env.local
cp config.example.yaml config.yaml

# API 키 설정
echo "OPENAI_API_KEY=sk-your-key" >> .env.local

# 시작
docker compose --env-file .env.local up -d

# 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f

# 중지
docker compose down

# 완전 삭제 (데이터 포함)
docker compose down -v
```

---

## PostgreSQL 사용 (프로덕션)

```yaml
# docker-compose.prod.yaml

services:
  postgres:
    image: postgres:15
    environment:
      - POSTGRES_USER=wrenai
      - POSTGRES_PASSWORD=${PG_PASSWORD}
      - POSTGRES_DB=wrenai
    volumes:
      - pg_data:/var/lib/postgresql/data
    networks:
      - wren

  wren-ui:
    image: ghcr.io/canner/wren-ui:latest
    environment:
      - DB_TYPE=pg
      - PG_HOST=postgres
      - PG_PORT=5432
      - PG_USER=wrenai
      - PG_PASSWORD=${PG_PASSWORD}
      - PG_DATABASE=wrenai
    depends_on:
      - postgres
    # ... 기타 설정

volumes:
  pg_data:
```

---

## Kubernetes 배포

### Namespace 생성

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: wrenai
```

### ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: wren-config
  namespace: wrenai
data:
  config.yaml: |
    type: llm
    provider: litellm_llm
    models:
      - alias: default
        model: gpt-4o-mini
        context_window_size: 128000
        kwargs:
          temperature: 0

    type: embedder
    provider: litellm_embedder
    models:
      - model: text-embedding-3-large
        alias: default
        dimension: 3072

    type: document_store
    provider: qdrant
    location: http://qdrant:6333
    embedding_model_dim: 3072
```

### Secret

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: wren-secrets
  namespace: wrenai
type: Opaque
stringData:
  OPENAI_API_KEY: "sk-your-key-here"
  PG_PASSWORD: "your-pg-password"
```

### Qdrant 배포

```yaml
# qdrant.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: wrenai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
        - name: qdrant
          image: qdrant/qdrant:latest
          ports:
            - containerPort: 6333
            - containerPort: 6334
          volumeMounts:
            - name: qdrant-storage
              mountPath: /qdrant/storage
      volumes:
        - name: qdrant-storage
          persistentVolumeClaim:
            claimName: qdrant-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant
  namespace: wrenai
spec:
  selector:
    app: qdrant
  ports:
    - name: http
      port: 6333
    - name: grpc
      port: 6334
```

### AI Service 배포

```yaml
# ai-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wren-ai-service
  namespace: wrenai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wren-ai-service
  template:
    metadata:
      labels:
        app: wren-ai-service
    spec:
      containers:
        - name: ai-service
          image: ghcr.io/canner/wren-ai-service:latest
          ports:
            - containerPort: 5555
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wren-secrets
                  key: OPENAI_API_KEY
            - name: WREN_AI_SERVICE_HOST
              value: "0.0.0.0"
            - name: WREN_AI_SERVICE_PORT
              value: "5555"
          volumeMounts:
            - name: config
              mountPath: /app/config.yaml
              subPath: config.yaml
          resources:
            requests:
              memory: "512Mi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2000m"
      volumes:
        - name: config
          configMap:
            name: wren-config
---
apiVersion: v1
kind: Service
metadata:
  name: wren-ai-service
  namespace: wrenai
spec:
  selector:
    app: wren-ai-service
  ports:
    - port: 5555
```

### UI 배포

```yaml
# ui.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wren-ui
  namespace: wrenai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wren-ui
  template:
    metadata:
      labels:
        app: wren-ui
    spec:
      containers:
        - name: ui
          image: ghcr.io/canner/wren-ui:latest
          ports:
            - containerPort: 3000
          env:
            - name: WREN_ENGINE_ENDPOINT
              value: "http://wren-engine:8080"
            - name: WREN_AI_ENDPOINT
              value: "http://wren-ai-service:5555"
            - name: DB_TYPE
              value: "pg"
            - name: PG_HOST
              value: "postgres"
            - name: PG_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: wren-secrets
                  key: PG_PASSWORD
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: wren-ui
  namespace: wrenai
spec:
  selector:
    app: wren-ui
  ports:
    - port: 3000
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: wren-ingress
  namespace: wrenai
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
spec:
  ingressClassName: nginx
  rules:
    - host: wrenai.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: wren-ui
                port:
                  number: 3000
```

### 배포 명령

```bash
# 네임스페이스 생성
kubectl apply -f namespace.yaml

# 설정 적용
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

# 서비스 배포
kubectl apply -f qdrant.yaml
kubectl apply -f ai-service.yaml
kubectl apply -f ui.yaml
kubectl apply -f ingress.yaml

# 상태 확인
kubectl get pods -n wrenai
kubectl get services -n wrenai

# 로그 확인
kubectl logs -f deployment/wren-ai-service -n wrenai
```

---

## 환경별 설정

### 개발 환경

```yaml
settings:
  logging_level: DEBUG
  development: true
  query_cache_ttl: 60  # 1분
```

### 프로덕션 환경

```yaml
settings:
  logging_level: INFO
  development: false
  query_cache_ttl: 3600  # 1시간
  langfuse_enable: true
```

---

## 헬스체크

```bash
# AI Service
curl http://localhost:5555/health

# UI (GraphQL)
curl http://localhost:3000/api/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ __typename }"}'

# Qdrant
curl http://localhost:6333/collections
```

---

## 백업 및 복구

### 데이터 백업

```bash
# SQLite 백업
docker compose exec wren-ui cp /app/data/db/db.sqlite3 /app/data/db/backup.sqlite3

# Qdrant 스냅샷
curl -X POST http://localhost:6333/collections/table_schema/snapshots

# PostgreSQL 백업
docker compose exec postgres pg_dump -U wrenai wrenai > backup.sql
```

### 복구

```bash
# SQLite 복구
docker compose exec wren-ui cp /app/data/db/backup.sqlite3 /app/data/db/db.sqlite3

# PostgreSQL 복구
cat backup.sql | docker compose exec -T postgres psql -U wrenai wrenai
```

---

*다음 글에서는 확장 및 커스터마이징 방법을 살펴봅니다.*
