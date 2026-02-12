---
layout: post
title: "Tambo 완벽 가이드 (09) - 자체 호스팅 및 운영 가이드"
date: 2026-02-12
permalink: /tambo-guide-09-self-host-ops/
author: Tambo AI
categories: [AI 에이전트, 웹 개발]
tags: [Tambo, 자체 호스팅, Docker, PostgreSQL, 운영]
original_url: "https://github.com/tambo-ai/tambo"
excerpt: "Docker로 Tambo 운영하기"
---

## 운영 구성(Operators Guide)

OPERATORS.md는 Tambo 운영을 위해 아래 서비스를 기준으로 설명합니다.

- Web(Next.js) 기본 포트 3210
- API(NestJS) 기본 포트 3211
- PostgreSQL 17 기본 포트 5433

`docker-compose.yml`에는 부가적으로 **MinIO(S3 호환 스토리지)**도 포함되어 있습니다(9000/9001).

---

## Docker 빠른 시작(OPERATORS.md)

```bash
git clone https://github.com/tambo-ai/tambo.git
cd tambo

./scripts/cloud/tambo-setup.sh
./scripts/cloud/tambo-start.sh
./scripts/cloud/init-database.sh
```

---

## 최소 필수 환경변수(OPERATORS.md)

`docker.env`에 최소로 필요한 값들이 명시되어 있습니다. 대표적으로:

- `POSTGRES_PASSWORD`
- `API_KEY_SECRET` (32자 이상)
- `PROVIDER_KEY_SECRET` (32자 이상)
- `NEXTAUTH_SECRET`
- `FALLBACK_OPENAI_API_KEY`

---

## 운영 팁

- OAuth/이메일 로그인 설정을 하지 않으면 대시보드 로그인 자체가 막힐 수 있습니다(OPERATORS.md).
- HTTPS 종료는 reverse proxy(nginx/Caddy/Traefik)를 두는 형태를 권장합니다.

---

## 마무리

Tambo는 "LLM + UI"를 제품에서 현실적으로 운영하기 위한 구성 요소를 넓게 제공합니다.
- 프론트엔드(React)에서는 컴포넌트를 등록하고 UI 경험을 설계
- 백엔드에서는 상태/에이전트 루프/통합(MCP)과 운영(Cloud or self-host)을 처리

다음 단계:
- Docs: https://docs.tambo.co
- Repo: https://github.com/tambo-ai/tambo
