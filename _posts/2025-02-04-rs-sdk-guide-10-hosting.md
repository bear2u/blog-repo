---
layout: post
title: "RS-SDK 가이드 - 서버 호스팅"
date: 2025-02-04
category: AI
tags: [rs-sdk, hosting, server, deployment, docker]
series: rs-sdk-guide
part: 10
author: MaxBittker
original_url: https://github.com/MaxBittker/rs-sdk
---

## 서버 호스팅 개요

RS-SDK 서버를 직접 호스팅하면 데이터 지속성, 커스터마이징, 프라이버시를 확보할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────┐
│                   Hosting Architecture                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│   │  Engine  │◀──▶│  Gateway │◀──▶│   SDK    │            │
│   │  Server  │    │  Server  │    │  Clients │            │
│   └────┬─────┘    └──────────┘    └──────────┘            │
│        │                                                    │
│        ▼                                                    │
│   ┌──────────┐                                             │
│   │ Database │                                             │
│   │(SQLite/  │                                             │
│   │ MySQL)   │                                             │
│   └──────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 로컬 서버 설정

### 1. 저장소 클론

```bash
git clone https://github.com/MaxBittker/rs-sdk.git
cd rs-sdk
bun install
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env
```

기본 환경 변수:

```bash
# .env
NODE_ENV=development
PORT=80

# 데이터베이스
DATABASE_URL=file:./dev.db

# 로그인 서버 (선택)
LOGIN_SERVER=false
LOGIN_HOST=localhost
LOGIN_PORT=43500
```

### 3. 데이터베이스 설정

```bash
cd engine

# SQLite 사용 (기본)
bun run sqlite:migrate

# 또는 MySQL 사용
bun run db:migrate
```

### 4. 콘텐츠 빌드

```bash
cd engine
bun run build
```

### 5. 서버 시작

```bash
# 엔진 서버
cd engine
bun start

# 게이트웨이 서버 (별도 터미널)
cd gateway
bun run gateway
```

## 웹 클라이언트 빌드

```bash
cd webclient

# 프로덕션 빌드
bun run build

# 엔진에 복사
cp out/standard/client.js ../engine/public/client/
cp out/bot/client.js ../engine/public/bot/
```

## 개발 모드

핫 리로드와 디버깅을 위한 개발 모드:

```bash
# 엔진 (핫 리로드)
cd engine
bun run dev

# 게이트웨이 (핫 리로드)
cd gateway
bun run gateway:dev

# 웹클라이언트 (워치 모드)
cd webclient
bun run build:dev
```

## Docker 배포

### Dockerfile

```dockerfile
FROM oven/bun:1-alpine

WORKDIR /app

# 의존성 복사 및 설치
COPY package.json bun.lockb ./
RUN bun install --frozen-lockfile

# 소스 복사
COPY . .

# 빌드
WORKDIR /app/engine
RUN bun run build

WORKDIR /app/webclient
RUN bun run build

# 클라이언트 복사
RUN cp out/standard/client.js ../engine/public/client/
RUN cp out/bot/client.js ../engine/public/bot/

WORKDIR /app/engine

EXPOSE 80

CMD ["bun", "start"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  engine:
    build: .
    ports:
      - "80:80"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=file:/data/game.db
    volumes:
      - game-data:/data
    restart: unless-stopped

  gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    ports:
      - "8080:8080"
    depends_on:
      - engine
    restart: unless-stopped

volumes:
  game-data:
```

### Docker 실행

```bash
# 이미지 빌드
docker-compose build

# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

## Fly.io 배포

### fly.toml

```toml
app = "my-rs-sdk-server"
primary_region = "nrt"  # 도쿄

[build]
  builder = "heroku/buildpacks:20"

[http_service]
  internal_port = 80
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

[mounts]
  source = "game_data"
  destination = "/data"
```

### 배포

```bash
# Fly CLI 설치
curl -L https://fly.io/install.sh | sh

# 로그인
fly auth login

# 앱 생성
fly launch

# 볼륨 생성 (데이터 지속성)
fly volumes create game_data --region nrt --size 1

# 배포
fly deploy
```

## 서버 설정 커스터마이징

### 게임 밸런스 수정

```typescript
// engine/src/lostcity/engine/World.ts

// XP 배율 조정
const XP_MULTIPLIER = 10; // 기본: 1

// 런 에너지 무제한
const UNLIMITED_RUN = true;

// 랜덤 이벤트 비활성화
const DISABLE_RANDOM_EVENTS = true;
```

### 플레이어 제한

```typescript
// 동시 접속자 제한
const MAX_PLAYERS = 100;

// 봇 계정 제한
const MAX_BOTS_PER_IP = 5;
```

### 로그인 서버 활성화

보안을 위해 로그인 서버를 활성화할 수 있습니다:

```bash
# .env
LOGIN_SERVER=true
LOGIN_HOST=0.0.0.0
LOGIN_PORT=43500
```

## 모니터링

### 상태 확인

```bash
# 서버 상태
curl http://localhost/status

# 플레이어 수
curl http://localhost/api/players/count

# 하이스코어
curl http://localhost/api/hiscores
```

### 로그

```bash
# 엔진 로그
tail -f engine/logs/server.log

# 게이트웨이 로그
tail -f gateway/logs/gateway.log

# 봇 상태
ls gateway/agent-state/
```

### 프로메테우스 메트릭 (선택)

```typescript
// 메트릭 엔드포인트 추가
import { collectDefaultMetrics, register } from 'prom-client';

collectDefaultMetrics();

app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

## 백업

### 데이터베이스 백업

```bash
# SQLite 백업
cp engine/data/game.db backups/game-$(date +%Y%m%d).db

# 자동 백업 (cron)
0 */6 * * * /path/to/backup.sh
```

### 플레이어 데이터 백업

```bash
# 플레이어 데이터 디렉토리 백업
tar -czf players-$(date +%Y%m%d).tar.gz engine/data/players/
```

## 트러블슈팅

### 포트 충돌

```bash
# 사용 중인 포트 확인
lsof -i :80
lsof -i :8080

# 프로세스 종료
kill -9 <PID>
```

### 데이터베이스 오류

```bash
# 마이그레이션 리셋
cd engine
bun run db:reset
bun run db:migrate
```

### 메모리 부족

```bash
# Node 메모리 증가
NODE_OPTIONS="--max-old-space-size=4096" bun start
```

### 연결 문제

```bash
# 방화벽 확인
sudo ufw status
sudo ufw allow 80
sudo ufw allow 8080
```

## 프로덕션 체크리스트

- [ ] HTTPS 설정 (Let's Encrypt)
- [ ] 데이터베이스 백업 자동화
- [ ] 로그 로테이션 설정
- [ ] 모니터링 알림 설정
- [ ] 리소스 제한 설정
- [ ] 보안 헤더 추가
- [ ] Rate limiting 설정

## 리소스

- **GitHub**: [github.com/MaxBittker/rs-sdk](https://github.com/MaxBittker/rs-sdk)
- **Discord**: [discord.gg/3DcuU5cMJN](https://discord.gg/3DcuU5cMJN)
- **데모 서버**: [rs-sdk-demo.fly.dev](https://rs-sdk-demo.fly.dev)
- **LostCity**: [lostcity.rs](https://lostcity.rs)

---

**이전 글**: [베스트 프랙티스](/rs-sdk-guide-09-best-practices/)

**시리즈 처음으로**: [소개](/rs-sdk-guide-01-intro/)
