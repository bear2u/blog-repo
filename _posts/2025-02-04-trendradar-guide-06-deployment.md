---
layout: post
title: "TrendRadar 완벽 가이드 (6) - 배포 및 활용"
date: 2025-02-04
permalink: /trendradar-guide-06-deployment/
author: sansan0
category: AI
tags: [TrendRadar, Deployment, Docker, GitHub Actions]
series: trendradar-guide
part: 6
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "TrendRadar의 다양한 배포 방법과 실전 활용 가이드입니다."
---

## 배포 옵션 개요

TrendRadar는 다양한 배포 방식을 지원합니다.

| 방식 | 난이도 | 비용 | 특징 |
|------|--------|------|------|
| **GitHub Actions** | ⭐ | 무료 | 30초 배포, 권장 |
| **Docker** | ⭐⭐ | 서버 비용 | 커스터마이징 용이 |
| **로컬 설치** | ⭐⭐ | 무료 | 개발/테스트용 |
| **GitHub Pages** | ⭐ | 무료 | 정적 웹 리포트 |

---

## 방법 1: GitHub Actions (권장)

가장 쉬운 배포 방법입니다. **30초 만에** 설정 완료!

### 단계별 가이드

```bash
# 1. 레포지토리 Fork
# GitHub에서 Fork 버튼 클릭

# 2. Settings > Secrets and variables > Actions
# 필요한 시크릿 추가:
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_CHAT_ID
# - OPENAI_API_KEY (선택)

# 3. Actions 탭에서 워크플로우 활성화
# "I understand my workflows, go ahead and enable them" 클릭

# 4. 자동으로 1시간마다 실행됨!
```

### 워크플로우 파일

{% raw %}
```yaml
# .github/workflows/news.yml

name: News Update

on:
  schedule:
    - cron: '0 * * * *'  # 매시 정각
  workflow_dispatch:      # 수동 실행

jobs:
  update:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run TrendRadar
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python -m trendradar --once

      - name: Commit changes
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add -A
          git diff --staged --quiet || git commit -m "Update news"
          git push
```
{% endraw %}

### 커스텀 스케줄

```yaml
# 더 자주 실행 (15분마다)
schedule:
  - cron: '*/15 * * * *'

# 특정 시간대만 (한국 시간 오전 9시-오후 9시)
schedule:
  - cron: '0 0-12 * * *'  # UTC 기준
```

---

## 방법 2: Docker

### 기본 실행

```bash
# 이미지 풀
docker pull wantcat/trendradar

# 실행
docker run -d \
  --name trendradar \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_chat_id \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output:/app/output \
  wantcat/trendradar
```

### Docker Compose

```yaml
# docker-compose.yml

version: '3.8'

services:
  trendradar:
    image: wantcat/trendradar
    container_name: trendradar
    restart: unless-stopped
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./config:/app/config
      - ./output:/app/output
      - ./data:/app/data
```

```bash
# 환경 변수 파일
echo "TELEGRAM_BOT_TOKEN=your_token" > .env
echo "TELEGRAM_CHAT_ID=your_chat_id" >> .env

# 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 직접 빌드

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "trendradar"]
```

```bash
docker build -t trendradar .
docker run -d --env-file .env trendradar
```

---

## 방법 3: 로컬 설치

### Linux/macOS

```bash
# 클론
git clone https://github.com/sansan0/TrendRadar.git
cd TrendRadar

# 가상 환경 생성
python -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 설정 파일 복사 및 편집
cp config/config.example.yaml config/config.yaml
nano config/config.yaml

# 실행
python -m trendradar
```

### Windows

```batch
# setup-windows.bat 실행
setup-windows.bat

# 또는 수동 설치
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 실행
python -m trendradar
```

### 자동 실행 설정 (Linux)

```bash
# systemd 서비스 생성
sudo nano /etc/systemd/system/trendradar.service
```

```ini
[Unit]
Description=TrendRadar News Monitor
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/TrendRadar
ExecStart=/path/to/TrendRadar/venv/bin/python -m trendradar
Restart=always
RestartSec=60
Environment=TELEGRAM_BOT_TOKEN=your_token
Environment=TELEGRAM_CHAT_ID=your_chat_id

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable trendradar
sudo systemctl start trendradar
```

---

## 방법 4: GitHub Pages (웹 리포트)

정적 HTML 리포트를 GitHub Pages로 호스팅합니다.

### 설정

```yaml
# config/config.yaml

report:
  enabled: true
  format: html
  output_dir: docs  # GitHub Pages 기본 디렉토리
```

### 워크플로우

{% raw %}
```yaml
# .github/workflows/pages.yml

name: Deploy Report

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 */6 * * *'  # 6시간마다

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Generate Report
        run: python -m trendradar --report-only

      - name: Deploy to Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```
{% endraw %}

---

## 설정 최적화

### 성능 튜닝

```yaml
# config/config.yaml

performance:
  # 동시 크롤링 수
  max_concurrent_crawlers: 5

  # HTTP 타임아웃
  http_timeout: 30

  # 재시도 설정
  retry_count: 3
  retry_delay: 5

  # 캐시 설정
  cache_ttl: 3600  # 1시간
```

### 알림 최적화

```yaml
# 과도한 알림 방지
notifications:
  rate_limit:
    max_items_per_push: 10  # 한 번에 최대 10개
    min_interval: 600       # 최소 10분 간격
    quiet_hours:            # 방해 금지 시간
      start: "23:00"
      end: "08:00"
      timezone: "Asia/Seoul"
```

### 필터링 강화

```yaml
filters:
  # 키워드 필터
  include:
    - AI
    - LLM
    - Claude

  exclude:
    - 광고
    - sponsored

  # 중복 판단 기준
  dedup:
    similarity_threshold: 0.8  # 80% 유사도 이상 중복
    time_window: 86400         # 24시간 내
```

---

## 모니터링

### 헬스 체크

```python
# 간단한 헬스 체크 엔드포인트
from aiohttp import web

async def health_check(request):
    return web.json_response({
        "status": "ok",
        "last_run": last_run_time.isoformat(),
        "items_processed": total_items,
    })

app = web.Application()
app.router.add_get('/health', health_check)
```

### 로그 설정

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/trendradar.log
  max_size: 10MB
  backup_count: 5
```

---

## 트러블슈팅

### 일반적인 문제

| 문제 | 해결책 |
|------|--------|
| 알림이 오지 않음 | 봇 토큰/채팅 ID 확인 |
| 뉴스가 수집되지 않음 | 네트워크 연결 확인 |
| AI 분석 실패 | API 키 및 잔액 확인 |
| 중복 알림 | 캐시/DB 파일 확인 |

### 디버그 모드

```bash
# 상세 로그 출력
python -m trendradar --debug

# 단일 실행 (스케줄 무시)
python -m trendradar --once

# 드라이런 (알림 전송 안함)
python -m trendradar --dry-run
```

---

## 활용 시나리오

### 1. 개인 뉴스 큐레이션

- Telegram 봇으로 관심 분야 뉴스 구독
- AI 요약으로 빠른 파악
- 출퇴근 시간에 알림 수신

### 2. 팀 정보 공유

- Slack 채널에 기술 뉴스 공유
- 주간 리포트 자동 생성
- MCP로 Claude와 뉴스 대화

### 3. 리서치 자동화

- 경쟁사 동향 모니터링
- 키워드 기반 뉴스 필터링
- 웹훅으로 자체 시스템 연동

---

## 마무리

TrendRadar는 **30초 배포**로 시작할 수 있는 강력한 트렌드 모니터링 도구입니다. 핵심 장점:

- **경량화** - 최소 리소스로 동작
- **다중 채널** - 10개 이상 알림 채널
- **AI 분석** - LLM 기반 요약/번역
- **MCP 통합** - AI 에이전트와 연동
- **GitHub Actions** - 무료 자동 실행

---

## 리소스

- **GitHub**: [github.com/sansan0/TrendRadar](https://github.com/sansan0/TrendRadar)
- **Docker Hub**: [wantcat/trendradar](https://hub.docker.com/r/wantcat/trendradar)
- **라이선스**: GPL-3.0

---

*이 가이드 시리즈가 TrendRadar를 활용하는 데 도움이 되길 바랍니다.*
