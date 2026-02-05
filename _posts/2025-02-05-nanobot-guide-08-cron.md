---
layout: post
title: "Nanobot 완벽 가이드 (8) - Cron & Heartbeat"
date: 2025-02-05
permalink: /nanobot-guide-08-cron/
author: HKUDS
categories: [AI 에이전트, Nanobot]
tags: [Nanobot, Cron, Heartbeat, Scheduling, Automation]
original_url: "https://github.com/HKUDS/nanobot"
excerpt: "Nanobot의 스케줄링 및 주기적 작업 시스템을 알아봅니다."
---

## 스케줄링 개요

Nanobot은 두 가지 스케줄링 메커니즘을 제공합니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    스케줄링 시스템                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                      Cron                            │   │
│  │           (사용자 정의 스케줄 작업)                    │   │
│  │                                                      │   │
│  │  • 특정 시간에 메시지 전송                            │   │
│  │  • 반복 작업 실행                                     │   │
│  │  • 일회성 알림                                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Heartbeat                         │   │
│  │             (자동 주기적 작업)                         │   │
│  │                                                      │   │
│  │  • HEARTBEAT.md 기반 자동 실행                        │   │
│  │  • 상태 체크, 보고서 생성                             │   │
│  │  • 프로액티브 알림                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Cron 시스템

### CLI 명령어

```bash
# 작업 추가 - Cron 표현식
nanobot cron add --name "morning" --message "Good morning!" --cron "0 9 * * *"

# 작업 추가 - 간격 (초 단위)
nanobot cron add --name "hourly" --message "Hourly check" --every 3600

# 작업 추가 - 일회성
nanobot cron add --name "reminder" --message "Meeting!" --at "2025-02-05T15:00:00"

# 작업 목록
nanobot cron list

# 작업 제거
nanobot cron remove <job_id>
```

### Cron 표현식

```
┌───────────── 분 (0-59)
│ ┌─────────── 시 (0-23)
│ │ ┌───────── 일 (1-31)
│ │ │ ┌─────── 월 (1-12)
│ │ │ │ ┌───── 요일 (0-6, 일=0)
│ │ │ │ │
* * * * *
```

**예시:**

| 표현식 | 설명 |
|--------|------|
| `0 9 * * *` | 매일 오전 9시 |
| `0 9 * * 1-5` | 평일 오전 9시 |
| `0 */2 * * *` | 2시간마다 |
| `30 8 1 * *` | 매월 1일 오전 8:30 |
| `0 0 * * 0` | 매주 일요일 자정 |

---

## CronManager 구현

```python
# cron/manager.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date import DateTrigger

class CronManager:
    """Cron 작업 관리자"""

    def __init__(self, bus: MessageBus, storage_path: Path):
        self.bus = bus
        self.storage_path = storage_path
        self.scheduler = AsyncIOScheduler()
        self.jobs: dict[str, CronJob] = {}

    async def start(self) -> None:
        """스케줄러 시작"""
        # 저장된 작업 로드
        self._load_jobs()

        # 스케줄러 시작
        self.scheduler.start()
        logger.info("Cron manager started")

    async def stop(self) -> None:
        """스케줄러 중지"""
        self.scheduler.shutdown()

    def add_job(
        self,
        name: str,
        message: str,
        cron: str | None = None,
        every: int | None = None,
        at: str | None = None,
        channel: str = "telegram",
        chat_id: str | None = None,
    ) -> str:
        """작업 추가"""
        job_id = str(uuid.uuid4())[:8]

        # 트리거 생성
        if cron:
            trigger = CronTrigger.from_crontab(cron)
        elif every:
            trigger = IntervalTrigger(seconds=every)
        elif at:
            trigger = DateTrigger(run_date=datetime.fromisoformat(at))
        else:
            raise ValueError("One of cron, every, or at is required")

        # 작업 등록
        self.scheduler.add_job(
            self._execute_job,
            trigger=trigger,
            id=job_id,
            kwargs={
                "job_id": job_id,
                "message": message,
                "channel": channel,
                "chat_id": chat_id,
            }
        )

        # 메타데이터 저장
        self.jobs[job_id] = CronJob(
            id=job_id,
            name=name,
            message=message,
            cron=cron,
            every=every,
            at=at,
            channel=channel,
            chat_id=chat_id,
        )
        self._save_jobs()

        return job_id

    def remove_job(self, job_id: str) -> bool:
        """작업 제거"""
        if job_id not in self.jobs:
            return False

        self.scheduler.remove_job(job_id)
        del self.jobs[job_id]
        self._save_jobs()

        return True

    def list_jobs(self) -> list[CronJob]:
        """작업 목록"""
        return list(self.jobs.values())

    async def _execute_job(
        self,
        job_id: str,
        message: str,
        channel: str,
        chat_id: str | None,
    ) -> None:
        """작업 실행"""
        logger.info(f"Executing cron job {job_id}")

        # 메시지 버스로 전송
        await self.bus.publish_inbound(InboundMessage(
            channel=channel,
            chat_id=chat_id or "default",
            user_id="cron",
            content=message,
        ))

    def _save_jobs(self) -> None:
        """작업 저장"""
        data = {
            job_id: asdict(job)
            for job_id, job in self.jobs.items()
        }
        self.storage_path.write_text(json.dumps(data, indent=2))

    def _load_jobs(self) -> None:
        """작업 로드"""
        if not self.storage_path.exists():
            return

        data = json.loads(self.storage_path.read_text())
        for job_id, job_data in data.items():
            self.add_job(**job_data)
```

---

## CronJob 데이터 클래스

```python
# cron/manager.py

@dataclass
class CronJob:
    """Cron 작업 정의"""
    id: str
    name: str
    message: str
    cron: str | None = None
    every: int | None = None
    at: str | None = None
    channel: str = "telegram"
    chat_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
```

---

## Heartbeat 시스템

### HEARTBEAT.md

워크스페이스의 `HEARTBEAT.md` 파일로 주기적 작업을 정의합니다.

```markdown
# Heartbeat Configuration

## Schedule

Run every: 1 hour

## Tasks

### Market Check
Check the current market status and notify if there are significant changes.

### News Summary
Summarize the top 5 news articles for today.

### System Status
Check system resources and report any issues.

## Conditions

Only run between 9 AM and 6 PM on weekdays.
```

### HeartbeatManager 구현

```python
# heartbeat/manager.py

class HeartbeatManager:
    """Heartbeat 관리자"""

    def __init__(
        self,
        workspace: Path,
        bus: MessageBus,
        agent_loop: AgentLoop,
    ):
        self.workspace = workspace
        self.bus = bus
        self.agent_loop = agent_loop
        self.config = self._load_config()

    def _load_config(self) -> HeartbeatConfig:
        """HEARTBEAT.md 파싱"""
        heartbeat_md = self.workspace / "HEARTBEAT.md"

        if not heartbeat_md.exists():
            return HeartbeatConfig(enabled=False)

        content = heartbeat_md.read_text()

        # 간단한 파싱 (실제로는 더 정교한 파싱 필요)
        config = HeartbeatConfig(enabled=True)

        # "Run every: X hour" 파싱
        if match := re.search(r"Run every:\s*(\d+)\s*(hour|minute)", content):
            value = int(match.group(1))
            unit = match.group(2)
            config.interval_seconds = value * (3600 if unit == "hour" else 60)

        # Tasks 섹션 파싱
        tasks_section = re.search(r"## Tasks\n(.*?)(?=\n## |$)", content, re.DOTALL)
        if tasks_section:
            config.tasks = self._parse_tasks(tasks_section.group(1))

        return config

    def _parse_tasks(self, tasks_content: str) -> list[str]:
        """태스크 목록 파싱"""
        tasks = []
        current_task = []

        for line in tasks_content.split("\n"):
            if line.startswith("### "):
                if current_task:
                    tasks.append("\n".join(current_task))
                current_task = [line[4:]]  # ### 제거
            elif line.strip():
                current_task.append(line)

        if current_task:
            tasks.append("\n".join(current_task))

        return tasks

    async def start(self) -> None:
        """Heartbeat 시작"""
        if not self.config.enabled:
            logger.info("Heartbeat disabled")
            return

        asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Heartbeat started (interval: {self.config.interval_seconds}s)")

    async def _heartbeat_loop(self) -> None:
        """Heartbeat 루프"""
        while True:
            await asyncio.sleep(self.config.interval_seconds)

            if not self._should_run():
                continue

            for task in self.config.tasks:
                await self._execute_task(task)

    def _should_run(self) -> bool:
        """실행 조건 체크"""
        now = datetime.now()

        # 평일 9-18시만 실행 (예시)
        if now.weekday() >= 5:  # 주말
            return False
        if not (9 <= now.hour < 18):
            return False

        return True

    async def _execute_task(self, task: str) -> None:
        """태스크 실행"""
        logger.info(f"Executing heartbeat task: {task[:50]}...")

        # 에이전트에게 태스크 전달
        await self.bus.publish_inbound(InboundMessage(
            channel="heartbeat",
            chat_id="system",
            user_id="heartbeat",
            content=f"[Heartbeat Task] {task}",
        ))
```

---

## HeartbeatConfig 데이터 클래스

```python
# heartbeat/manager.py

@dataclass
class HeartbeatConfig:
    """Heartbeat 설정"""
    enabled: bool = False
    interval_seconds: int = 3600  # 기본 1시간
    tasks: list[str] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)
```

---

## 실용적인 사용 예시

### 1. 일일 브리핑

```bash
nanobot cron add \
  --name "daily_briefing" \
  --message "오늘의 일정과 중요 뉴스를 알려줘" \
  --cron "0 8 * * 1-5"
```

### 2. 시스템 모니터링

**HEARTBEAT.md:**

```markdown
# System Heartbeat

## Schedule
Run every: 30 minute

## Tasks

### Server Health
Check CPU, memory, and disk usage. Alert if any exceed 80%.

### API Status
Ping all external APIs and report any failures.

### Log Analysis
Analyze recent logs for errors or warnings.
```

### 3. 주식 알림

```bash
nanobot cron add \
  --name "market_open" \
  --message "주식 시장이 열렸습니다. 오늘의 관심 종목을 분석해주세요." \
  --cron "0 9 * * 1-5"

nanobot cron add \
  --name "market_close" \
  --message "오늘 시장 마감 요약을 알려주세요." \
  --cron "30 15 * * 1-5"
```

### 4. 리마인더

```bash
# 회의 10분 전 알림
nanobot cron add \
  --name "meeting_reminder" \
  --message "10분 후 회의가 있습니다!" \
  --at "2025-02-05T14:50:00"
```

---

## Cron vs Heartbeat

| 특징 | Cron | Heartbeat |
|------|------|-----------|
| **설정 방식** | CLI 명령어 | HEARTBEAT.md 파일 |
| **유연성** | 높음 (다양한 스케줄) | 단일 간격 |
| **용도** | 사용자 정의 알림 | 자동화된 모니터링 |
| **조건부 실행** | ❌ | ✅ |
| **복잡한 태스크** | 단순 메시지 | 복잡한 지시문 |

---

## 고급 설정

### 타임존 설정

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone

scheduler = AsyncIOScheduler(timezone=timezone('Asia/Seoul'))
```

### 작업 지속성

```python
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

jobstores = {
    'default': SQLAlchemyJobStore(url='sqlite:///jobs.db')
}

scheduler = AsyncIOScheduler(jobstores=jobstores)
```

### 에러 핸들링

```python
def job_error_listener(event):
    if event.exception:
        logger.error(f"Job {event.job_id} failed: {event.exception}")
        # 알림 전송

scheduler.add_listener(
    job_error_listener,
    EVENT_JOB_ERROR
)
```

---

## 워크플로우 통합

```
┌─────────────────────────────────────────────────────────────┐
│                    전체 워크플로우                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────┐      ┌─────────┐      ┌─────────────────┐     │
│  │  Cron   │─────►│         │      │                 │     │
│  └─────────┘      │ Message │─────►│   Agent Loop    │     │
│  ┌─────────┐      │   Bus   │      │                 │     │
│  │Heartbeat│─────►│         │      │  • LLM 호출     │     │
│  └─────────┘      └─────────┘      │  • 도구 실행    │     │
│                                     │  • 응답 생성    │     │
│                                     └────────┬────────┘     │
│                                              │               │
│                                              ▼               │
│                                     ┌─────────────────┐     │
│                                     │    Channels     │     │
│                                     │  (응답 전송)    │     │
│                                     └─────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*다음 글에서는 Providers 시스템을 분석합니다.*
