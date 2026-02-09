---
layout: post
title: "effGen 완벽 가이드 (07) - 고급 활용 및 프로덕션"
date: 2026-02-09
permalink: /effgen-guide-07-advanced/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen을 프로덕션 환경에 배포하는 방법, API 서버 구축, 보안, 성능 최적화, 모니터링 가이드"
---

# effGen 완벽 가이드 (07) - 고급 활용 및 프로덕션

## 목차
1. [API 서버 구축](#api-서버-구축)
2. [Docker 샌드박스 보안](#docker-샌드박스-보안)
3. [속도 제한 및 안정성](#속도-제한-및-안정성)
4. [프롬프트 최적화 전략](#프롬프트-최적화-전략)
5. [토큰 예산 관리](#토큰-예산-관리)
6. [벡터 DB 통합](#벡터-db-통합)
7. [프로덕션 배포 체크리스트](#프로덕션-배포-체크리스트)
8. [모니터링 및 로깅](#모니터링-및-로깅)
9. [성능 튜닝 팁](#성능-튜닝-팁)
10. [향후 로드맵 및 기여 방법](#향후-로드맵-및-기여-방법)

---

## API 서버 구축

effGen을 REST API 서버로 배포하는 방법입니다.

### 기본 API 서버

effGen은 내장 API 서버를 제공합니다.

```bash
# CLI로 서버 시작
effgen serve \
    --model Qwen/Qwen2.5-3B-Instruct \
    --backend vllm \
    --port 8000 \
    --host 0.0.0.0 \
    --workers 4
```

### 프로그래밍 방식 서버

```python
from effgen.server import EffGenServer
from effgen import load_model
from effgen.tools.builtin import Calculator, WebSearch, FileOps

# 모델 로드
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    gpu_memory_utilization=0.9
)

# 서버 설정
server = EffGenServer(
    model=model,
    tools=[Calculator(), WebSearch(), FileOps()],
    host="0.0.0.0",
    port=8000,
    workers=4,
    max_concurrent_requests=50,
    timeout=120,
    enable_cors=True,
    api_key_required=True
)

# 서버 시작
server.start()
```

### API 엔드포인트

effGen API는 OpenAI 호환 엔드포인트를 제공합니다.

```python
import requests

# 1. Chat Completions (OpenAI 호환)
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={
        "Authorization": "Bearer your-api-key",
        "Content-Type": "application/json"
    },
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Calculate 15% of 250"}
        ],
        "temperature": 0.7,
        "max_tokens": 2048
    }
)

print(response.json())
# {
#   "id": "chatcmpl-123",
#   "object": "chat.completion",
#   "created": 1707123456,
#   "model": "Qwen/Qwen2.5-7B-Instruct",
#   "choices": [{
#     "index": 0,
#     "message": {
#       "role": "assistant",
#       "content": "15% of 250 is 37.5"
#     },
#     "finish_reason": "stop"
#   }]
# }

# 2. Agent Run (effGen 전용)
response = requests.post(
    "http://localhost:8000/v1/agent/run",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "query": "Search for the current Bitcoin price and calculate 10% of it",
        "tools": ["calculator", "web_search"],
        "enable_decomposition": True,
        "max_steps": 10
    }
)

print(response.json())
# {
#   "result": "The current Bitcoin price is $52,000. 10% of that is $5,200.",
#   "steps": [
#     {"tool": "web_search", "input": "current Bitcoin price", "output": "$52,000"},
#     {"tool": "calculator", "input": "52000 * 0.1", "output": "5200"}
#   ],
#   "execution_time": 3.42
# }

# 3. Streaming (스트리밍 응답)
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode('utf-8').replace('data: ', ''))
        if chunk['choices'][0]['finish_reason'] is None:
            print(chunk['choices'][0]['delta']['content'], end='', flush=True)
```

### FastAPI 커스텀 서버

더 많은 제어가 필요한 경우 FastAPI로 직접 구축할 수 있습니다.

```python
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, WebSearch
import uvicorn

app = FastAPI(title="effGen API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드 (startup 시 한 번만)
model = None

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model(
        "Qwen/Qwen2.5-7B-Instruct",
        backend="vllm",
        gpu_memory_utilization=0.9
    )

# 요청 모델
class ChatRequest(BaseModel):
    messages: list[dict]
    temperature: float = 0.7
    max_tokens: int = 2048

class AgentRequest(BaseModel):
    query: str
    tools: list[str] = ["calculator", "web_search"]
    max_steps: int = 10

# API 키 검증
async def verify_api_key(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid API key")
    api_key = authorization.replace("Bearer ", "")
    # 실제 환경에서는 데이터베이스에서 검증
    if api_key != "your-secret-api-key":
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# 엔드포인트
@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        agent = Agent(config=AgentConfig(
            model=model,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens
        ))

        # 마지막 메시지 추출
        user_message = request.messages[-1]["content"]
        result = agent.run(user_message)

        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/agent/run")
async def agent_run(
    request: AgentRequest,
    api_key: str = Depends(verify_api_key)
):
    try:
        # 도구 매핑
        tool_map = {
            "calculator": Calculator(),
            "web_search": WebSearch()
        }
        tools = [tool_map[t] for t in request.tools if t in tool_map]

        agent = Agent(config=AgentConfig(
            model=model,
            tools=tools,
            enable_decomposition=True,
            max_steps=request.max_steps
        ))

        result = agent.run(request.query)

        return {
            "result": result,
            "steps": agent.get_execution_history(),
            "execution_time": agent.last_execution_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 서버 실행
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

### 클라이언트 SDK

Python 클라이언트 예제:

```python
from openai import OpenAI

# OpenAI 클라이언트로 effGen 서버 사용
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

# Chat Completions
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "Calculate 15% of 250"}
    ]
)

print(response.choices[0].message.content)

# 스트리밍
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain AI agents"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

---

## Docker 샌드박스 보안

CodeExecutor의 Docker 샌드박스를 안전하게 설정하는 방법입니다.

### 기본 보안 설정

```python
from effgen.tools.builtin import CodeExecutor

executor = CodeExecutor(
    language="python",
    sandbox=True,
    # 보안 설정
    network_access=False,       # 네트워크 차단
    filesystem_readonly=True,   # 읽기 전용 파일시스템
    memory_limit="512m",        # 메모리 제한
    cpu_limit=1.0,              # CPU 코어 제한
    timeout=30,                 # 실행 시간 제한
    max_output_size="1m",       # 출력 크기 제한

    # 금지 모듈
    banned_modules=[
        "os", "subprocess", "sys", "socket",
        "urllib", "requests", "http"
    ],

    # 허용 모듈만 명시
    allowed_modules=[
        "math", "random", "datetime",
        "numpy", "pandas", "matplotlib"
    ]
)

# 안전하게 실행
result = executor.run("""
import numpy as np
data = np.random.rand(100)
print(f"Mean: {data.mean()}")
""")
```

### Docker 컨테이너 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  effgen-sandbox:
    image: effgen/sandbox:latest
    container_name: effgen-sandbox

    # 보안 설정
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined

    # 리소스 제한
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M

    # 네트워크 격리
    networks:
      - isolated

    # 읽기 전용 루트 파일시스템
    read_only: true

    # 임시 파일 시스템
    tmpfs:
      - /tmp:size=100M,mode=1777
      - /workspace:size=50M,mode=1777

    # 환경 변수
    environment:
      - PYTHONPATH=/usr/local/lib/python3.11
      - MAX_EXECUTION_TIME=30

    # 권한 제한
    user: "1000:1000"

    # 재시작 정책
    restart: unless-stopped

networks:
  isolated:
    driver: bridge
    internal: true  # 외부 인터넷 차단
```

### 샌드박스 정책

```python
from effgen.tools.builtin import CodeExecutor, SandboxPolicy

# 커스텀 보안 정책
policy = SandboxPolicy(
    # 시스템 호출 제한
    allowed_syscalls=[
        "read", "write", "open", "close",
        "stat", "fstat", "lseek"
    ],

    # 파일 경로 제한
    allowed_paths=[
        "/workspace",
        "/tmp"
    ],

    # 프로세스 제한
    max_processes=1,
    max_threads=4,

    # 리소스 제한
    max_file_size="10m",
    max_open_files=50,

    # 권한 제한
    allow_setuid=False,
    allow_ptrace=False
)

executor = CodeExecutor(
    sandbox=True,
    sandbox_policy=policy
)
```

### 코드 검증

실행 전에 위험한 코드를 감지합니다.

```python
from effgen.security import CodeValidator

validator = CodeValidator(
    # 금지 패턴
    forbidden_patterns=[
        r"import\s+os",
        r"import\s+subprocess",
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"open\s*\(.*(\'w\'|\"w\")",  # 쓰기 모드
    ],

    # 의심스러운 패턴 (경고)
    suspicious_patterns=[
        r"while\s+True:",  # 무한 루프
        r"for.*range\s*\(\s*\d{6,}",  # 큰 루프
    ]
)

code = """
import os
os.system('rm -rf /')
"""

validation_result = validator.validate(code)

if not validation_result.is_safe:
    print("Dangerous code detected!")
    print("Violations:", validation_result.violations)
    # ["import os", "os.system"]
else:
    executor.run(code)
```

---

## 속도 제한 및 안정성

프로덕션 환경에서 API를 안정적으로 운영하는 방법입니다.

### Rate Limiting

```python
from fastapi import FastAPI, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI()

# Rate limiter 설정
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 엔드포인트별 제한
@app.post("/v1/agent/run")
@limiter.limit("10/minute")  # 분당 10 요청
async def agent_run(request: Request, agent_request: AgentRequest):
    # 처리 로직
    pass

@app.post("/v1/chat/completions")
@limiter.limit("30/minute")  # 분당 30 요청
async def chat_completions(request: Request, chat_request: ChatRequest):
    # 처리 로직
    pass

# 사용자별 제한
from functools import lru_cache

@lru_cache()
def get_user_limit(api_key: str) -> str:
    # 데이터베이스에서 사용자 tier 조회
    user_tier = db.get_user_tier(api_key)

    limits = {
        "free": "10/hour",
        "pro": "100/hour",
        "enterprise": "1000/hour"
    }

    return limits.get(user_tier, "10/hour")

@app.post("/v1/agent/run")
async def agent_run_with_user_limit(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    limit = get_user_limit(api_key)
    # 동적 제한 적용
    pass
```

### 연결 풀링

```python
from effgen import Agent, load_model
from concurrent.futures import ThreadPoolExecutor
import queue

class AgentPool:
    """에이전트 연결 풀"""

    def __init__(self, model, pool_size=10):
        self.pool_size = pool_size
        self.agents = queue.Queue(maxsize=pool_size)

        # 풀 초기화
        for _ in range(pool_size):
            agent = Agent(config=AgentConfig(model=model))
            self.agents.put(agent)

    def acquire(self, timeout=30):
        """에이전트 획득"""
        try:
            return self.agents.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No agent available")

    def release(self, agent):
        """에이전트 반환"""
        self.agents.put(agent)

    def __enter__(self):
        self.agent = self.acquire()
        return self.agent

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release(self.agent)

# 사용
model = load_model("Qwen/Qwen2.5-7B-Instruct", backend="vllm")
pool = AgentPool(model, pool_size=10)

@app.post("/v1/agent/run")
async def agent_run(request: AgentRequest):
    with pool as agent:
        result = agent.run(request.query)
        return {"result": result}
```

### Circuit Breaker

장애 전파를 방지합니다.

```python
from pybreaker import CircuitBreaker

# Circuit Breaker 설정
breaker = CircuitBreaker(
    fail_max=5,           # 5번 실패 시 차단
    timeout_duration=60,  # 60초 후 재시도
    exclude=[HTTPException]
)

@app.post("/v1/agent/run")
@breaker
async def agent_run(request: AgentRequest):
    try:
        with pool as agent:
            result = agent.run(request.query)
            return {"result": result}
    except Exception as e:
        # Circuit breaker가 카운트
        raise

# 상태 확인
@app.get("/circuit/status")
async def circuit_status():
    return {
        "state": breaker.current_state,
        "fail_counter": breaker.fail_counter,
        "last_failure": breaker.last_failure
    }
```

### Graceful Shutdown

```python
import signal
import asyncio

class GracefulShutdown:
    def __init__(self):
        self.is_shutting_down = False
        self.active_requests = 0

    async def shutdown(self, signal_type):
        print(f"Received {signal_type}, shutting down gracefully...")
        self.is_shutting_down = True

        # 새 요청 거부
        while self.active_requests > 0:
            print(f"Waiting for {self.active_requests} requests to complete...")
            await asyncio.sleep(1)

        print("All requests completed, shutting down.")

shutdown_handler = GracefulShutdown()

@app.middleware("http")
async def shutdown_middleware(request, call_next):
    if shutdown_handler.is_shutting_down:
        return Response("Server is shutting down", status_code=503)

    shutdown_handler.active_requests += 1
    try:
        response = await call_next(request)
        return response
    finally:
        shutdown_handler.active_requests -= 1

# 시그널 핸들러 등록
def handle_signal(sig):
    asyncio.create_task(shutdown_handler.shutdown(sig))

signal.signal(signal.SIGINT, lambda s, f: handle_signal("SIGINT"))
signal.signal(signal.SIGTERM, lambda s, f: handle_signal("SIGTERM"))
```

---

## 프롬프트 최적화 전략

SLM에 최적화된 프롬프트 엔지니어링 기법입니다.

### 1. 컨텍스트 압축

```python
from effgen.core.optimizer import PromptOptimizer

optimizer = PromptOptimizer(
    compression_ratio=0.3,  # 70% 압축
    preserve_keywords=True,
    preserve_structure=True
)

# 긴 문서
long_context = """
[5000 토큰의 긴 문서...]
Based on this document, what are the key findings?
"""

# 압축 (1500 토큰으로)
compressed = optimizer.compress(long_context)

# 에이전트에 사용
agent.run(compressed)
```

### 2. Few-shot 프롬프팅

```python
from effgen.core.agent import AgentConfig

few_shot_examples = """
Example 1:
User: Calculate 15% of 200
Assistant: I'll use the calculator tool.
<tool>calculator</tool>
<input>200 * 0.15</input>
Result: 30

Example 2:
User: Search for Python tutorials
Assistant: I'll search the web.
<tool>web_search</tool>
<input>Python tutorials</input>
Result: [search results]
"""

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Calculator(), WebSearch()],
    system_prompt=f"""
You are a helpful assistant with access to tools.

Here are examples of how to use tools:
{few_shot_examples}

Now, help the user with their request.
"""
))
```

### 3. Chain of Thought

```python
system_prompt = """
You are a helpful assistant. When solving complex problems:

1. Break down the problem into steps
2. Show your reasoning for each step
3. Use tools when needed
4. Verify your final answer

Example:
User: If I save $500/month for 2 years at 5% annual interest, how much will I have?

Your reasoning:
Step 1: Identify variables
- Monthly savings: $500
- Duration: 2 years = 24 months
- Annual interest: 5% = 0.05
- Monthly interest: 0.05/12 ≈ 0.00417

Step 2: Calculate future value
- Use compound interest formula
- FV = P * [(1 + r)^n - 1] / r

Step 3: Calculate
<tool>calculator</tool>
<input>500 * ((1 + 0.00417)^24 - 1) / 0.00417</input>

Final answer: $12,639.56
"""

agent = Agent(config=AgentConfig(
    model=model,
    system_prompt=system_prompt
))
```

### 4. 역할 프롬프팅

```python
# 전문가 페르소나 부여
expert_prompts = {
    "data_analyst": """
You are an expert data analyst with 10 years of experience.
You excel at:
- Exploratory data analysis
- Statistical testing
- Data visualization
- Identifying trends and patterns

Always explain your analysis clearly and back up conclusions with data.
""",

    "software_architect": """
You are a senior software architect specializing in scalable systems.
You focus on:
- System design and architecture
- Performance optimization
- Security best practices
- Code quality and maintainability

Provide detailed technical explanations and consider edge cases.
""",

    "research_assistant": """
You are an academic research assistant with expertise in literature review.
You are skilled at:
- Finding relevant papers
- Summarizing key findings
- Identifying research gaps
- Synthesizing information

Always cite sources and maintain academic rigor.
"""
}

# 태스크에 따라 페르소나 선택
agent = Agent(config=AgentConfig(
    model=model,
    system_prompt=expert_prompts["data_analyst"]
))
```

### 5. 프롬프트 템플릿

```python
from string import Template

class PromptTemplates:
    """재사용 가능한 프롬프트 템플릿"""

    RESEARCH_TASK = Template("""
Research the following topic: $topic

Steps to follow:
1. Search for recent information (last $timeframe)
2. Find the top $num_sources sources
3. Extract key points from each
4. Synthesize into a coherent summary

Focus on: $focus_areas
""")

    DATA_ANALYSIS = Template("""
Analyze the dataset: $dataset_path

Analysis requirements:
- Descriptive statistics
- $analysis_type analysis
- Identify $num_insights key insights
- Create $num_visualizations visualizations

Output format: $output_format
""")

    CODE_REVIEW = Template("""
Review the following code:
$code

Check for:
- Bugs and errors
- Security vulnerabilities
- Performance issues
- Code style (follow $style_guide)
- Best practices

Provide specific, actionable feedback.
""")

# 사용
prompt = PromptTemplates.RESEARCH_TASK.substitute(
    topic="AI agent frameworks",
    timeframe="2 years",
    num_sources=5,
    focus_areas="performance, ease of use, community support"
)

agent.run(prompt)
```

---

## 토큰 예산 관리

토큰 사용을 최적화하여 비용과 성능을 개선합니다.

### 토큰 카운팅

```python
from effgen.utils import TokenCounter

counter = TokenCounter(model_name="Qwen/Qwen2.5-7B-Instruct")

text = "This is a sample text to count tokens."
num_tokens = counter.count(text)
print(f"Tokens: {num_tokens}")

# 메시지 리스트 카운팅
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence..."}
]

total_tokens = counter.count_messages(messages)
print(f"Total tokens: {total_tokens}")
```

### 토큰 예산 제한

```python
from effgen.core.agent import AgentConfig, TokenBudget

budget = TokenBudget(
    max_input_tokens=4096,      # 최대 입력 토큰
    max_output_tokens=2048,     # 최대 출력 토큰
    max_total_tokens=6144,      # 최대 총 토큰
    reserve_tokens=512          # 예약 토큰 (도구 사용 등)
)

agent = Agent(config=AgentConfig(
    model=model,
    token_budget=budget,
    auto_truncate=True  # 초과 시 자동 트리밍
))

# 긴 컨텍스트 자동 처리
long_query = "..." * 10000  # 매우 긴 쿼리
result = agent.run(long_query)
# 자동으로 예산 내로 압축됨
```

### 토큰 사용 모니터링

```python
from effgen.monitoring import TokenUsageMonitor

monitor = TokenUsageMonitor()

agent = Agent(config=AgentConfig(
    model=model,
    usage_monitor=monitor
))

# 여러 요청 실행
for query in queries:
    agent.run(query)

# 사용량 통계
stats = monitor.get_statistics()
print(stats)
# {
#   "total_requests": 100,
#   "total_input_tokens": 45230,
#   "total_output_tokens": 23450,
#   "total_tokens": 68680,
#   "average_input_tokens": 452.3,
#   "average_output_tokens": 234.5,
#   "peak_usage": 4096,
#   "estimated_cost": 0.0  # 로컬 실행
# }

# 시간별 사용량
hourly_usage = monitor.get_usage_by_hour()
# [
#   {"hour": "2026-02-09T10:00:00", "tokens": 5234},
#   {"hour": "2026-02-09T11:00:00", "tokens": 8921},
#   ...
# ]
```

### 스마트 캐싱

반복되는 컨텍스트를 캐싱하여 토큰 절약:

```python
from effgen.cache import PromptCache

cache = PromptCache(
    max_size=100,           # 최대 캐시 항목
    ttl=3600,               # 1시간 TTL
    similarity_threshold=0.9  # 유사도 임계값
)

agent = Agent(config=AgentConfig(
    model=model,
    prompt_cache=cache
))

# 첫 번째 요청 (캐시 미스)
result1 = agent.run("Explain quantum computing")
# 토큰: 1500

# 유사한 요청 (캐시 히트)
result2 = agent.run("Explain quantum computing in simple terms")
# 토큰: 200 (이전 컨텍스트 재사용)

# 캐시 통계
cache_stats = cache.get_stats()
# {
#   "hits": 45,
#   "misses": 55,
#   "hit_rate": 0.45,
#   "tokens_saved": 32450
# }
```

---

## 벡터 DB 통합

다양한 벡터 데이터베이스와 통합하는 방법입니다.

### 1. FAISS

```python
from effgen.tools.builtin import Retrieval
import faiss

# FAISS 인덱스 설정
retrieval = Retrieval(
    index_path="./docs",
    vector_store="faiss",
    index_type="IVF",        # Inverted File Index
    nlist=100,               # 클러스터 수
    nprobe=10,               # 검색할 클러스터 수
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 문서 인덱싱
retrieval.index_documents([
    "./docs/paper1.pdf",
    "./docs/paper2.pdf",
    "./docs/paper3.pdf"
])

# 검색
results = retrieval.search("transformer attention mechanism", top_k=5)
```

### 2. ChromaDB

```python
from effgen.tools.builtin import Retrieval
import chromadb

# ChromaDB 클라이언트
retrieval = Retrieval(
    index_path="./chroma_db",
    vector_store="chromadb",
    collection_name="research_papers",
    embedding_model="sentence-transformers/all-mpnet-base-v2",

    # ChromaDB 설정
    chroma_settings={
        "anonymized_telemetry": False,
        "allow_reset": True
    }
)

# 메타데이터와 함께 인덱싱
retrieval.index_documents([
    {
        "path": "./docs/paper1.pdf",
        "metadata": {
            "author": "John Doe",
            "year": 2026,
            "topic": "AI agents",
            "citations": 42
        }
    }
])

# 필터링된 검색
results = retrieval.search(
    query="multi-agent systems",
    top_k=10,
    filter={"year": {"$gte": 2024}, "citations": {"$gte": 10}}
)
```

### 3. Qdrant

```python
from effgen.tools.builtin import Retrieval
from qdrant_client import QdrantClient

# Qdrant 클라이언트
retrieval = Retrieval(
    index_path="./qdrant_data",
    vector_store="qdrant",
    collection_name="documents",
    qdrant_url="http://localhost:6333",
    qdrant_api_key="your-api-key",

    # 벡터 설정
    vector_size=384,
    distance_metric="cosine"
)

# 배치 인덱싱
retrieval.batch_index_documents(
    documents=document_list,
    batch_size=100,
    parallel_workers=4
)

# 하이브리드 검색 (벡터 + 키워드)
results = retrieval.hybrid_search(
    query="machine learning",
    top_k=5,
    alpha=0.7  # 0.7 벡터, 0.3 키워드
)
```

### 4. Weaviate

```python
from effgen.tools.builtin import Retrieval
import weaviate

# Weaviate 클라이언트
retrieval = Retrieval(
    index_path="./weaviate_data",
    vector_store="weaviate",
    weaviate_url="http://localhost:8080",
    weaviate_api_key="your-api-key",
    class_name="Document",

    # 스키마 정의
    schema={
        "class": "Document",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "title", "dataType": ["string"]},
            {"name": "author", "dataType": ["string"]},
            {"name": "year", "dataType": ["int"]}
        ]
    }
)

# GraphQL 쿼리
results = retrieval.graphql_search("""
{
  Get {
    Document(
      nearText: {concepts: ["AI agents"]}
      limit: 5
      where: {
        path: ["year"],
        operator: GreaterThan,
        valueInt: 2024
      }
    ) {
      content
      title
      author
      _additional {distance}
    }
  }
}
""")
```

### 벡터 DB 성능 비교

| DB | 속도 | 확장성 | 기능 | 설정 난이도 |
|----|------|--------|------|-------------|
| FAISS | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | 쉬움 |
| ChromaDB | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 쉬움 |
| Qdrant | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 중간 |
| Weaviate | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 중간 |

---

## 프로덕션 배포 체크리스트

프로덕션 환경에 배포하기 전에 확인할 사항입니다.

### 1. 보안

- [ ] API 키 인증 구현
- [ ] HTTPS 활성화
- [ ] Rate limiting 설정
- [ ] Input validation
- [ ] Docker 샌드박스 보안 설정
- [ ] 환경 변수로 시크릿 관리
- [ ] CORS 정책 설정
- [ ] SQL injection 방지
- [ ] XSS 방지

### 2. 성능

- [ ] vLLM 백엔드 사용 (로컬 배포 시)
- [ ] 연결 풀링 구현
- [ ] 응답 캐싱 설정
- [ ] GPU 메모리 최적화
- [ ] 로드 밸런싱 설정
- [ ] CDN 사용 (정적 리소스)
- [ ] 데이터베이스 인덱싱
- [ ] 쿼리 최적화

### 3. 모니터링

- [ ] 로깅 시스템 구축
- [ ] 메트릭 수집 (Prometheus)
- [ ] 알림 설정 (PagerDuty, Slack)
- [ ] 에러 추적 (Sentry)
- [ ] 성능 모니터링 (APM)
- [ ] 사용자 분석
- [ ] 비용 추적

### 4. 안정성

- [ ] Health check 엔드포인트
- [ ] Graceful shutdown
- [ ] Circuit breaker 구현
- [ ] 자동 재시작 정책
- [ ] 백업 시스템
- [ ] 장애 복구 계획
- [ ] 롤백 전략

### 5. 확장성

- [ ] 수평 확장 가능한 아키텍처
- [ ] 상태 비저장(stateless) 설계
- [ ] 분산 캐싱 (Redis)
- [ ] 메시지 큐 (RabbitMQ, Kafka)
- [ ] 오토스케일링 설정
- [ ] 데이터베이스 샤딩

### 6. 문서화

- [ ] API 문서 (Swagger/OpenAPI)
- [ ] 배포 가이드
- [ ] 트러블슈팅 가이드
- [ ] 아키텍처 다이어그램
- [ ] 런북 (runbook)
- [ ] 변경 로그

---

## 모니터링 및 로깅

프로덕션 시스템의 건강 상태를 추적합니다.

### 구조화된 로깅

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_object = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        if hasattr(record, "extra"):
            log_object.update(record.extra)

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object)

# 로거 설정
logger = logging.getLogger("effgen")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)

# 사용
logger.info("Agent request received", extra={
    "user_id": "user123",
    "query": "Calculate 2+2",
    "tools": ["calculator"]
})
```

### Prometheus 메트릭

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 메트릭 정의
requests_total = Counter(
    'effgen_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'effgen_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

active_agents = Gauge(
    'effgen_active_agents',
    'Number of currently active agents'
)

token_usage = Counter(
    'effgen_tokens_total',
    'Total tokens used',
    ['type']  # input, output
)

# FastAPI 미들웨어
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    active_agents.inc()

    try:
        response = await call_next(request)
        status = response.status_code
    except Exception as e:
        status = 500
        raise
    finally:
        duration = time.time() - start_time
        active_agents.dec()

        requests_total.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status
        ).inc()

        request_duration.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

    return response

# Prometheus 서버 시작
start_http_server(9090)
```

### Grafana 대시보드

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'effgen'
    static_configs:
      - targets: ['localhost:9090']
```

주요 메트릭 대시보드:
- 요청 처리량 (requests/sec)
- 응답 시간 (p50, p95, p99)
- 에러율
- 활성 에이전트 수
- GPU 메모리 사용량
- 토큰 사용량

### 에러 추적 (Sentry)

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastAPIIntegration

sentry_sdk.init(
    dsn="https://your-sentry-dsn",
    integrations=[FastAPIIntegration()],
    traces_sample_rate=0.1,  # 10% 트레이싱
    environment="production"
)

@app.post("/v1/agent/run")
async def agent_run(request: AgentRequest):
    try:
        with sentry_sdk.start_transaction(op="agent_run", name="Agent Execution"):
            result = agent.run(request.query)
            return {"result": result}
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise
```

---

## 성능 튜닝 팁

최적의 성능을 위한 팁입니다.

### 1. 모델 최적화

```python
# vLLM 설정 최적화
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",

    # GPU 메모리 최대 활용
    gpu_memory_utilization=0.95,

    # 텐서 병렬화 (멀티 GPU)
    tensor_parallel_size=2,

    # KV 캐시 최적화
    max_model_len=8192,
    block_size=16,

    # 스케줄링
    max_num_batched_tokens=8192,
    max_num_seqs=256,

    # 정밀도
    dtype="half",  # FP16

    # 양자화 (추가 속도 향상)
    quantization="awq"  # AWQ 4bit
)
```

### 2. 배치 처리

```python
from effgen import Agent

agent = Agent(config=AgentConfig(model=model))

# 개별 처리 (느림)
results = []
for query in queries:
    result = agent.run(query)
    results.append(result)

# 배치 처리 (빠름)
results = agent.batch_run(queries, batch_size=32)
```

### 3. 비동기 처리

```python
import asyncio
from effgen import Agent

agent = Agent(config=AgentConfig(model=model))

# 동기 (순차적)
def sync_process(queries):
    return [agent.run(q) for q in queries]

# 비동기 (병렬)
async def async_process(queries):
    tasks = [agent.arun(q) for q in queries]
    return await asyncio.gather(*tasks)

# 10배 빠를 수 있음
results = asyncio.run(async_process(queries))
```

### 4. 캐싱 전략

```python
from functools import lru_cache
from hashlib import md5

# 결과 캐싱
@lru_cache(maxsize=1000)
def cached_agent_run(query_hash):
    # 실제 query는 외부에서 저장
    return agent.run(queries_store[query_hash])

def run_with_cache(query):
    query_hash = md5(query.encode()).hexdigest()
    return cached_agent_run(query_hash)

# Redis 캐싱
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def run_with_redis_cache(query):
    cache_key = f"effgen:result:{md5(query.encode()).hexdigest()}"

    # 캐시 확인
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # 실행 및 캐싱
    result = agent.run(query)
    redis_client.setex(cache_key, 3600, json.dumps(result))  # 1시간 TTL

    return result
```

### 5. 프로파일링

```python
import cProfile
import pstats
from pstats import SortKey

# 프로파일링
profiler = cProfile.Profile()
profiler.enable()

# 코드 실행
agent.run("Complex query...")

profiler.disable()

# 결과 분석
stats = pstats.Stats(profiler)
stats.sort_stats(SortKey.CUMULATIVE)
stats.print_stats(20)  # 상위 20개 함수

# 라인 프로파일링
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(agent.run)
lp.run('agent.run("query")')
lp.print_stats()
```

---

## 향후 로드맵 및 기여 방법

effGen의 미래와 커뮤니티 참여 방법입니다.

### 로드맵

**v0.1.0 (2026 Q2)**
- [ ] 더 많은 SLM 모델 지원 (Mistral, Llama 3.3)
- [ ] 웹 UI 대시보드
- [ ] 고급 메모리 시스템 (그래프 메모리)
- [ ] 도구 마켓플레이스

**v0.2.0 (2026 Q3)**
- [ ] 멀티모달 지원 (이미지, 오디오)
- [ ] 분산 에이전트 시스템
- [ ] 자동 파인튜닝 파이프라인
- [ ] 벤치마크 스위트

**v0.3.0 (2026 Q4)**
- [ ] 엔터프라이즈 기능
- [ ] 고급 보안 기능
- [ ] 성능 최적화 (10x 목표)
- [ ] 프로덕션 템플릿

### 기여 방법

#### 1. 코드 기여

```bash
# 저장소 포크
git clone https://github.com/your-username/effGen.git
cd effGen

# 개발 환경 설정
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# 브랜치 생성
git checkout -b feature/your-feature

# 변경 작업
# ...

# 테스트 실행
pytest tests/

# 커밋 및 푸시
git add .
git commit -m "Add your feature"
git push origin feature/your-feature

# Pull Request 생성
```

#### 2. 도구 개발

새로운 도구를 만들어 기여할 수 있습니다.

```python
from effgen.tools.base import Tool
from pydantic import BaseModel, Field

class YourToolInput(BaseModel):
    param: str = Field(description="Parameter description")

class YourTool(Tool):
    name: str = "your_tool"
    description: str = "Your tool description"
    input_schema: type[BaseModel] = YourToolInput

    def _run(self, param: str) -> str:
        # 구현
        return "result"

# tests/test_your_tool.py
def test_your_tool():
    tool = YourTool()
    result = tool.run(param="test")
    assert result == "expected"
```

#### 3. 문서 기여

```markdown
# docs/guides/your-guide.md

# Your Guide Title

## Introduction
...

## Usage
...

## Examples
...
```

#### 4. 버그 리포트

좋은 버그 리포트 예시:

```markdown
**버그 설명**
에이전트가 Calculator 도구를 사용할 때 음수 결과를 잘못 처리합니다.

**재현 방법**
1. Calculator 도구로 에이전트 생성
2. "Calculate -5 * 3" 실행
3. 결과 확인

**예상 동작**
-15 반환

**실제 동작**
15 반환 (부호 누락)

**환경**
- effGen 버전: 0.0.2
- Python 버전: 3.11
- OS: Ubuntu 22.04
- GPU: RTX 3090

**추가 컨텍스트**
로그: [로그 첨부]
```

### 커뮤니티 리소스

- **GitHub Discussions**: 질문 및 아이디어 공유
- **Discord**: 실시간 채팅 및 지원
- **Twitter**: [@effGenAI](https://twitter.com/effGenAI)
- **Blog**: 튜토리얼 및 사례 연구
- **YouTube**: 비디오 튜토리얼

### 인정 및 보상

기여자는 다음과 같이 인정받습니다:
- README.md의 기여자 목록
- 릴리스 노트에 언급
- 특별 Discord 역할
- 커뮤니티 스포트라이트

---

## 결론

이제 effGen을 프로덕션 환경에 배포하고 최적화하는 방법을 모두 배웠습니다.

### 주요 포인트

1. **API 서버**: FastAPI 기반 안정적인 서버 구축
2. **보안**: Docker 샌드박스와 입력 검증으로 안전성 확보
3. **성능**: vLLM, 캐싱, 배치 처리로 최적화
4. **모니터링**: Prometheus, Grafana로 시스템 건강 추적
5. **확장성**: 로드 밸런싱과 오토스케일링으로 성장 대비

### 다음 단계

- effGen을 프로덕션에 배포해보세요
- 커뮤니티에 참여하여 경험을 공유하세요
- 새로운 도구나 기능을 개발하여 기여하세요
- effGen 생태계를 함께 성장시키세요

---

## 참고 자료

1. effGen Official Documentation. https://effgen.org/docs
2. FastAPI Production Guide. https://fastapi.tiangolo.com/deployment/
3. vLLM Performance Optimization. https://docs.vllm.ai/en/latest/
4. Prometheus Best Practices. https://prometheus.io/docs/practices/
5. Docker Security. https://docs.docker.com/engine/security/

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/)
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/)
- [04장: 모델 및 백엔드](/effgen-guide-04-models/)
- [05장: 도구 시스템 및 프로토콜](/effgen-guide-05-tools/)
- [06장: 멀티에이전트 및 태스크 분해](/effgen-guide-06-multi-agent/)
- [07장: 고급 활용 및 프로덕션](/effgen-guide-07-advanced/) ← 현재 문서

---

**effGen 완벽 가이드 시리즈를 마치며**

축하합니다! effGen의 모든 핵심 기능과 프로덕션 배포 방법을 마스터했습니다. 이제 Small Language Models을 활용하여 강력한 AI 에이전트 시스템을 구축할 수 있습니다.

질문이나 피드백이 있으시면 [GitHub Issues](https://github.com/ctrl-gaurav/effGen/issues)나 [Discord](https://discord.gg/effgen)에서 공유해주세요.

Happy Building with effGen!
