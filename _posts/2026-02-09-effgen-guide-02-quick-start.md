---
layout: post
title: "effGen 완벽 가이드 (02) - 설치 및 빠른 시작"
date: 2026-02-09
permalink: /effgen-guide-02-quick-start/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen 프레임워크 설치부터 첫 AI 에이전트 생성까지 단계별 가이드"
---

# effGen 완벽 가이드 (02) - 설치 및 빠른 시작

## 목차
1. [시스템 요구사항](#시스템-요구사항)
2. [설치 방법](#설치-방법)
3. [CLI 사용법](#cli-사용법)
4. [Python API 기본](#python-api-기본)
5. [첫 에이전트 만들기](#첫-에이전트-만들기)
6. [멀티-툴 에이전트](#멀티-툴-에이전트)
7. [문제 해결](#문제-해결)

---

## 시스템 요구사항

effGen을 실행하기 위한 최소 및 권장 사양입니다.

### 소프트웨어 요구사항

| 구성요소 | 최소 버전 | 권장 버전 |
|---------|----------|-----------|
| Python | 3.8+ | 3.10+ |
| PyTorch | 2.0+ | 2.3+ |
| CUDA (GPU 사용 시) | 11.8+ | 12.1+ |
| Docker (샌드박스용) | 20.10+ | 24.0+ |

### 하드웨어 요구사항

**CPU 전용 (기본 사용)**
- RAM: 8GB 이상
- 저장공간: 10GB 이상
- 속도: ~5-10 tokens/sec (1.5B 모델)

**GPU 가속 (권장)**

| 모델 크기 | VRAM | 권장 GPU | 추론 속도 |
|----------|------|----------|-----------|
| 1.5B (4bit) | 2-3GB | RTX 3060, T4 | ~45 tokens/sec |
| 3B (4bit) | 4-5GB | RTX 3060 Ti, RTX 4060 | ~35 tokens/sec |
| 7B (4bit) | 6-8GB | RTX 3080, RTX 4070 | ~25 tokens/sec |
| 14B (4bit) | 12-16GB | RTX 4090, A100 | ~15 tokens/sec |

**양자화 옵션별 메모리**

```python
# Qwen2.5-1.5B 모델 기준
FP16:  ~3GB VRAM   (최고 품질, 느림)
INT8:  ~2GB VRAM   (균형)
INT4:  ~1.5GB VRAM (권장, 속도/품질 밸런스)
INT2:  ~1GB VRAM   (최대 압축, 품질 저하)
```

### 지원 운영체제

- **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+ (완전 지원)
- **macOS**: 11.0+ (Metal 가속 지원)
- **Windows**: 10/11 (WSL2 권장)

---

## 설치 방법

effGen은 3가지 방법으로 설치할 수 있습니다.

### 방법 1: PyPI 설치 (권장)

가장 간단한 방법입니다:

```bash
# 기본 설치
pip install effgen

# 특정 버전 설치
pip install effgen==0.0.2

# 업그레이드
pip install --upgrade effgen
```

**설치 확인**:

```bash
effgen --version
# effgen, version 0.0.2
```

### 방법 2: vLLM 백엔드 포함 설치

고속 추론을 위한 vLLM 백엔드를 함께 설치:

```bash
# vLLM 포함 설치
pip install effgen[vllm]

# 또는 개별 설치
pip install effgen
pip install vllm
```

**vLLM 사용 시 장점**:
- 2-4배 빠른 추론 속도
- PagedAttention으로 메모리 효율 향상
- 배치 처리 최적화

**벤치마크 비교** (Qwen2.5-7B, RTX 4090):

```
Backend     | Tokens/sec | Latency (first token)
------------|------------|----------------------
Transformers|     25     |        1.2s
vLLM        |     95     |        0.3s
```

### 방법 3: 소스에서 설치 (개발자용)

최신 개발 버전을 사용하거나 기여하고 싶다면:

```bash
# 저장소 클론
git clone https://github.com/ctrl-gaurav/effGen.git
cd effGen

# 설치 스크립트 실행
chmod +x install.sh
./install.sh

# 또는 수동 설치
pip install -e .
```

**개발 모드 설치**:

```bash
# 개발 의존성 포함
pip install -e ".[dev]"

# 테스트 실행
pytest tests/

# 린팅
ruff check .
```

### 선택적 의존성

추가 기능을 위한 선택적 패키지:

```bash
# 벡터 데이터베이스 (Retrieval 도구용)
pip install chromadb faiss-cpu

# 이미지 처리 (멀티모달용)
pip install pillow transformers[vision]

# 고급 NLP
pip install spacy
python -m spacy download en_core_web_sm

# 문서 파싱
pip install pypdf docx2txt
```

---

## CLI 사용법

effGen은 강력한 명령줄 인터페이스를 제공합니다.

### 기본 명령어

#### 1. `effgen run` - 단일 작업 실행

```bash
# 기본 사용
effgen run "What is the capital of France?"

# 모델 지정
effgen run "Calculate 15% of 250" --model Qwen/Qwen2.5-1.5B-Instruct

# 도구 지정
effgen run "Search for latest AI news" --tools WebSearch Calculator

# 출력 형식 지정
effgen run "List prime numbers up to 50" --output json

# 상세 로그
effgen run "Complex task" --verbose
```

**실행 예시**:

```bash
$ effgen run "Calculate the square root of 144 and add 20"

[effGen] Loading model: Qwen/Qwen2.5-1.5B-Instruct
[effGen] Initializing tools: Calculator
[effGen] Processing query...

Thought: I need to calculate sqrt(144) first, then add 20.
Action: Calculator
Action Input: sqrt(144)
Observation: 12.0

Thought: Now I'll add 20 to the result.
Action: Calculator
Action Input: 12 + 20
Observation: 32.0

Final Answer: The result is 32.0
```

#### 2. `effgen chat` - 대화형 모드

```bash
# 기본 채팅
effgen chat

# 모델 및 도구 지정
effgen chat --model Qwen/Qwen2.5-3B-Instruct --tools Calculator WebSearch FileOps

# 시스템 프롬프트 커스텀
effgen chat --system-prompt "You are a Python programming expert."

# 메모리 활성화
effgen chat --memory ./chat_history.db
```

**채팅 세션 예시**:

```
$ effgen chat --tools Calculator WebSearch

[effGen] Starting interactive chat session
[effGen] Type 'exit' to quit, 'clear' to reset conversation

You: What's the current Bitcoin price?

Agent: [Thinking] I need to search for the current Bitcoin price.
[Tool: WebSearch] Query: "Bitcoin price USD now"
[Result] Bitcoin is currently trading at $45,234.56

The current Bitcoin price is approximately $45,234.56 USD.

You: Calculate 15% of that

Agent: [Thinking] I'll use the calculator to find 15% of 45234.56
[Tool: Calculator] 45234.56 * 0.15
[Result] 6785.184

15% of $45,234.56 is $6,785.18.

You: exit

[effGen] Conversation saved to ./chat_history.db
```

#### 3. `effgen serve` - API 서버 실행

REST API 서버로 effGen을 배포:

```bash
# 기본 서버 실행 (포트 8000)
effgen serve

# 포트 및 호스트 지정
effgen serve --host 0.0.0.0 --port 8080

# 워커 수 지정 (프로덕션)
effgen serve --workers 4

# 특정 모델로 서버 시작
effgen serve --model Qwen/Qwen2.5-7B-Instruct --tools all
```

**API 사용 예시**:

```bash
# 서버 시작
$ effgen serve --port 8000

# 다른 터미널에서 API 호출
$ curl -X POST http://localhost:8000/v1/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate the factorial of 10",
    "tools": ["Calculator"]
  }'

# 응답
{
  "result": "The factorial of 10 is 3,628,800",
  "steps": [
    {"tool": "Calculator", "input": "10!", "output": "3628800"}
  ],
  "execution_time": 1.23
}
```

#### 4. `effgen` - 대화형 설정 마법사

인자 없이 실행하면 대화형 설정 시작:

```bash
$ effgen

Welcome to effGen! Let's set up your first agent.

? Select a model:
  > Qwen/Qwen2.5-1.5B-Instruct (Fast, 2GB VRAM)
    Qwen/Qwen2.5-3B-Instruct (Balanced, 4GB VRAM)
    Qwen/Qwen2.5-7B-Instruct (Powerful, 8GB VRAM)
    Custom model path

? Select tools (Space to select, Enter to confirm):
  [x] Calculator
  [x] WebSearch
  [ ] CodeExecutor
  [x] FileOps
  [ ] PythonREPL

? Enable memory? (Y/n): Y
? Memory storage path: ./effgen_memory.db

? Start in chat mode? (Y/n): Y

[effGen] Initializing agent with your settings...
[effGen] Ready! Starting chat session.
```

---

## Python API 기본

프로그래밍 방식으로 effGen 사용하기.

### 기본 구조

모든 effGen 프로그램은 다음 패턴을 따릅니다:

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Tool1, Tool2

# 1. 모델 로드
model = load_model("model_name")

# 2. 도구 초기화
tools = [Tool1(), Tool2()]

# 3. 에이전트 설정
config = AgentConfig(
    name="agent_name",
    model=model,
    tools=tools,
    system_prompt="Custom instructions"
)

# 4. 에이전트 생성
agent = Agent(config=config)

# 5. 실행
result = agent.run("Your query here")
```

### 모델 로드 옵션

```python
from effgen import load_model

# 기본 로드 (자동 양자화)
model = load_model("Qwen/Qwen2.5-1.5B-Instruct")

# 명시적 양자화
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit"  # "8bit", "4bit", "2bit", None
)

# GPU 장치 지정
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    device="cuda:0"  # 또는 "cuda:1", "cpu", "mps"
)

# vLLM 백엔드 사용
model = load_model(
    "Qwen/Qwen2.5-7B-Instruct",
    backend="vllm",
    tensor_parallel_size=2  # 멀티GPU
)

# 로컬 모델 경로
model = load_model(
    "/path/to/local/model",
    trust_remote_code=True
)

# 생성 파라미터 설정
model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 512,
        "do_sample": True
    }
)
```

### AgentConfig 옵션

```python
from effgen.core.agent import AgentConfig

config = AgentConfig(
    # 필수
    name="my_agent",
    model=model,

    # 도구
    tools=[Calculator(), WebSearch()],
    max_tool_calls=10,  # 최대 도구 호출 횟수

    # 프롬프트
    system_prompt="Custom system instructions",
    user_prompt_template="User: {query}\nAssistant:",

    # 메모리
    memory=UnifiedMemory(),
    enable_memory_retrieval=True,

    # 실행
    max_iterations=15,  # 최대 반복 횟수
    timeout=300,        # 타임아웃 (초)

    # 분해
    enable_decomposition=True,
    decomposition_threshold=0.7,  # 복잡도 임계값

    # 멀티에이전트
    enable_sub_agents=True,
    max_sub_agents=3,

    # 로깅
    verbose=True,
    log_file="./agent.log"
)
```

---

## 첫 에이전트 만들기

단계별로 첫 번째 에이전트를 만들어봅시다.

### 예제 1: Calculator 에이전트

**프로젝트 구조**:
```
my_first_agent/
├── agent.py
└── requirements.txt
```

**requirements.txt**:
```
effgen>=0.0.2
torch>=2.0.0
```

**agent.py**:
```python
"""
첫 번째 effGen 에이전트 - 계산기 도우미
"""

from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator

def main():
    # 1. 모델 로드 (1.5B 모델, 4bit 양자화)
    print("📦 Loading model...")
    model = load_model(
        "Qwen/Qwen2.5-1.5B-Instruct",
        quantization="4bit",
        device="auto"  # 자동으로 GPU 또는 CPU 선택
    )

    # 2. 도구 초기화
    calculator = Calculator()

    # 3. 에이전트 설정
    config = AgentConfig(
        name="calculator_assistant",
        model=model,
        tools=[calculator],
        system_prompt=(
            "You are a helpful math assistant. "
            "Use the Calculator tool to perform accurate calculations. "
            "Always show your work step by step."
        ),
        max_iterations=5,
        verbose=True
    )

    # 4. 에이전트 생성
    agent = Agent(config=config)

    # 5. 테스트 쿼리들
    test_queries = [
        "Calculate 15% tip on a $85.50 bill",
        "What is the square root of 2025?",
        "If I invest $10,000 at 5% annual interest, how much will I have after 3 years?",
    ]

    print("\n" + "="*60)
    print("🤖 Calculator Agent Ready!")
    print("="*60 + "\n")

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"Query {i}: {query}")
        print('─'*60)

        result = agent.run(query)

        print(f"\n✅ Result: {result}")
        print()

    # 6. 대화형 모드
    print("\n" + "="*60)
    print("Entering interactive mode. Type 'quit' to exit.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            result = agent.run(user_input)
            print(f"\nAgent: {result}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    main()
```

**실행**:

```bash
# 의존성 설치
pip install -r requirements.txt

# 실행
python agent.py
```

**예상 출력**:

```
📦 Loading model...
[effGen] Downloading Qwen/Qwen2.5-1.5B-Instruct...
[effGen] Model loaded successfully (1.2GB)

============================================================
🤖 Calculator Agent Ready!
============================================================

────────────────────────────────────────────────────────────
Query 1: Calculate 15% tip on a $85.50 bill
────────────────────────────────────────────────────────────

[Agent] Thought: I need to calculate 15% of $85.50
[Agent] Action: Calculator
[Agent] Action Input: 85.50 * 0.15
[Agent] Observation: 12.825
[Agent] Thought: The tip amount is $12.83 (rounded)

✅ Result: The 15% tip on a $85.50 bill is $12.83

────────────────────────────────────────────────────────────
Query 2: What is the square root of 2025?
────────────────────────────────────────────────────────────

[Agent] Thought: I'll use the calculator to find sqrt(2025)
[Agent] Action: Calculator
[Agent] Action Input: sqrt(2025)
[Agent] Observation: 45.0

✅ Result: The square root of 2025 is 45

...
```

---

## 멀티-툴 에이전트

여러 도구를 사용하는 고급 에이전트를 만들어봅시다.

### 예제 2: 연구 보조 에이전트

```python
"""
멀티-툴 에이전트 - 연구 및 분석 보조
"""

from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import (
    Calculator,
    WebSearch,
    FileOps,
    PythonREPL
)
from effgen.memory import UnifiedMemory

def create_research_agent():
    """연구 보조 에이전트 생성"""

    # 모델 로드 (더 큰 모델 사용)
    model = load_model(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization="4bit",
        generation_config={
            "temperature": 0.3,  # 더 일관된 출력
            "top_p": 0.9,
            "max_new_tokens": 1024
        }
    )

    # 도구들 초기화
    calculator = Calculator()
    web_search = WebSearch(max_results=5)
    file_ops = FileOps(base_dir="./research_output")
    python_repl = PythonREPL()

    # 메모리 시스템
    memory = UnifiedMemory(
        short_term_size=20,
        long_term_storage="./research_memory.db",
        vector_store="chromadb"
    )

    # 에이전트 설정
    config = AgentConfig(
        name="research_assistant",
        model=model,
        tools=[calculator, web_search, file_ops, python_repl],
        memory=memory,
        system_prompt="""You are an expert research assistant with access to:
        - Calculator: for mathematical computations
        - WebSearch: for finding information online
        - FileOps: for reading and writing files
        - PythonREPL: for data analysis and visualization

        When given a research task:
        1. Search for relevant information
        2. Analyze the data using Python if needed
        3. Perform calculations if necessary
        4. Save results to files
        5. Provide a comprehensive summary

        Always cite sources and show your work.""",
        max_iterations=20,
        enable_decomposition=True,
        verbose=True
    )

    return Agent(config=config)

def example_research_task():
    """예제 연구 작업"""

    agent = create_research_agent()

    # 복잡한 멀티스텝 작업
    query = """
    Research task: Analyze the growth of AI agent frameworks in 2025-2026.

    Steps:
    1. Search for popular AI agent frameworks released in 2025-2026
    2. Compare their GitHub stars and activity
    3. Calculate growth percentages
    4. Create a simple visualization (save as plot.png)
    5. Write a summary report (save as report.md)
    """

    print("🔬 Starting research task...\n")
    result = agent.run(query)

    print("\n" + "="*60)
    print("📊 Research Complete!")
    print("="*60)
    print(f"\n{result}\n")

# 실행 예시
if __name__ == "__main__":
    example_research_task()
```

**실행 결과 예시**:

```python
🔬 Starting research task...

[Decomposition] Breaking down complex task into 5 subtasks:
  ├─ Task 1: WebSearch for AI agent frameworks 2025-2026
  ├─ Task 2: Extract GitHub statistics (depends on 1)
  ├─ Task 3: Calculate growth metrics (depends on 2)
  ├─ Task 4: Create visualization (depends on 3)
  └─ Task 5: Write report (depends on 4)

[Executing Task 1: WebSearch]
[Tool: WebSearch] Query: "AI agent frameworks 2025 2026 GitHub"
[Result] Found 5 relevant frameworks:
  - effGen (2026): 1.2k stars
  - LangGraph (2025): 8.5k stars
  - AutoGen (2025): 15k stars
  - CrewAI (2025): 6.3k stars
  - AgentOps (2026): 890 stars

[Executing Task 2: Extract statistics]
[Tool: WebSearch] Fetching detailed GitHub stats...
[Result] Data collected for all frameworks

[Executing Task 3: Calculate growth]
[Tool: Calculator] Computing growth percentages...
[Result] Average monthly growth: 23.4%

[Executing Task 4: Visualization]
[Tool: PythonREPL] Creating bar chart...
```python
import matplotlib.pyplot as plt

frameworks = ['effGen', 'LangGraph', 'AutoGen', 'CrewAI', 'AgentOps']
stars = [1200, 8500, 15000, 6300, 890]

plt.figure(figsize=(10, 6))
plt.bar(frameworks, stars, color='steelblue')
plt.title('AI Agent Frameworks - GitHub Stars (2025-2026)')
plt.ylabel('Stars')
plt.savefig('./research_output/plot.png')
```
[Result] Chart saved to ./research_output/plot.png

[Executing Task 5: Write report]
[Tool: FileOps] Writing to report.md...
[Result] Report saved

============================================================
📊 Research Complete!
============================================================

Summary: Successfully analyzed 5 AI agent frameworks from 2025-2026.
Key findings:
- AutoGen leads with 15k stars
- Average monthly growth: 23.4%
- effGen and AgentOps are newer but showing strong growth
- Full report saved to ./research_output/report.md
- Visualization saved to ./research_output/plot.png

Sources:
- GitHub API
- DuckDuckGo Search
```

---

## 문제 해결

일반적인 문제와 해결 방법입니다.

### 1. CUDA Out of Memory

**증상**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**해결**:
```python
# 더 작은 모델 사용
model = load_model("Qwen/Qwen2.5-1.5B-Instruct")  # 대신 3B

# 더 강한 양자화
model = load_model(
    "Qwen/Qwen2.5-3B-Instruct",
    quantization="4bit"  # 또는 "2bit"
)

# CPU로 폴백
model = load_model(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device="cpu"
)
```

### 2. 모델 다운로드 느림

**증상**: Hugging Face에서 모델 다운로드가 매우 느림

**해결**:
```bash
# 미러 사용 (중국/아시아)
export HF_ENDPOINT=https://hf-mirror.com
pip install effgen

# 또는 직접 다운로드 후 로컬 경로 사용
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir ./models/qwen-1.5b

# Python에서
model = load_model("./models/qwen-1.5b")
```

### 3. 도구 실행 실패

**증상**: 에이전트가 도구를 호출하지 않거나 잘못 호출

**해결**:
```python
# 시스템 프롬프트에 명시적 지침 추가
config = AgentConfig(
    system_prompt="""You MUST use tools to answer questions.

Available tools:
- Calculator: Use for ANY mathematical calculation
- WebSearch: Use to find current information
- FileOps: Use to read/write files

Format:
Thought: [your reasoning]
Action: [tool name]
Action Input: [input to tool]
""",
    # 도구 사용 강제
    force_tool_use=True
)
```

### 4. Docker 샌드박스 오류

**증상**: `CodeExecutor` 사용 시 Docker 관련 오류

**해결**:
```bash
# Docker 설치 확인
docker --version

# Docker 데몬 시작
sudo systemctl start docker

# 사용자를 docker 그룹에 추가
sudo usermod -aG docker $USER

# 또는 샌드박스 비활성화 (주의!)
```

```python
executor = CodeExecutor(sandbox=False)  # 로컬 실행
```

### 5. 메모리 누수

**증상**: 장시간 실행 시 메모리 사용량 증가

**해결**:
```python
# 메모리 제한 설정
memory = UnifiedMemory(
    short_term_size=10,  # 줄이기
    enable_cleanup=True,
    cleanup_interval=100  # 100 대화마다 정리
)

# 또는 수동 정리
agent.memory.clear_short_term()
agent.memory.compact_long_term()
```

### 6. vLLM 설치 오류

**증상**: `pip install effgen[vllm]` 실패

**해결**:
```bash
# Python 버전 확인 (3.8-3.11 지원)
python --version

# CUDA 버전 확인
nvcc --version

# 호환되는 버전 설치
pip install vllm==0.3.1  # 특정 버전

# 빌드 의존성
pip install ninja packaging
```

---

## 다음 단계

이제 effGen을 설치하고 기본 에이전트를 만들 수 있게 되었습니다. 다음 챕터에서는 effGen의 내부 아키텍처를 깊이 있게 살펴보겠습니다.

**[다음: 챕터 03 - 핵심 아키텍처 →](/effgen-guide-03-architecture/)**

---

## 참고 자료

1. [effGen 공식 문서](https://effgen.org/docs)
2. [Qwen2.5 모델 카드](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
3. [vLLM 문서](https://docs.vllm.ai/)
4. [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/) ← 현재 문서
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/)