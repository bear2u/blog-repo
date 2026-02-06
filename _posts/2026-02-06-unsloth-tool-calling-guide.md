---
layout: post
title: "Unsloth Tool Calling 완벽 가이드 - 로컬 LLM에서 함수 호출 마스터하기"
date: 2026-02-06
permalink: /unsloth-tool-calling-guide/
author: Unsloth AI
categories: [AI, LLM]
tags: [unsloth, tool-calling, llm, local-llm, function-calling, llama-cpp, python, ai-agent]
original_url: "https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms"
excerpt: "로컬 LLM에서 Tool Calling(함수 호출)을 활용하여 계산기, 터미널, Python 코드 실행 등 실제 작업을 수행하는 방법을 완벽히 알아봅니다."
image:
  path: /assets/img/unsloth-tool-calling.png
  alt: Unsloth Tool Calling Guide
---

## 목차
- [Tool Calling이란?](#tool-calling이란)
- [환경 구성](#환경-구성)
- [도구 함수 정의](#도구-함수-정의)
- [추론 함수 구현](#추론-함수-구현)
- [실전 예제](#실전-예제)
- [지원 모델](#지원-모델)
- [결론](#결론)

---

## Tool Calling이란?

**Tool Calling**(도구 호출 또는 함수 호출)은 LLM이 단순히 텍스트를 생성하는 것을 넘어, **구조화된 요청을 통해 특정 함수를 트리거**할 수 있게 해주는 강력한 기능입니다.

### 기존 LLM vs Tool Calling

| 구분 | 기존 LLM | Tool Calling |
|------|---------|--------------|
| 동작 방식 | 텍스트 추측 (text guessing) | 함수 실행 (function execution) |
| 정확성 | 환상(hallucination) 발생 가능 | 실제 계산/조회 수행 |
| 신뢰성 | 낮음 | 높음 (최신 데이터 사용 가능) |
| 예시 | "2+2는 4입니다" (추측) | `add_number(2, 2)` → `4` (실행) |

### Tool Calling의 핵심 이점

1. **신뢰성 향상**: 실제 함수 실행으로 환상(hallucination) 방지
2. **최신성 보장**: 실시간 데이터베이스 쿼리, API 호출 가능
3. **시스템 연동**: 파일 검색, 터미널 명령, 계산기 등 외부 시스템 제어
4. **사실 검증**: 답변을 실제 데이터로 검증 가능

---

## 환경 구성

### 1. llama.cpp 설치 및 빌드

Tool Calling을 사용하려면 먼저 **llama.cpp**를 GPU 지원과 함께 설치해야 합니다.

```bash
# 필수 패키지 설치
apt-get update
apt-get install pciutils build-essential cmake curl libcurl4-openssl-dev -y

# llama.cpp 클론
git clone https://github.com/ggml-org/llama.cpp

# GPU 가속(CUDA) 활성화 빌드
cmake llama.cpp -B llama.cpp/build \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON  # CPU만 사용하려면 OFF로 변경

# 병렬 빌드 (모든 필요한 바이너리 생성)
cmake --build llama.cpp/build --config Release -j \
    --clean-first \
    --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

# 바이너리 복사
cp llama.cpp/build/bin/llama-* llama.cpp
```

**주요 옵션:**
- `-DGGML_CUDA=ON`: NVIDIA GPU 가속 활성화
- `-j`: 병렬 컴파일 (CPU 코어 수만큼 사용)
- `--target`: 필요한 실행 파일만 빌드

---

## 도구 함수 정의

Tool Calling의 핵심은 **LLM이 호출할 수 있는 함수 세트**를 정의하는 것입니다.

### Python 함수 구현

```python
import json
import subprocess
import random
from typing import Any

# 1. 산술 연산 도구
def add_number(a: float | str, b: float | str) -> float:
    """두 숫자를 더합니다."""
    return float(a) + float(b)

def multiply_number(a: float | str, b: float | str) -> float:
    """두 숫자를 곱합니다."""
    return float(a) * float(b)

def substract_number(a: float | str, b: float | str) -> float:
    """첫 번째 숫자에서 두 번째 숫자를 뺍니다."""
    return float(a) - float(b)

# 2. 스토리 생성 도구
def write_a_story() -> str:
    """무작위 스토리를 생성합니다."""
    return random.choice([
        "A long time ago in a galaxy far far away...",
        "There were 2 friends who loved sloths and code...",
        "The world was ending because every sloth evolved to have superhuman intelligence...",
        "Unbeknownst to one friend, the other accidentally coded a program to evolve sloths...",
    ])

# 3. 터미널 명령 실행 (보안 제한 포함)
def terminal(command: str) -> str:
    """
    안전한 터미널 명령을 실행합니다.
    위험한 명령어(rm, sudo, dd, chmod)는 차단됩니다.
    """
    # 위험한 명령어 필터링
    dangerous_commands = ["rm", "sudo", "dd", "chmod"]
    if any(cmd in command for cmd in dangerous_commands):
        msg = f"Cannot execute '{', '.join(dangerous_commands)}' commands since they are dangerous"
        print(msg)
        return msg

    print(f"Executing terminal command `{command}`")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command failed: {e.stderr}"

# 4. Python 코드 실행
def python(code: str) -> str:
    """Python 코드를 실행하고 결과를 반환합니다."""
    data = {}
    exec(code, data)
    del data["__builtins__"]  # 내장 함수 제거
    return str(data)

# 함수 매핑 딕셔너리
MAP_FN = {
    "add_number": add_number,
    "multiply_number": multiply_number,
    "substract_number": substract_number,
    "write_a_story": write_a_story,
    "terminal": terminal,
    "python": python,
}
```

### OpenAI 호환 도구 스키마 정의

LLM이 함수를 이해할 수 있도록 **JSON 스키마**로 정의합니다.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_number",
            "description": "Add two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "string",
                        "description": "The first number."
                    },
                    "b": {
                        "type": "string",
                        "description": "The second number."
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_number",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "The first number."},
                    "b": {"type": "string", "description": "The second number."},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "substract_number",
            "description": "Subtract the second number from the first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "The first number."},
                    "b": {"type": "string", "description": "The second number."},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_a_story",
            "description": "Write a random story.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Execute a terminal command. Dangerous commands like 'rm', 'sudo', 'dd', 'chmod' are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The terminal command to execute."
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute."
                    },
                },
                "required": ["code"],
            },
        },
    },
]
```

**스키마 핵심 요소:**
- `name`: 함수 이름
- `description`: LLM이 언제 이 함수를 호출할지 결정하는 설명
- `parameters`: 함수 인자의 타입과 설명
- `required`: 필수 파라미터 목록

---

## 추론 함수 구현

이제 **LLM과 대화하며 도구 호출을 자동으로 처리**하는 메인 함수를 구현합니다.

```python
from openai import OpenAI

def unsloth_inference(
    messages,
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    min_p=0.01,
    repetition_penalty=1.0,
):
    """
    Unsloth 모델과 대화하며 도구 호출을 자동으로 처리합니다.

    Args:
        messages: 대화 메시지 리스트
        temperature: 창의성 조절 (낮을수록 결정적)
        top_p: 누적 확률 (nucleus sampling)
        top_k: 상위 K개 토큰만 고려 (-1: 비활성화)
        min_p: 최소 확률 필터
        repetition_penalty: 반복 억제 (1.0: 비활성화)

    Returns:
        도구 호출 결과가 포함된 전체 메시지 히스토리
    """
    messages = messages.copy()

    # OpenAI 호환 클라이언트 생성
    openai_client = OpenAI(
        base_url="http://127.0.0.1:8001/v1",
        api_key="sk-no-key-required",
    )

    # 사용 가능한 모델 확인
    model_name = next(iter(openai_client.models.list())).id
    print(f"Using model = {model_name}")

    has_tool_calls = True

    # 도구 호출이 더 이상 없을 때까지 반복
    while has_tool_calls:
        print(f"Current messages = {messages}")

        # LLM에 요청
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            extra_body={
                "top_k": top_k,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty
            }
        )

        # 응답 파싱
        tool_calls = response.choices[0].message.tool_calls or []
        content = response.choices[0].message.content or ""
        tool_calls_dict = [tc.to_dict() for tc in tool_calls] if tool_calls else tool_calls

        # Assistant 메시지 추가
        messages.append({
            "role": "assistant",
            "tool_calls": tool_calls_dict,
            "content": content
        })

        # 각 도구 호출 실행
        for tool_call in tool_calls:
            fx = tool_call.function.name
            args = tool_call.function.arguments
            _id = tool_call.id

            # 함수 실행
            out = MAP_FN[fx](**json.loads(args))

            # 도구 실행 결과를 메시지에 추가
            messages.append({
                "role": "tool",
                "tool_call_id": _id,
                "name": fx,
                "content": str(out)
            })
        else:
            # 도구 호출이 없으면 종료
            has_tool_calls = False

    return messages
```

**핵심 로직:**
1. **반복 루프**: 도구 호출이 더 이상 없을 때까지 계속 실행
2. **자동 실행**: LLM이 요청한 도구를 자동으로 찾아서 실행
3. **결과 반영**: 도구 실행 결과를 다시 LLM에 전달
4. **멀티턴 지원**: 여러 단계의 도구 호출을 자동으로 처리

---

## 실전 예제

### 예제 1: 스토리 생성

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Could you write me a story?"}],
}]

result = unsloth_inference(
    messages,
    temperature=0.15,
    top_p=1.0,
    top_k=-1,
    min_p=0.00
)

print(result[-1]["content"])
```

**실행 결과:**
```
Using model = unsloth/GLM-4.7
Current messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Could you write me a story?'}]}]
Current messages = [
  {'role': 'user', ...},
  {'role': 'assistant', 'tool_calls': [{'function': {'name': 'write_a_story', ...}}]},
  {'role': 'tool', 'content': 'There were 2 friends who loved sloths and code...'}
]

"Of course! Here's a story for you: There were 2 friends who loved sloths and code..."
```

### 예제 2: 날짜 계산

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "What is today's date plus 3 days?"}],
}]

result = unsloth_inference(
    messages,
    temperature=0.15,
    top_p=1.0,
    top_k=-1,
    min_p=0.00
)
```

**도구 호출 흐름:**
1. LLM이 `terminal("date")` 호출 → 오늘 날짜 확인
2. LLM이 `add_number(today_day, 3)` 호출 → 3일 더하기
3. 최종 답변 생성

### 예제 3: Python 코드 실행

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Create a Fibonacci function in Python and find fib(20)."}],
}]

result = unsloth_inference(
    messages,
    temperature=0.15,
    top_p=1.0,
    top_k=-1,
    min_p=0.00
)
```

**도구 호출 흐름:**
```python
# LLM이 자동으로 생성한 코드
python(code="""
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

result = fib(20)
""")
```

**결과:**
```
{'result': 6765}
```

### 예제 4: 터미널 명령 체인

```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Write 'I'm a happy Sloth' to a file, then print it back to me."}],
}]

result = unsloth_inference(
    messages,
    temperature=0.15,
    top_p=1.0,
    top_k=-1,
    min_p=0.00
)
```

**도구 호출 흐름:**
1. `terminal("echo \"I'm a happy Sloth\" > sloth.txt")`
2. `terminal("cat sloth.txt")`
3. 결과를 사용자에게 반환

---

## 지원 모델

Unsloth는 다양한 Tool Calling 지원 모델을 GGUF 형식으로 제공합니다.

### 1. GLM-4.7-Flash (추천 - 빠른 속도)

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download

# 모델 다운로드 (Q2_K_XL 양자화 버전)
snapshot_download(
    repo_id="unsloth/GLM-4.7-GGUF",
    local_dir="unsloth/GLM-4.7-GGUF",
    allow_patterns=["*UD-Q2_K_XL*"],
)
```

**서버 실행:**
```bash
./llama.cpp/llama-server \
    --model unsloth/GLM-4.7-GGUF/UD-Q2_K_XL/GLM-4.7-UD-Q2_K_XL-00001-of-00003.gguf \
    --alias "unsloth/GLM-4.7" \
    --threads -1 \
    --fit on \
    --prio 3 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --port 8001 \
    --jinja
```

**주요 파라미터:**
- `--ctx-size 16384`: 컨텍스트 윈도우 크기 (16K 토큰)
- `--threads -1`: 모든 CPU 코어 사용
- `--fit on`: 메모리 최적화 활성화
- `--jinja`: Jinja2 템플릿 지원 (채팅 형식)

### 2. Qwen3-Coder-Next (코딩 특화)

```python
snapshot_download(
    repo_id="unsloth/Qwen3-Coder-Next-14B-GGUF",
    local_dir="unsloth/Qwen3-Coder-Next-14B-GGUF",
    allow_patterns=["*UD-Q4_K_M*"],
)
```

### 3. Devstral-2 (멀티모달 지원)

```python
snapshot_download(
    repo_id="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
    local_dir="unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF",
    allow_patterns=["*UD-Q4_K_XL*", "*mmproj-F16*"],
)
```

**서버 실행 (이미지 지원 포함):**
```bash
./llama.cpp/llama-server \
    --model unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/Devstral-Small-2-24B-Instruct-2512-UD-Q4_K_XL.gguf \
    --mmproj unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF/mmproj-F16.gguf \
    --alias "unsloth/Devstral-Small-2-24B-Instruct-2512" \
    --threads -1 \
    --fit on \
    --prio 3 \
    --min_p 0.01 \
    --ctx-size 16384 \
    --port 8001 \
    --jinja
```

**특징:**
- `--mmproj`: 멀티모달 프로젝션 모델 (이미지 처리)
- 텍스트 + 이미지 동시 처리 가능

### 지원 모델 전체 목록

| 모델 | 크기 | 특화 분야 | 권장 양자화 |
|------|------|-----------|-------------|
| GLM-4.7-Flash | 4.7B | 빠른 응답 | Q2_K_XL |
| GLM-4.7 | 4.7B | 일반 대화 | Q4_K_M |
| Qwen3-Coder-Next | 14B | 코드 생성 | Q4_K_M |
| DeepSeek-R1-Distill | 7B/14B/32B | 논리 추론 | Q4_K_M |
| Devstral-2 | 24B | 멀티모달 | Q4_K_XL |
| NVIDIA Nemotron | 70B | 엔터프라이즈 | Q4_K_M |

---

## 고급 기능

### 1. 반복적 도구 호출

LLM이 **여러 도구를 순차적으로 호출**하여 복잡한 작업을 수행할 수 있습니다.

**예제: 파일 분석 및 통계**
```python
messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "List all Python files in the current directory, count them, and multiply by 2."
    }],
}]
```

**도구 호출 체인:**
1. `terminal("ls *.py")` → 파일 목록 확인
2. `terminal("ls *.py | wc -l")` → 개수 세기
3. `multiply_number(count, 2)` → 2배 계산

### 2. 조건부 도구 호출

LLM이 상황에 따라 **다른 도구를 선택**할 수 있습니다.

```python
messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "If today is Monday, write a story. Otherwise, calculate 10 + 5."
    }],
}]
```

### 3. 병렬 도구 호출

일부 모델은 **여러 도구를 동시에 호출**할 수 있습니다.

```python
# GLM-4.7은 하나의 응답에서 여러 도구를 병렬 호출 가능
messages = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": "Calculate 2+3, 5*6, and 10-4 at the same time."
    }],
}]
```

---

## 샘플링 파라미터 최적화

각 모델마다 **최적의 샘플링 파라미터**가 다릅니다.

### GLM-4.7 권장 설정

```python
unsloth_inference(
    messages,
    temperature=0.15,  # 낮은 값으로 결정적 출력
    top_p=1.0,         # nucleus sampling 비활성화
    top_k=-1,          # top-k sampling 비활성화
    min_p=0.00,        # 최소 확률 필터 비활성화
    repetition_penalty=1.0  # 반복 억제 비활성화
)
```

### Qwen3-Coder-Next 권장 설정

```python
unsloth_inference(
    messages,
    temperature=0.7,   # 중간 창의성
    top_p=0.95,        # 상위 95% 토큰만 고려
    top_k=40,          # 상위 40개 토큰만 고려
    min_p=0.01,        # 최소 1% 확률 이상만 허용
    repetition_penalty=1.1  # 약간의 반복 억제
)
```

### DeepSeek-R1 권장 설정 (논리 추론)

```python
unsloth_inference(
    messages,
    temperature=0.1,   # 매우 낮은 값으로 논리적 출력
    top_p=1.0,
    top_k=-1,
    min_p=0.00,
    repetition_penalty=1.0
)
```

---

## 보안 고려사항

### 1. 위험한 명령어 차단

```python
def terminal(command: str) -> str:
    # 블랙리스트 방식
    dangerous = ["rm", "sudo", "dd", "chmod", "mkfs", "fdisk"]
    if any(cmd in command for cmd in dangerous):
        return "Command blocked for security reasons"

    # 화이트리스트 방식 (더 안전)
    allowed_prefixes = ["ls", "cat", "echo", "date", "pwd", "whoami"]
    if not any(command.startswith(prefix) for prefix in allowed_prefixes):
        return "Only safe commands are allowed"

    return subprocess.run(command, capture_output=True, text=True, shell=True).stdout
```

### 2. 샌드박스 환경 사용

```python
import docker

def safe_python(code: str) -> str:
    """Docker 컨테이너에서 안전하게 Python 코드 실행"""
    client = docker.from_env()
    container = client.containers.run(
        "python:3.11-slim",
        f"python -c '{code}'",
        remove=True,
        mem_limit="512m",
        network_disabled=True,
        timeout=5
    )
    return container.decode()
```

### 3. 입력 검증

```python
def add_number(a: float | str, b: float | str) -> float:
    try:
        return float(a) + float(b)
    except ValueError:
        raise ValueError(f"Invalid numbers: {a}, {b}")
```

---

## 트러블슈팅

### 1. 모델이 도구를 호출하지 않는 경우

**원인:**
- 도구 설명(`description`)이 불명확
- 모델이 도구 호출을 학습하지 않음

**해결:**
```python
# 나쁜 예
"description": "Do math."

# 좋은 예
"description": "Add two numbers together. Use this when the user asks to add, sum, or combine two numeric values."
```

### 2. 무한 루프 발생

**원인:**
- 도구가 항상 같은 결과를 반환
- LLM이 종료 조건을 인식하지 못함

**해결:**
```python
MAX_ITERATIONS = 10
iteration = 0

while has_tool_calls and iteration < MAX_ITERATIONS:
    iteration += 1
    # ... 도구 호출 로직
```

### 3. 메모리 부족

**원인:**
- 모델 크기가 너무 큼
- 컨텍스트 윈도우가 너무 큼

**해결:**
```bash
# 더 작은 양자화 사용
--model *Q2_K_XL*  # 대신 Q4_K_M 사용

# 컨텍스트 크기 줄이기
--ctx-size 8192  # 16384 대신 8192 사용

# 배치 크기 줄이기
--batch-size 128
```

---

## 결론

Unsloth Tool Calling은 **로컬 LLM의 능력을 극대화**하는 강력한 기능입니다.

### 핵심 정리

1. **환경 구성**: llama.cpp + GGUF 모델
2. **도구 정의**: Python 함수 + OpenAI 스키마
3. **추론 루프**: 자동 도구 호출 및 결과 반영
4. **보안**: 위험 명령어 차단 + 샌드박스

### 활용 분야

| 분야 | 활용 예시 |
|------|-----------|
| 개발 자동화 | 코드 생성 → 실행 → 테스트 |
| 데이터 분석 | SQL 쿼리 → 결과 분석 → 시각화 |
| 시스템 관리 | 로그 분석 → 문제 진단 → 자동 복구 |
| 고객 지원 | 질문 파싱 → DB 조회 → 답변 생성 |

### 다음 단계

1. **Langchain 연동**: 더 복잡한 에이전트 워크플로우 구축
2. **벡터 DB 연동**: RAG(Retrieval-Augmented Generation) 구현
3. **멀티모달**: 이미지 분석 + 도구 호출 결합
4. **Fine-tuning**: 커스텀 도구에 특화된 모델 학습

Unsloth Tool Calling으로 **단순한 챗봇을 넘어 실제 작업을 수행하는 AI 에이전트**를 구축해보세요!

---

## 참고 자료

- [Unsloth 공식 문서](https://unsloth.ai/docs)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)
- [OpenAI Function Calling 가이드](https://platform.openai.com/docs/guides/function-calling)
- [Unsloth GGUF 모델 허브](https://huggingface.co/unsloth)

---

**원문**: [Tool Calling Guide for Local LLMs - Unsloth AI](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms)

**번역 및 정리**: Bear (2026-02-06)
