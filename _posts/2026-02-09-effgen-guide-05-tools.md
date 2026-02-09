---
layout: post
title: "effGen 완벽 가이드 (05) - 도구 시스템 및 프로토콜"
date: 2026-02-09
permalink: /effgen-guide-05-tools/
author: Gaurav Srivastava
categories: [AI 에이전트, Python]
tags: [SLM, AI Agent, Small Language Models, Tool Use, Multi-Agent, Python, Qwen, vLLM]
original_url: "https://github.com/ctrl-gaurav/effGen"
excerpt: "effGen의 강력한 도구 시스템, 7가지 내장 도구 상세 가이드, 그리고 MCP/A2A/ACP 프로토콜 통합"
---

# effGen 완벽 가이드 (05) - 도구 시스템 및 프로토콜

## 목차
1. [도구 시스템 개요](#도구-시스템-개요)
2. [내장 도구 상세 가이드](#내장-도구-상세-가이드)
3. [커스텀 도구 만들기](#커스텀-도구-만들기)
4. [프로토콜 어댑터](#프로토콜-어댑터)
5. [도구 사용 패턴](#도구-사용-패턴)
6. [베스트 프랙티스](#베스트-프랙티스)

---

## 도구 시스템 개요

effGen의 도구 시스템은 AI 에이전트가 외부 기능을 실행할 수 있게 합니다. SLM에 최적화된 도구 호출 메커니즘을 사용합니다.

### 도구 실행 플로우

```
사용자 쿼리
    ↓
[에이전트 분석]
    ↓
도구 필요? ──No──> 직접 응답
    ↓ Yes
[도구 선택 및 파라미터 추출]
    ↓
[도구 실행]
    ↓
[결과 통합]
    ↓
최종 응답
```

### 기본 사용법

```python
from effgen import Agent, load_model
from effgen.core.agent import AgentConfig
from effgen.tools.builtin import Calculator, WebSearch

# 모델 로드
model = load_model("Qwen/Qwen2.5-3B-Instruct", quantization="4bit")

# 도구와 함께 에이전트 생성
agent = Agent(config=AgentConfig(
    name="assistant",
    model=model,
    tools=[Calculator(), WebSearch()],
    system_prompt="You are a helpful assistant with access to tools."
))

# 도구를 자동으로 사용
result = agent.run("What is 15% of the current Bitcoin price?")
# 에이전트가 자동으로:
# 1. WebSearch로 Bitcoin 가격 검색
# 2. Calculator로 15% 계산
# 3. 결과를 자연어로 응답
```

---

## 내장 도구 상세 가이드

effGen에 포함된 7가지 도구를 상세히 살펴봅니다.

### 1. Calculator - 수학 계산 및 단위 변환

안전한 수학 표현식 평가 및 단위 변환 도구입니다.

#### 기본 사용

```python
from effgen.tools.builtin import Calculator

calc = Calculator()

# 기본 계산
result = calc.run("2 + 2")  # "4"
result = calc.run("sqrt(16)")  # "4.0"
result = calc.run("sin(pi/2)")  # "1.0"

# 복잡한 수식
result = calc.run("(123.45 * 67.89) + sqrt(12345)")
# "8492.1405"

# 백분율 계산
result = calc.run("15% of 85.50")  # "12.825"
result = calc.run("increase 100 by 20%")  # "120.0"
```

#### 단위 변환

```python
# 길이
calc.run("convert 100 meters to feet")  # "328.084 feet"
calc.run("5 km to miles")  # "3.10686 miles"

# 무게
calc.run("convert 10 kg to pounds")  # "22.0462 pounds"

# 온도
calc.run("convert 100 celsius to fahrenheit")  # "212.0 fahrenheit"

# 시간
calc.run("convert 2 hours to minutes")  # "120 minutes"

# 통화 (실시간 환율)
calc.run("convert 100 USD to EUR")  # 환율 API 사용
```

#### 고급 기능

```python
from effgen.tools.builtin import Calculator

# 커스텀 함수 추가
calc = Calculator(custom_functions={
    "double": lambda x: x * 2,
    "triple": lambda x: x * 3
})

result = calc.run("double(21)")  # "42"

# 안전 모드 설정
calc = Calculator(
    allow_variables=False,  # 변수 할당 금지
    timeout=5,              # 5초 타임아웃
    max_operations=100      # 최대 연산 수 제한
)

# 변수 사용 (allow_variables=True 시)
calc = Calculator(allow_variables=True)
calc.run("x = 10")
calc.run("y = x * 2")
result = calc.run("x + y")  # "30"
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import Calculator

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Calculator()],
    system_prompt="You are a math tutor."
))

# 자연어 수학 문제
agent.run("If I have $1000 and it grows by 5% annually, how much will I have after 3 years?")
# 에이전트가 자동으로 compound interest 공식 계산
```

### 2. WebSearch - DuckDuckGo 웹 검색

개인정보 보호에 중점을 둔 웹 검색 도구입니다.

#### 기본 검색

```python
from effgen.tools.builtin import WebSearch

search = WebSearch(max_results=5)

# 텍스트 검색
results = search.run("latest AI agent frameworks 2026")

# 결과 형식:
# [
#   {
#     "title": "Top AI Agent Frameworks in 2026",
#     "url": "https://example.com/article",
#     "snippet": "This article reviews the best AI agent frameworks...",
#     "published_date": "2026-02-01"
#   },
#   ...
# ]
```

#### 고급 검색

```python
# 시간 범위 지정
search = WebSearch(
    max_results=10,
    time_range="week"  # "day", "week", "month", "year"
)

results = search.run("Python 3.13 release")

# 지역 설정
search = WebSearch(
    max_results=5,
    region="kr-kr",  # 한국
    safe_search="moderate"  # "off", "moderate", "strict"
)

results = search.run("서울 날씨")

# 특정 사이트 검색
search = WebSearch()
results = search.run("site:github.com AI agent frameworks")
```

#### 뉴스 검색

```python
from effgen.tools.builtin import WebSearch

news_search = WebSearch(
    search_type="news",
    max_results=10
)

results = news_search.run("AI breakthroughs")
# 최신 뉴스 기사만 반환
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import WebSearch, Calculator

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator()],
    system_prompt="You are a research assistant."
))

# 실시간 데이터 검색 및 분석
agent.run(
    "Find the current stock price of Tesla and calculate "
    "what 100 shares would be worth"
)
```

### 3. CodeExecutor - Docker 샌드박스 코드 실행

안전한 샌드박스 환경에서 코드를 실행합니다.

#### Python 실행

```python
from effgen.tools.builtin import CodeExecutor

executor = CodeExecutor(
    language="python",
    sandbox=True,          # Docker 샌드박스 사용
    timeout=30,            # 30초 타임아웃
    memory_limit="512m",   # 메모리 제한
    network_access=False   # 네트워크 차단
)

# 간단한 코드 실행
result = executor.run("""
print("Hello, World!")
""")
# Output: "Hello, World!"

# 데이터 처리
result = executor.run("""
import numpy as np

data = np.random.rand(1000)
mean = data.mean()
std = data.std()

print(f"Mean: {mean:.4f}")
print(f"Std: {std:.4f}")
""")
```

#### 파일 시스템 접근

```python
# 작업 디렉토리 설정
executor = CodeExecutor(
    language="python",
    working_dir="/workspace",
    allow_file_ops=True  # 파일 작업 허용
)

# 데이터 파일 처리
result = executor.run("""
import pandas as pd

# CSV 읽기
df = pd.read_csv('/workspace/data.csv')

# 분석
summary = df.describe()
print(summary)
""")
```

#### 다양한 언어 지원

```python
# JavaScript
js_executor = CodeExecutor(language="javascript")
result = js_executor.run("""
const arr = [1, 2, 3, 4, 5];
const sum = arr.reduce((a, b) => a + b, 0);
console.log(sum);
""")

# Bash
bash_executor = CodeExecutor(language="bash")
result = bash_executor.run("""
#!/bin/bash
echo "System Info:"
uname -a
df -h
""")
```

#### 패키지 관리

```python
# 사전 설치된 패키지 지정
executor = CodeExecutor(
    language="python",
    packages=["numpy", "pandas", "matplotlib", "scikit-learn"]
)

# 또는 실행 시 설치
result = executor.run("""
# 런타임에 패키지 설치
import subprocess
subprocess.check_call(['pip', 'install', 'requests'])

import requests
response = requests.get('https://api.github.com')
print(response.status_code)
""")
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import CodeExecutor, FileOps

agent = Agent(config=AgentConfig(
    model=model,
    tools=[CodeExecutor(), FileOps()],
    system_prompt="You are a coding assistant."
))

# 자연어로 코드 요청
agent.run(
    "Create a Python script that generates a Fibonacci sequence "
    "up to 100 and saves it to fibonacci.txt"
)
```

### 4. PythonREPL - 대화형 Python 환경

상태를 유지하는 대화형 Python 환경입니다.

#### 기본 사용

```python
from effgen.tools.builtin import PythonREPL

repl = PythonREPL()

# 세션 1: 변수 정의
repl.run("x = 42")

# 세션 2: 이전 변수 사용
repl.run("y = x * 2")

# 세션 3: 결과 확인
result = repl.run("print(x, y)")  # "42 84"

# 세션 4: 함수 정의
repl.run("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
""")

# 세션 5: 함수 사용
result = repl.run("print([fibonacci(i) for i in range(10)])")
# "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
```

#### 데이터 분석 워크플로우

```python
from effgen.tools.builtin import PythonREPL

repl = PythonREPL()

# Step 1: 라이브러리 임포트
repl.run("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
""")

# Step 2: 데이터 로드
repl.run("""
df = pd.read_csv('sales_data.csv')
print(df.head())
""")

# Step 3: 데이터 전처리
repl.run("""
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
""")

# Step 4: 분석
repl.run("""
monthly_sales = df.groupby('month')['sales'].sum()
print(monthly_sales)
""")

# Step 5: 시각화
repl.run("""
monthly_sales.plot(kind='bar')
plt.savefig('sales_chart.png')
print("Chart saved!")
""")
```

#### 세션 관리

```python
from effgen.tools.builtin import PythonREPL

# 격리된 세션
repl1 = PythonREPL(session_id="session_1")
repl2 = PythonREPL(session_id="session_2")

repl1.run("x = 100")
repl2.run("x = 200")

print(repl1.run("x"))  # "100"
print(repl2.run("x"))  # "200"

# 세션 초기화
repl1.reset()
print(repl1.run("x"))  # NameError: x is not defined
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import PythonREPL, FileOps

agent = Agent(config=AgentConfig(
    model=model,
    tools=[PythonREPL(), FileOps()],
    system_prompt="You are a data analyst."
))

# 멀티스텝 분석
agent.run(
    "Load the data.csv file, perform exploratory data analysis, "
    "identify trends, and create visualizations"
)
# 에이전트가 여러 REPL 세션을 통해 단계적으로 분석
```

### 5. FileOps - 파일 시스템 읽기/쓰기

안전한 파일 시스템 작업 도구입니다.

#### 기본 작업

```python
from effgen.tools.builtin import FileOps

files = FileOps(
    base_dir="./workspace",  # 작업 디렉토리
    read_only=False          # 쓰기 허용
)

# 파일 읽기
content = files.read("data.txt")
print(content)

# 파일 쓰기
files.write("output.txt", "Hello, effGen!")

# 파일 추가
files.append("log.txt", "New log entry\n")

# 파일 삭제
files.delete("temp.txt")

# 디렉토리 목록
file_list = files.list("./")
print(file_list)
# ["data.txt", "output.txt", "log.txt"]
```

#### 디렉토리 작업

```python
from effgen.tools.builtin import FileOps

files = FileOps(base_dir="./workspace")

# 디렉토리 생성
files.mkdir("reports")
files.mkdir("data/processed", parents=True)  # 중간 경로도 생성

# 재귀적 목록
all_files = files.list("./", recursive=True)
# ["data.txt", "reports/", "reports/report1.pdf", ...]

# 파일 존재 확인
exists = files.exists("data.txt")  # True

# 파일 정보
info = files.info("data.txt")
# {
#   "size": 1024,
#   "created": "2026-02-09T10:00:00",
#   "modified": "2026-02-09T11:30:00",
#   "is_file": True
# }
```

#### 고급 기능

```python
from effgen.tools.builtin import FileOps

# 패턴 매칭
files = FileOps(base_dir="./workspace")

# 특정 확장자만
py_files = files.list("./", pattern="*.py")

# 정규식 사용
csv_files = files.list("./", pattern=r".*\.csv$")

# 파일 복사
files.copy("source.txt", "backup/source.txt")

# 파일 이동
files.move("old_location.txt", "new_location.txt")

# 압축
files.compress("data/", "data_backup.zip")

# 압축 해제
files.extract("data_backup.zip", "restored/")
```

#### 안전 기능

```python
from effgen.tools.builtin import FileOps

# 읽기 전용 모드
read_only_files = FileOps(
    base_dir="./data",
    read_only=True
)

# 쓰기 시도 시 에러
try:
    read_only_files.write("new.txt", "content")
except PermissionError:
    print("Read-only mode!")

# 경로 제한 (디렉토리 탈출 방지)
safe_files = FileOps(
    base_dir="./workspace",
    allow_absolute_paths=False,  # 절대 경로 금지
    allow_parent_access=False    # ../ 금지
)

try:
    safe_files.read("../../etc/passwd")
except ValueError:
    print("Path traversal blocked!")
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import FileOps, PythonREPL

agent = Agent(config=AgentConfig(
    model=model,
    tools=[FileOps(), PythonREPL()],
    system_prompt="You are a file management assistant."
))

# 파일 작업 자동화
agent.run(
    "Read all CSV files in the data folder, merge them, "
    "and save the result as combined_data.csv"
)
```

### 6. Retrieval - RAG 기반 문서 검색

의미 기반 문서 검색 및 RAG(검색 증강 생성) 도구입니다.

#### 기본 설정

```python
from effgen.tools.builtin import Retrieval

retrieval = Retrieval(
    index_path="./docs",                               # 인덱스 저장 경로
    chunk_size=500,                                    # 청크 크기
    chunk_overlap=50,                                  # 청크 오버랩
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # 임베딩 모델
)

# 문서 인덱싱
retrieval.index_documents([
    "./papers/paper1.pdf",
    "./papers/paper2.txt",
    "./papers/paper3.md"
])

# 검색
results = retrieval.search("How does attention mechanism work?", top_k=3)

# 결과 형식:
# [
#   {
#     "content": "The attention mechanism allows...",
#     "source": "./papers/paper1.pdf",
#     "score": 0.87,
#     "metadata": {"page": 5, "section": "Methods"}
#   },
#   ...
# ]
```

#### 고급 임베딩 모델

```python
from effgen.tools.builtin import Retrieval

# OpenAI 임베딩
retrieval = Retrieval(
    index_path="./docs",
    embedding_model="openai",
    embedding_model_name="text-embedding-3-large"
)

# Cohere 임베딩
retrieval = Retrieval(
    index_path="./docs",
    embedding_model="cohere",
    embedding_model_name="embed-multilingual-v3.0"
)

# 로컬 임베딩 (다국어)
retrieval = Retrieval(
    index_path="./docs",
    embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
```

#### 메타데이터 필터링

```python
from effgen.tools.builtin import Retrieval

retrieval = Retrieval(index_path="./docs")

# 메타데이터와 함께 인덱싱
retrieval.index_documents([
    {
        "path": "./papers/paper1.pdf",
        "metadata": {
            "author": "John Doe",
            "year": 2026,
            "topic": "AI agents"
        }
    },
    {
        "path": "./papers/paper2.pdf",
        "metadata": {
            "author": "Jane Smith",
            "year": 2025,
            "topic": "NLP"
        }
    }
])

# 필터링된 검색
results = retrieval.search(
    query="transformer architecture",
    top_k=5,
    filter={"year": 2026, "topic": "AI agents"}
)
```

#### 벡터 DB 백엔드

```python
from effgen.tools.builtin import Retrieval

# FAISS (기본)
retrieval_faiss = Retrieval(
    index_path="./docs",
    vector_store="faiss",
    index_type="IVF"  # Inverted File Index
)

# ChromaDB
retrieval_chroma = Retrieval(
    index_path="./docs",
    vector_store="chromadb",
    collection_name="my_docs"
)

# Qdrant
retrieval_qdrant = Retrieval(
    index_path="./docs",
    vector_store="qdrant",
    qdrant_url="http://localhost:6333",
    collection_name="documents"
)

# Weaviate
retrieval_weaviate = Retrieval(
    index_path="./docs",
    vector_store="weaviate",
    weaviate_url="http://localhost:8080",
    class_name="Document"
)
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import Retrieval, WebSearch

# RAG 에이전트
retrieval = Retrieval(index_path="./knowledge_base")

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Retrieval(), WebSearch()],
    system_prompt="You are a research assistant. Use the knowledge base first, then web search if needed."
))

# 지식 기반 질문 응답
agent.run("What are the latest developments in multi-agent systems?")
# 에이전트가 자동으로:
# 1. Retrieval로 로컬 지식 검색
# 2. 부족하면 WebSearch 사용
# 3. 결과 종합하여 응답
```

### 7. AgenticSearch - Grep 기반 정확한 검색

고급 패턴 매칭, 정규식 검색, 구조화된 데이터 추출 도구입니다.

#### 기본 검색

```python
from effgen.tools.builtin import AgenticSearch

search = AgenticSearch()

# 텍스트에서 패턴 찾기
text = """
Name: John Doe
Email: john@example.com
Phone: 123-456-7890

Name: Jane Smith
Email: jane@example.com
Phone: 098-765-4321
"""

# 이메일 추출
emails = search.find_pattern(
    text=text,
    pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
)
# ["john@example.com", "jane@example.com"]

# 전화번호 추출
phones = search.find_pattern(
    text=text,
    pattern=r"\b\d{3}-\d{3}-\d{4}\b"
)
# ["123-456-7890", "098-765-4321"]
```

#### 파일 검색

```python
from effgen.tools.builtin import AgenticSearch

search = AgenticSearch()

# 디렉토리 내 모든 파일에서 검색
results = search.search_in_files(
    directory="./src",
    pattern=r"def\s+(\w+)\(",  # Python 함수 정의
    file_pattern="*.py",
    recursive=True
)

# 결과:
# [
#   {
#     "file": "./src/main.py",
#     "line": 10,
#     "match": "def process_data(",
#     "context": "..."
#   },
#   ...
# ]
```

#### 구조화된 데이터 추출

```python
from effgen.tools.builtin import AgenticSearch

search = AgenticSearch()

document = """
2026-02-09: Meeting with team about project X
2026-02-10: Review code changes
2026-02-11: Deploy to production
"""

# 날짜 추출
dates = search.extract_dates(document)
# ["2026-02-09", "2026-02-10", "2026-02-11"]

# 엔티티 추출 (NER)
text = "Apple Inc. announced a new product in Cupertino on Monday."

entities = search.extract_entities(
    text=text,
    entity_types=["ORG", "LOC", "DATE"]
)
# {
#   "ORG": ["Apple Inc."],
#   "LOC": ["Cupertino"],
#   "DATE": ["Monday"]
# }
```

#### 코드 분석

```python
from effgen.tools.builtin import AgenticSearch

search = AgenticSearch()

code = """
import numpy as np
import pandas as pd

def process_data(df):
    return df.dropna()

class DataProcessor:
    def __init__(self):
        self.data = []

    def add_data(self, item):
        self.data.append(item)
"""

# 임포트 추출
imports = search.find_pattern(
    text=code,
    pattern=r"^import\s+(\S+)|^from\s+(\S+)\s+import",
    multiline=True
)
# ["numpy", "pandas"]

# 함수 정의 추출
functions = search.find_pattern(
    text=code,
    pattern=r"def\s+(\w+)\s*\("
)
# ["process_data", "__init__", "add_data"]

# 클래스 정의 추출
classes = search.find_pattern(
    text=code,
    pattern=r"class\s+(\w+)"
)
# ["DataProcessor"]
```

#### 에이전트와 통합

```python
from effgen import Agent
from effgen.tools.builtin import AgenticSearch, FileOps

agent = Agent(config=AgentConfig(
    model=model,
    tools=[AgenticSearch(), FileOps()],
    system_prompt="You are a code analysis assistant."
))

# 코드베이스 분석
agent.run(
    "Find all TODO comments in the Python files under ./src "
    "and create a summary report"
)
```

---

## 커스텀 도구 만들기

자신만의 도구를 만드는 방법입니다.

### 기본 도구 구조

```python
from effgen.tools.base import Tool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    """도구 입력 스키마"""
    query: str = Field(description="검색 쿼리")
    max_results: int = Field(default=5, description="최대 결과 수")

class MyTool(Tool):
    """커스텀 도구 예제"""

    name: str = "my_tool"
    description: str = "This tool does something useful"
    input_schema: type[BaseModel] = MyToolInput

    def _run(self, query: str, max_results: int = 5) -> str:
        """도구 실행 로직"""
        # 실제 로직 구현
        results = self.perform_search(query, max_results)
        return f"Found {len(results)} results"

    def perform_search(self, query: str, max_results: int):
        # 검색 로직
        return []

# 사용
tool = MyTool()
result = tool.run(query="test", max_results=10)
```

### 실전 예제: GitHub API 도구

```python
from effgen.tools.base import Tool
from pydantic import BaseModel, Field
import requests

class GitHubSearchInput(BaseModel):
    query: str = Field(description="검색 쿼리")
    search_type: str = Field(
        default="repositories",
        description="검색 타입: repositories, issues, users"
    )
    max_results: int = Field(default=5, description="최대 결과 수")

class GitHubSearch(Tool):
    """GitHub 검색 도구"""

    name: str = "github_search"
    description: str = "Search GitHub for repositories, issues, or users"
    input_schema: type[BaseModel] = GitHubSearchInput

    def __init__(self, api_token: str = None):
        super().__init__()
        self.api_token = api_token
        self.base_url = "https://api.github.com/search"

    def _run(
        self,
        query: str,
        search_type: str = "repositories",
        max_results: int = 5
    ) -> str:
        """GitHub API 검색 실행"""
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"token {self.api_token}"

        url = f"{self.base_url}/{search_type}"
        params = {"q": query, "per_page": max_results}

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        # 결과 포맷팅
        results = []
        for item in items:
            if search_type == "repositories":
                results.append({
                    "name": item["full_name"],
                    "description": item.get("description", ""),
                    "stars": item["stargazers_count"],
                    "url": item["html_url"]
                })

        return str(results)

# 에이전트에 통합
from effgen import Agent

github_tool = GitHubSearch(api_token="ghp_...")

agent = Agent(config=AgentConfig(
    model=model,
    tools=[github_tool],
    system_prompt="You are a GitHub search assistant."
))

agent.run("Find the top 5 Python AI agent frameworks")
```

### 실전 예제: SQL Database 도구

```python
from effgen.tools.base import Tool
from pydantic import BaseModel, Field
import sqlite3
import pandas as pd

class SQLQueryInput(BaseModel):
    query: str = Field(description="SQL 쿼리")
    fetch_all: bool = Field(default=True, description="모든 결과 반환")

class SQLDatabase(Tool):
    """SQL 데이터베이스 쿼리 도구"""

    name: str = "sql_database"
    description: str = "Execute SQL queries on the database"
    input_schema: type[BaseModel] = SQLQueryInput

    def __init__(self, db_path: str, read_only: bool = True):
        super().__init__()
        self.db_path = db_path
        self.read_only = read_only

    def _run(self, query: str, fetch_all: bool = True) -> str:
        """SQL 쿼리 실행"""
        # 읽기 전용 모드에서 쓰기 쿼리 차단
        if self.read_only:
            forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
            if any(keyword in query.upper() for keyword in forbidden):
                return "Error: Write operations not allowed in read-only mode"

        try:
            conn = sqlite3.connect(self.db_path)

            if fetch_all:
                df = pd.read_sql_query(query, conn)
                result = df.to_string()
            else:
                cursor = conn.cursor()
                cursor.execute(query)
                result = str(cursor.fetchone())

            conn.close()
            return result

        except Exception as e:
            return f"Error executing query: {str(e)}"

# 사용 예제
db_tool = SQLDatabase(db_path="./sales.db", read_only=True)

agent = Agent(config=AgentConfig(
    model=model,
    tools=[db_tool],
    system_prompt="You are a data analyst with SQL database access."
))

agent.run("Show me the top 10 customers by total sales")
```

---

## 프로토콜 어댑터

effGen은 3가지 에이전트 프로토콜을 지원합니다.

### 1. MCP (Model Context Protocol)

Anthropic의 Model Context Protocol을 통해 외부 도구에 접근합니다.

#### 기본 사용

```python
from effgen.tools.protocols import MCPTool

# MCP 서버에 연결
mcp_tools = MCPTool.from_server(
    server_url="sqlite://./data.db",
    server_type="database"
)

# 에이전트에 추가
agent = Agent(config=AgentConfig(
    model=model,
    tools=mcp_tools,
    system_prompt="You have access to database tools via MCP."
))

# 자동으로 MCP 도구 사용
agent.run("Query the database for user statistics")
```

#### 커스텀 MCP 서버

```python
from effgen.tools.protocols import MCPTool

# 파일 시스템 MCP
file_mcp = MCPTool.from_server(
    server_url="file:///workspace",
    server_type="filesystem"
)

# API MCP
api_mcp = MCPTool.from_server(
    server_url="https://api.example.com/mcp",
    server_type="api",
    auth_token="Bearer token123"
)

# 여러 MCP 서버 통합
agent = Agent(config=AgentConfig(
    model=model,
    tools=[*file_mcp, *api_mcp],
))
```

### 2. A2A (Agent-to-Agent)

OpenAI의 Agent-to-Agent 프로토콜로 다른 에이전트와 통신합니다.

#### 기본 사용

```python
from effgen.tools.protocols import A2ATool

# A2A 엔드포인트에 연결
a2a_tools = A2ATool.from_endpoint(
    endpoint="https://agent.example.com/a2a",
    api_key="key123"
)

# 에이전트 통합
agent = Agent(config=AgentConfig(
    model=model,
    tools=a2a_tools,
    system_prompt="You can communicate with other agents via A2A."
))

# 다른 에이전트에게 작업 위임
agent.run("Ask the translation agent to translate this to Korean: Hello World")
```

#### 멀티 에이전트 협업

```python
from effgen import Agent
from effgen.tools.protocols import A2ATool

# 연구 에이전트
research_a2a = A2ATool.from_endpoint(
    endpoint="https://research-agent.example.com/a2a",
    agent_name="researcher"
)

# 코딩 에이전트
coding_a2a = A2ATool.from_endpoint(
    endpoint="https://coding-agent.example.com/a2a",
    agent_name="coder"
)

# 조율 에이전트
coordinator = Agent(config=AgentConfig(
    model=model,
    tools=[research_a2a, coding_a2a],
    system_prompt="You coordinate tasks between specialized agents."
))

coordinator.run(
    "Research the latest transformer architectures and "
    "implement a simple version in PyTorch"
)
```

### 3. ACP (Agent Communication Protocol)

표준화된 에이전트 통신 프로토콜입니다.

#### 기본 사용

```python
from effgen.tools.protocols import ACPTool

# ACP 레지스트리에서 도구 발견
acp_tools = ACPTool.from_registry(
    registry_url="https://registry.example.com",
    tool_categories=["data_processing", "visualization"]
)

# 에이전트에 통합
agent = Agent(config=AgentConfig(
    model=model,
    tools=acp_tools,
    system_prompt="You have access to ACP tools."
))
```

#### 도구 발견 및 등록

```python
from effgen.tools.protocols import ACPTool, ACPRegistry

# 레지스트리 연결
registry = ACPRegistry(url="https://registry.example.com")

# 사용 가능한 도구 검색
available_tools = registry.discover_tools(
    capabilities=["file_processing", "data_analysis"],
    min_version="1.0.0"
)

# 필터링된 도구 로드
selected_tools = ACPTool.from_registry(
    registry_url="https://registry.example.com",
    tool_ids=["tool-123", "tool-456"]
)

# 자신의 도구를 레지스트리에 등록
from effgen.tools.base import Tool

registry.register_tool(
    tool=MyCustomTool(),
    metadata={
        "version": "1.0.0",
        "capabilities": ["custom_processing"],
        "author": "Your Name"
    }
)
```

---

## 도구 사용 패턴

효과적인 도구 사용을 위한 패턴입니다.

### 1. 도구 체이닝

여러 도구를 순차적으로 사용합니다.

```python
from effgen import Agent
from effgen.tools.builtin import WebSearch, Calculator, FileOps

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator(), FileOps()],
    system_prompt="You can chain multiple tools to solve complex tasks."
))

# 자동 도구 체이닝
agent.run(
    "Search for the current gold price, calculate what 10 ounces would cost, "
    "and save the result to gold_price.txt"
)

# 실행 플로우:
# 1. WebSearch: "current gold price"
# 2. Calculator: "price * 10"
# 3. FileOps: write to file
```

### 2. 조건부 도구 사용

상황에 따라 다른 도구를 선택합니다.

```python
from effgen import Agent
from effgen.tools.builtin import Retrieval, WebSearch

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Retrieval(index_path="./docs"), WebSearch()],
    system_prompt="""
    First try to answer using the local knowledge base (Retrieval).
    If no relevant information is found, use WebSearch.
    """
))

agent.run("What is the company's vacation policy?")
# Retrieval로 먼저 검색, 없으면 WebSearch
```

### 3. 병렬 도구 실행

독립적인 도구를 동시에 실행합니다.

```python
from effgen import Agent
from effgen.tools.builtin import WebSearch, Calculator

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator()],
    enable_parallel_tools=True,  # 병렬 실행 활성화
    system_prompt="You can use multiple tools in parallel when appropriate."
))

agent.run(
    "What are the current prices of Bitcoin and Ethereum, "
    "and what is 5% of each?"
)

# 병렬 실행:
# - WebSearch("Bitcoin price") || WebSearch("Ethereum price")
# - Calculator("BTC * 0.05") || Calculator("ETH * 0.05")
```

### 4. 에러 처리 및 재시도

```python
from effgen import Agent
from effgen.tools.builtin import WebSearch

agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch()],
    tool_retry_max=3,           # 최대 3회 재시도
    tool_timeout=30,            # 30초 타임아웃
    system_prompt="Handle tool errors gracefully."
))

# 도구 실패 시 자동 재시도 및 대체 전략
agent.run("Search for very obscure topic that might fail")
```

---

## 베스트 프랙티스

도구를 효과적으로 사용하기 위한 권장사항입니다.

### 1. 도구 선택 최소화

```python
# 나쁜 예: 너무 많은 도구
agent = Agent(config=AgentConfig(
    model=model,
    tools=[
        Calculator(), WebSearch(), CodeExecutor(), PythonREPL(),
        FileOps(), Retrieval(), AgenticSearch(), GitHubSearch(),
        SQLDatabase(), EmailSender(), SlackBot(), ...  # 15개+
    ]
))

# 좋은 예: 태스크에 필요한 도구만
agent = Agent(config=AgentConfig(
    model=model,
    tools=[WebSearch(), Calculator()]  # 현재 태스크에 필요한 것만
))
```

**이유**: SLM은 도구가 많을수록 선택 정확도가 떨어집니다.

### 2. 명확한 도구 설명

```python
class MyTool(Tool):
    name: str = "my_tool"

    # 나쁜 예
    description: str = "Does stuff"

    # 좋은 예
    description: str = (
        "Searches the product database for items matching the query. "
        "Returns product name, price, and availability. "
        "Use this when the user asks about product information."
    )
```

### 3. 입력 검증

```python
from pydantic import BaseModel, Field, validator

class MyToolInput(BaseModel):
    query: str = Field(
        description="Search query",
        min_length=1,
        max_length=500
    )
    max_results: int = Field(
        default=5,
        ge=1,  # >= 1
        le=100  # <= 100
    )

    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()
```

### 4. 시스템 프롬프트 최적화

```python
# 도구 사용 가이드를 시스템 프롬프트에 명시
system_prompt = """
You are a helpful assistant with access to the following tools:

1. Calculator: Use for mathematical calculations
2. WebSearch: Use for current information from the internet
3. FileOps: Use for reading/writing files

Guidelines:
- Always use Calculator for math, don't compute manually
- Check files before searching the web
- Explain your tool usage to the user
"""

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Calculator(), WebSearch(), FileOps()],
    system_prompt=system_prompt
))
```

### 5. 도구 실행 로깅

```python
from effgen import Agent
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("effgen.tools")

agent = Agent(config=AgentConfig(
    model=model,
    tools=[Calculator(), WebSearch()],
    enable_tool_logging=True,  # 도구 실행 로깅
    log_level="INFO"
))

# 실행 추적
result = agent.run("Calculate 2+2 and search for Python tutorials")

# 로그 출력:
# [INFO] Tool 'calculator' called with args: {'expression': '2+2'}
# [INFO] Tool 'calculator' returned: 4
# [INFO] Tool 'web_search' called with args: {'query': 'Python tutorials'}
# [INFO] Tool 'web_search' returned: [...]
```

---

## 다음 단계

이제 effGen의 강력한 도구 시스템과 프로토콜을 이해했습니다. 다음 챕터에서는 멀티에이전트 조율과 태스크 분해 시스템을 살펴봅니다.

**[다음: 챕터 06 - 멀티에이전트 및 태스크 분해 →](/effgen-guide-06-multi-agent/)**

---

## 참고 자료

1. Model Context Protocol (MCP). https://modelcontextprotocol.io/
2. OpenAI Agent-to-Agent Protocol. https://openai.com/a2a
3. effGen Tools Documentation. https://effgen.org/docs/tools

---

**전체 가이드 목차**:
- [01장: 소개 및 개요](/effgen-guide-01-intro/)
- [02장: 설치 및 빠른 시작](/effgen-guide-02-quick-start/)
- [03장: 핵심 아키텍처](/effgen-guide-03-architecture/)
- [04장: 모델 및 백엔드](/effgen-guide-04-models/)
- [05장: 도구 시스템 및 프로토콜](/effgen-guide-05-tools/) ← 현재 문서
- [06장: 멀티에이전트 및 태스크 분해](/effgen-guide-06-multi-agent/)
- [07장: 고급 활용 및 프로덕션](/effgen-guide-07-advanced/)
