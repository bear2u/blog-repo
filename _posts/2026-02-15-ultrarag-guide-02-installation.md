---
layout: post
title: "UltraRAG 완벽 가이드 (02) - 설치 및 환경 설정"
date: 2026-02-15
permalink: /ultrarag-guide-02-installation/
author: UltraRAG Team
categories: [AI 에이전트, RAG]
tags: [RAG, MCP, LLM, UltraRAG, 설치, Docker]
original_url: "https://github.com/OpenBMB/UltraRAG"
excerpt: "uv를 활용한 UltraRAG 설치 방법과 Docker 컨테이너 배포 방법을 단계별로 안내합니다."
---

## 설치 방법

UltraRAG는 로컬 소스 코드 설치와 Docker 컨테이너 배포, 두 가지 설치 방법을 제공합니다.

### 방법 1: 소스 코드 설치 (권장)

[uv](https://github.com/astral-sh/uv)를 사용하여 Python 환경과 의존성을 관리할 것을 권장합니다. uv는 설치 속도를 크게 향상시킬 수 있습니다.

#### 환경 준비

아직 uv를 설치하지 않았다면 다음 명령을 실행하세요:

```shell
# pip으로 설치
pip install uv

# 또는 curl로 다운로드
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 소스 코드 다운로드

```shell
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG
```

#### 의존성 설치

사용 사례에 따라 다음 모드 중 하나를 선택하여 의존성을 설치합니다:

**A: 새 환경 생성**
`uv sync`를 사용하여 가상 환경을 자동으로 생성하고 의존성을 동기화합니다:

- **핵심 의존성**: 기본 핵심 기능만 필요할 경우 (예: UltraRAG UI만 사용)
  ```shell
  uv sync
  ```

- **전체 설치**: 검색, 생성, 코퍼스 처리, 평가 등 UltraRAG의 모든 기능을 완전히 경험하려면:
  ```shell
  uv sync --all-extras
  ```

- **필요시 설치**: 특정 모듈만 실행해야 하는 경우:
  ```shell
  uv sync --extra retriever   # 검색 모듈만
  uv sync --extra generation  # 생성 모듈만
  uv sync --extra corpus     # 코퍼스 모듈만
  uv sync --extra evaluation # 평가 모듈만
  ```

설치가 완료되면 가상 환경을 활성화합니다:

```shell
# Windows CMD
.venv\Scripts\activate.bat

# Windows Powershell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

**B: 기존 환경에 설치**
현재 활성화된 Python 환경에 UltraRAG를 설치하려면 `uv pip`을 사용합니다:

```shell
# 핵심 의존성
uv pip install -e .

# 전체 설치
uv pip install -e ".[all]"

# 필요시 설치
uv pip install -e ".[retriever]"
```

### 방법 2: Docker 컨테이너 배포

로컬 Python 환경을 구성하고 싶지 않다면 Docker를 사용하여 배포할 수 있습니다.

#### 코드 및 이미지 준비

```shell
# 1. 레포지토리 클론
git clone https://github.com/OpenBMB/UltraRAG.git --depth 1
cd UltraRAG

# 2. 이미지 준비 (선택)
# 옵션 A: Docker Hub에서-pull
docker pull hdxin2002/ultrarag:v0.3.0-base-cpu # 기본 버전 (CPU)
docker pull hdxin2002/ultrarag:v0.3.0-base-gpu # 기본 버전 (GPU)
docker pull hdxin2002/ultrarag:v0.3.0          # 전체 버전 (GPU)

# 옵션 B: 로컬에서 빌드
docker build -t ultrarag:v0.3.0 .
```

#### 컨테이너 시작

```shell
# 컨테이너 시작 (포트 5050이 자동으로 매핑됨)
docker run -it --gpus all -p 5050:5050 <docker_image_name>
```

**참고**: 컨테이너가 시작되면 UltraRAG UI가 자동으로 실행됩니다. 브라우저에서 `http://localhost:5050`에 직접 접근하여 사용할 수 있습니다.

---

## 설치 확인

설치 후 다음 예제 명령을 실행하여 환경이 정상인지 확인합니다:

```shell
ultrarag run examples/sayhello.yaml
```

다음 출력이 나타나면 설치가 성공한 것입니다:

```
Hello, UltraRAG v3!
```

---

## Python 버전 요구사항

UltraRAG은 다음 Python 버전을 요구합니다:

- **Python 3.11 ~ 3.12**

---

## 선택적 의존성

### 검색 (Retriever) 모듈

```toml
retriever = [
    "infinity-emb",
    "sentence-transformers",
    "openai",
    "bm25s",
    "faiss-gpu-cu12",
    "exa_py",
    "tavily-python",
    "pymilvus",
    "numba",
    "torch",
    "fastapi",
    "uvicorn",
    "pydantic",
]
```

### 생성 (Generation) 모듈

```toml
generation = [
    "vllm>=0.13.0",
    "openai",
    "transformers",
    "torch",
]
```

### 평가 (Evaluation) 모듈

```toml
evaluation = [
    "rouge-score",
    "pytrec-eval-terrier",
]
```

### 코퍼스 처리 모듈

```toml
corpus = [
    "mineru[core]",
]
```

---

## 빠른 시작 명령어

```shell
# CLI 도움말 확인
ultrarag --help

# 파이프라인 실행
ultrarag run <yaml_file>

# 서버 모드 실행
ultrarag serve <yaml_file>
```

---

*다음 글에서는 UltraRAG의 MCP 아키텍처 및 핵심 구성요소에 대해 살펴보겠습니다.*
