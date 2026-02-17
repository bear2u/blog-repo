---
layout: post
title: "MLX LLM Tutorial 가이드 (02) - 설치/환경 구성: Miniconda와 mlx-env로 재현 가능한 개발 환경 만들기"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-02-installation-and-environment/
author: ddttom
categories: ['LLM 학습', '개발 환경']
tags: [MLX, Miniconda, Python, Jupyter, Apple Silicon]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/blob/main/public/docs/installation.md"
excerpt: "설치 문서 기준으로 Miniconda, conda 환경, MLX 및 부가 의존성을 실제 커맨드 중심으로 정리합니다."
---

## 시스템 요구사항

문서 기준 최소 조건은 다음입니다.

- Apple Silicon Mac(M1/M2/M3)
- macOS 12+
- RAM 8GB 이상(권장 16GB)
- 여유 디스크 20GB+

---

## 작업 디렉토리 원칙

저장소 문서는 전용 디렉토리 사용을 강하게 권장합니다.

```bash
mkdir -p ~/ai-training
cd ~/ai-training
git clone https://github.com/ddttom/mlx-llm-tutorial
cd mlx-llm-tutorial
```

문서에는 `requirements.txt`를 `~/ai-training`로 복사해 거기서 설치하는 흐름도 제시됩니다. 핵심은 "한 곳에서 일관되게 관리"입니다.

---

## Miniconda + 환경 생성

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh | bash
source ~/.zshrc

conda create -n mlx-env python=3.10
conda activate mlx-env
```

이 프로젝트는 Python 3.10을 기준으로 설명합니다.

---

## 패키지 설치

`requirements.txt` 기준 핵심 패키지:

- `mlx`, `numpy`, `matplotlib`, `tqdm`
- `requests`, `huggingface_hub`
- `flask`, `flask-cors`

```bash
pip install -r requirements.txt
```

문서에는 Jupyter 설치도 포함되어 있습니다.

```bash
pip install jupyter
jupyter notebook
```

---

## 설치 검증

```bash
python -c "import mlx; print(mlx.__version__)"
```

여기까지 통과하면 다음 장의 MLX 개념/코드 실습으로 바로 넘어갈 수 있습니다.

