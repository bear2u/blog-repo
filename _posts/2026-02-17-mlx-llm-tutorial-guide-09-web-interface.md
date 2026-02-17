---
layout: post
title: "MLX LLM Tutorial 가이드 (09) - 웹 인터페이스: Flask API와 브라우저 데모 연결"
date: 2026-02-17
permalink: /mlx-llm-tutorial-guide-09-web-interface/
author: ddttom
categories: ['LLM 학습', '웹 데모']
tags: [Flask, Flask CORS, JavaScript, REST API, Visualization]
original_url: "https://github.com/ddttom/mlx-llm-tutorial/tree/main/code/web_interface"
excerpt: "web_interface 디렉토리의 서버/프런트 구조를 분석해 로컬 데모를 안정적으로 실행하는 방법을 정리합니다."
---

## 구성 요소

`code/web_interface/`는 크게 두 부분입니다.

- `server.py`: Flask API 서버
- `index.html`, `script.js`, `styles.css`: 프런트엔드

API 경로는 `/status`, `/models`, `/generate`, `/attention` 등을 제공합니다.

---

## 서버 실행

```bash
cd code/web_interface
python server.py --host 127.0.0.1 --port 8000 --model-dir ..
```

브라우저에서 `http://localhost:8000`으로 접속하면 UI를 확인할 수 있습니다.

---

## 모델 로딩 조건

simple LLM을 쓰려면 모델 파일이 필요합니다.

- `simple_llm_model.npz`
- `tokenizer.json`

이 파일들이 `model-dir` 기준 경로에 있어야 `simple-llm`이 로딩됩니다.

---

## 시각화 데이터의 성격

`server.py`를 보면 attention/token 확률 시각화는 현재 더미 생성 함수(`create_dummy_attention_data`, `create_dummy_token_probs`)를 사용합니다.

즉, UI 구조와 연동 방식 학습에는 좋지만,
실제 attention 해석 도구로 쓰려면 모델 내부 텐서 추출 로직을 추가해야 합니다.

