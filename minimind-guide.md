---
layout: default
title: "MiniMind 완벽 가이드"
permalink: /minimind-guide/
---

<div class="guide-container">

# MiniMind 완벽 가이드

**3원(약 600원), 2시간**으로 25.8M 파라미터 LLM을 **처음부터(from scratch)** 훈련하는 MiniMind 프로젝트의 완벽 가이드입니다.

<div class="guide-meta">
<span class="author">원저자: jingyaogong</span>
<span class="source"><a href="https://github.com/jingyaogong/minimind">GitHub Repository</a></span>
</div>

---

## 목차

### Part 1: 기초
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">01</span>
<a href="{{ '/minimind-guide-01-intro/' | relative_url }}">소개 및 개요</a>
<p>MiniMind란? 프로젝트 구조, 빠른 시작, 기술 스택</p>
</div>

<div class="chapter-item">
<span class="chapter-number">02</span>
<a href="{{ '/minimind-guide-02-architecture/' | relative_url }}">모델 아키텍처</a>
<p>Transformer, RoPE, RMSNorm, GQA, MoE 구현</p>
</div>

<div class="chapter-item">
<span class="chapter-number">03</span>
<a href="{{ '/minimind-guide-03-tokenizer/' | relative_url }}">Tokenizer 훈련</a>
<p>BPE 알고리즘, 토크나이저 훈련 및 설정</p>
</div>

</div>

### Part 2: 훈련
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">04</span>
<a href="{{ '/minimind-guide-04-pretrain/' | relative_url }}">Pretrain 사전 훈련</a>
<p>Causal LM 훈련, 데이터 처리, 분산 훈련(DDP)</p>
</div>

<div class="chapter-item">
<span class="chapter-number">05</span>
<a href="{{ '/minimind-guide-05-sft/' | relative_url }}">SFT 지도 학습</a>
<p>ChatML 형식, 레이블 마스킹, 대화 능력 부여</p>
</div>

<div class="chapter-item">
<span class="chapter-number">06</span>
<a href="{{ '/minimind-guide-06-lora/' | relative_url }}">LoRA 미세조정</a>
<p>저랭크 적응, 효율적 미세조정, 어댑터 관리</p>
</div>

<div class="chapter-item">
<span class="chapter-number">07</span>
<a href="{{ '/minimind-guide-07-rlhf/' | relative_url }}">RLHF 강화 학습</a>
<p>DPO, PPO, GRPO 알고리즘 구현</p>
</div>

</div>

### Part 3: 배포
<div class="chapter-list">

<div class="chapter-item">
<span class="chapter-number">08</span>
<a href="{{ '/minimind-guide-08-inference/' | relative_url }}">추론 및 배포</a>
<p>Streamlit, OpenAI API, llama.cpp, vLLM, Ollama 통합</p>
</div>

</div>

---

## 핵심 개념

| 개념 | 설명 |
|------|------|
| **3원 훈련** | GPU 서버 2시간 렌탈 비용 |
| **25.8M 파라미터** | GPT-3의 1/7000 크기 |
| **From Scratch** | 모든 핵심 알고리즘 직접 구현 |
| **완전 오픈소스** | 코드 + 데이터셋 + 모델 공개 |
| **교육 목적** | LLM 내부 동작 원리 학습 |

---

## 빠른 시작

```bash
# 클론
git clone https://github.com/jingyaogong/minimind.git
cd minimind

# 사전 훈련
python trainer/train_pretrain.py

# SFT
python trainer/train_full_sft.py

# 데모
python scripts/web_demo.py
```

---

## 모델 시리즈

<div class="model-table">

| 모델 | 파라미터 | VRAM |
|------|----------|------|
| MiniMind2-small | 26M | 0.5 GB |
| MiniMind2 | 104M | 1.0 GB |
| MiniMind2-MoE | 145M | 1.0 GB |

</div>

---

<div class="guide-footer">
<p>이 가이드는 <a href="https://github.com/jingyaogong/minimind">MiniMind GitHub 저장소</a>를 분석하여 작성되었습니다.</p>
</div>

</div>

<style>
.guide-container {
  max-width: 800px;
  margin: 0 auto;
}

.guide-meta {
  display: flex;
  gap: 20px;
  color: #666;
  font-size: 0.9rem;
  margin-bottom: 20px;
}

.chapter-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
  margin: 20px 0;
}

.chapter-item {
  display: flex;
  align-items: flex-start;
  gap: 15px;
  padding: 15px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #6366f1;
}

.chapter-number {
  font-size: 1.5rem;
  font-weight: bold;
  color: #6366f1;
  min-width: 40px;
}

.chapter-item a {
  font-size: 1.1rem;
  font-weight: 600;
  color: #333;
  text-decoration: none;
}

.chapter-item a:hover {
  color: #6366f1;
}

.chapter-item p {
  margin: 5px 0 0 0;
  color: #666;
  font-size: 0.9rem;
}

.guide-footer {
  margin-top: 40px;
  padding-top: 20px;
  border-top: 1px solid #eee;
  text-align: center;
  color: #666;
}

table {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}

th, td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #eee;
}

th {
  background: #f8f9fa;
  font-weight: 600;
}
</style>
