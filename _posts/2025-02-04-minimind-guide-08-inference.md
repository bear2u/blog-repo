---
layout: post
title: "MiniMind 완벽 가이드 (8) - 추론 및 배포"
date: 2025-02-04
permalink: /minimind-guide-08-inference/
author: jingyaogong
categories: [LLM 학습, MiniMind]
tags: [MiniMind, Inference, Deployment, API, llama.cpp, vLLM]
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 추론 최적화, API 서버, 외부 생태계 통합을 분석합니다."
---

## 추론 개요

훈련된 MiniMind 모델을 실제 서비스에서 사용하기 위한 다양한 방법을 알아봅니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Options                             │
│                                                                  │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │  Streamlit  │  │ OpenAI API  │  │   llama.cpp │            │
│   │    Demo     │  │   Server    │  │    vLLM     │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│                          │                                       │
│                          ▼                                       │
│                  ┌─────────────┐                                │
│                  │   MiniMind  │                                │
│                  │    Model    │                                │
│                  └─────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 기본 추론

### 텍스트 생성

```python
import torch
from model.model_minimind import MiniMind, MiniMindConfig
from transformers import AutoTokenizer

def generate(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    model.eval()

    # 입력 토큰화
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

    # KV 캐시 초기화
    kv_cache = None

    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward
            logits = model(generated[:, -1:] if kv_cache else generated, kv_cache=kv_cache)

            # 다음 토큰 샘플링
            next_token_logits = logits[:, -1, :] / temperature

            # Top-p 필터링
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            next_token_logits[sorted_indices_to_remove] = float('-inf')

            # 샘플링
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 종료 조건
            if next_token.item() == tokenizer.eos_token_id:
                break

            generated = torch.cat([generated, next_token], dim=-1)

    # 디코딩
    response = tokenizer.decode(generated[0], skip_special_tokens=True)
    return response.split("assistant\n")[-1]
```

---

## Streamlit 데모

```python
# scripts/web_demo.py

import streamlit as st
import torch

st.title("MiniMind Chat")

# 모델 로드 (캐싱)
@st.cache_resource
def load_model():
    model = MiniMind.from_pretrained("model/")
    tokenizer = AutoTokenizer.from_pretrained("model/")
    return model, tokenizer

model, tokenizer = load_model()

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("메시지를 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = generate(model, tokenizer, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
```

```bash
# 실행
streamlit run scripts/web_demo.py
```

---

## OpenAI 호환 API 서버

```python
# scripts/serve_openai_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "minimind"
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[dict]
    usage: dict

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # 메시지 포맷팅
    prompt = request.messages[-1].content

    # 생성
    response = generate(
        model, tokenizer, prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return ChatCompletionResponse(
        id="chatcmpl-xxx",
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response},
            "finish_reason": "stop",
        }],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# 실행
python scripts/serve_openai_api.py

# 테스트
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "안녕하세요"}]}'
```

---

## llama.cpp 통합

MiniMind는 **llama.cpp**와 호환됩니다.

### 모델 변환

```python
# scripts/convert_model.py

def convert_to_gguf(model_path, output_path):
    """MiniMind → GGUF 변환"""
    # llama.cpp의 convert.py 사용
    import subprocess
    subprocess.run([
        "python", "llama.cpp/convert.py",
        model_path,
        "--outfile", output_path,
        "--outtype", "f16",
    ])
```

### llama.cpp 실행

```bash
# 빌드
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 실행
./main -m minimind.gguf -p "안녕하세요" -n 256
```

---

## vLLM 통합

고성능 추론을 위해 **vLLM**을 사용할 수 있습니다.

```python
from vllm import LLM, SamplingParams

# 모델 로드
llm = LLM(model="path/to/minimind")

# 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256,
)

# 생성
prompts = ["안녕하세요, MiniMind입니다."]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## Ollama 통합

{% raw %}
```bash
# Modelfile 생성
cat > Modelfile << EOF
FROM minimind.gguf
TEMPLATE """<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Ollama 모델 생성
ollama create minimind -f Modelfile

# 실행
ollama run minimind "안녕하세요"
```
{% endraw %}

---

## 성능 최적화

### KV 캐시

```python
class KVCache:
    def __init__(self, batch_size, max_seq_len, n_layers, n_heads, head_dim):
        self.k_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim)
        self.v_cache = torch.zeros(n_layers, batch_size, max_seq_len, n_heads, head_dim)
        self.seq_len = 0

    def update(self, layer_idx, k, v):
        seq_len = k.size(1)
        self.k_cache[layer_idx, :, self.seq_len:self.seq_len+seq_len] = k
        self.v_cache[layer_idx, :, self.seq_len:self.seq_len+seq_len] = v

    def get(self, layer_idx):
        return (
            self.k_cache[layer_idx, :, :self.seq_len],
            self.v_cache[layer_idx, :, :self.seq_len],
        )
```

### 양자화

```python
import torch.quantization as quant

# 동적 양자화
model_int8 = quant.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8,
)
```

---

## 마무리

MiniMind는 **완전히 처음부터** LLM을 훈련하는 최소한의 예제입니다. 핵심 가치:

- **교육 목적** - LLM 내부 동작 이해
- **저비용** - 3원, 2시간으로 훈련 가능
- **완전 오픈소스** - 코드, 데이터, 모델 공개
- **생태계 호환** - llama.cpp, vLLM, Ollama 지원

---

## 리소스

- **GitHub**: [github.com/jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- **Hugging Face**: [MiniMind Collection](https://huggingface.co/collections/jingyaogong/minimind-66caf8d999f5c7fa64f399e5)
- **ModelScope**: [gongjy Profile](https://www.modelscope.cn/profile/gongjy)
- **라이선스**: Apache 2.0

---

*이 가이드 시리즈가 MiniMind를 이해하고 활용하는 데 도움이 되길 바랍니다.*
