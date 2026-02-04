---
layout: post
title: "MiniMind 완벽 가이드 (5) - SFT 지도 학습 미세조정"
date: 2025-02-04
permalink: /minimind-guide-05-sft/
author: jingyaogong
category: AI
tags: [MiniMind, SFT, Fine-tuning, Instruction, Chat]
series: minimind-guide
part: 5
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 SFT(Supervised Fine-Tuning) 구현을 분석합니다."
---

## SFT 개요

**SFT(Supervised Fine-Tuning)**는 사전 훈련된 모델을 **지시-응답 쌍**으로 미세조정하여 대화 능력을 부여하는 단계입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    SFT Pipeline                                  │
│                                                                  │
│   Pretrained Model ──▶ SFT Dataset ──▶ Fine-tuning ──▶ Chat Model │
│                                                                  │
│   데이터 형식:                                                   │
│   {"instruction": "...", "output": "..."}                       │
│   또는                                                          │
│   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} │
└─────────────────────────────────────────────────────────────────┘
```

---

## 데이터 형식

### 단일 턴 형식

```jsonl
{"instruction": "1+1은 얼마인가요?", "output": "2입니다."}
{"instruction": "파이썬이란 무엇인가요?", "output": "파이썬은 1991년에 만들어진 프로그래밍 언어입니다."}
```

### 멀티 턴 형식 (ChatML)

```jsonl
{
  "messages": [
    {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다."},
    {"role": "user", "content": "안녕하세요"},
    {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
    {"role": "user", "content": "오늘 날씨 어때요?"},
    {"role": "assistant", "content": "죄송합니다, 저는 실시간 날씨 정보에 접근할 수 없습니다."}
  ]
}
```

---

## SFT 데이터셋 클래스

```python
# trainer/train_full_sft.py

class SFTDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if 'messages' in item:
            # ChatML 형식
            text = self.tokenizer.apply_chat_template(
                item['messages'],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            # 단일 턴 형식
            text = self._format_single_turn(item)

        # 토큰화
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 레이블 생성 (어시스턴트 응답 부분만 학습)
        labels = self._create_labels(input_ids, item)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def _format_single_turn(self, item):
        return f"""<|im_start|>user
{item['instruction']}<|im_end|>
<|im_start|>assistant
{item['output']}<|im_end|>"""

    def _create_labels(self, input_ids, item):
        """어시스턴트 응답 부분만 레이블로 설정"""
        labels = input_ids.clone()

        # 유저 입력 부분은 -100으로 마스킹 (손실 계산 제외)
        # 구현 상세...

        return labels
```

---

## 레이블 마스킹

SFT에서는 **어시스턴트 응답 부분만** 손실을 계산합니다.

```python
def create_labels_with_masking(input_ids, tokenizer):
    """유저 입력은 마스킹, 어시스턴트 응답만 학습"""
    labels = input_ids.clone()

    # 특수 토큰 ID
    im_start_id = tokenizer.encode('<|im_start|>')[0]
    im_end_id = tokenizer.encode('<|im_end|>')[0]
    assistant_id = tokenizer.encode('assistant')[0]

    # 어시스턴트 응답 영역 찾기
    in_assistant = False
    for i, token_id in enumerate(input_ids):
        if not in_assistant:
            labels[i] = -100  # 마스킹

        # <|im_start|>assistant 시작 감지
        if token_id == assistant_id and i > 0 and input_ids[i-1] == im_start_id:
            in_assistant = True

        # <|im_end|> 종료 감지
        if token_id == im_end_id and in_assistant:
            in_assistant = False

    return labels
```

---

## SFT 훈련 루프

```python
def train_sft(
    model: nn.Module,
    train_dataset: SFTDataset,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
):
    model = model.to(device)
    model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * epochs,
    )

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            logits, loss = model(input_ids, targets=labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # 저장
    torch.save(model.state_dict(), "minimind_sft.pt")
```

---

## 고품질 SFT 데이터

### 데이터 품질 체크리스트

| 항목 | 설명 |
|------|------|
| **다양성** | 다양한 주제와 스타일 |
| **정확성** | 사실적으로 정확한 응답 |
| **안전성** | 유해 콘텐츠 제외 |
| **일관성** | 응답 스타일 일관성 |
| **길이** | 적절한 응답 길이 |

### 데이터 품질 향상

```python
def filter_sft_data(data: list) -> list:
    """저품질 데이터 필터링"""
    filtered = []

    for item in data:
        # 너무 짧은 응답 제외
        if len(item['output']) < 10:
            continue

        # 반복적인 패턴 제외
        if has_repetition(item['output']):
            continue

        # 유해 콘텐츠 제외
        if is_harmful(item['output']):
            continue

        filtered.append(item)

    return filtered
```

---

## 평가

```python
def evaluate_sft(model, eval_prompts: list, tokenizer):
    """SFT 모델 평가"""
    model.eval()

    for prompt in eval_prompts:
        input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Q: {prompt}")
        print(f"A: {response}")
        print("-" * 50)
```

---

*다음 글에서는 LoRA 미세조정을 살펴봅니다.*
