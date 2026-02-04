---
layout: post
title: "MiniMind 완벽 가이드 (7) - RLHF 강화 학습"
date: 2025-02-04
permalink: /minimind-guide-07-rlhf/
author: jingyaogong
category: AI
tags: [MiniMind, RLHF, DPO, PPO, GRPO, Reinforcement Learning]
series: minimind-guide
part: 7
original_url: "https://github.com/jingyaogong/minimind"
excerpt: "MiniMind의 DPO, PPO, GRPO 강화 학습 구현을 분석합니다."
---

## RLHF 개요

**RLHF(Reinforcement Learning from Human Feedback)**는 인간 선호도를 기반으로 모델을 정렬하는 기술입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                                 │
│                                                                  │
│   SFT Model ──▶ Preference Data ──▶ RL Training ──▶ Aligned Model │
│                                                                  │
│   MiniMind 지원 알고리즘:                                        │
│   • DPO (Direct Preference Optimization)                         │
│   • PPO (Proximal Policy Optimization)                          │
│   • GRPO (Group Relative Policy Optimization)                    │
│   • SPO (Self-Play Optimization)                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## DPO (Direct Preference Optimization)

DPO는 보상 모델 없이 **직접 선호도 최적화**를 수행합니다.

### DPO 손실 함수

```
L_DPO = -E[log σ(β(log π(y_w|x) - log π_ref(y_w|x)
                  - log π(y_l|x) + log π_ref(y_l|x)))]

y_w: 선호 응답 (winner)
y_l: 비선호 응답 (loser)
π: 현재 정책
π_ref: 참조 정책 (SFT 모델)
β: 온도 파라미터
```

### DPO 데이터 형식

```jsonl
{
  "prompt": "파이썬의 장점은 무엇인가요?",
  "chosen": "파이썬은 읽기 쉬운 문법, 풍부한 라이브러리, 다양한 응용 분야를 지원합니다.",
  "rejected": "파이썬은 좋은 언어입니다."
}
```

### DPO 구현

```python
# trainer/train_dpo.py

import torch
import torch.nn.functional as F

def compute_dpo_loss(
    model,
    ref_model,
    input_ids,
    chosen_ids,
    rejected_ids,
    beta: float = 0.1,
):
    """DPO 손실 계산"""
    # 현재 정책의 로그 확률
    chosen_logprobs = get_log_probs(model, input_ids, chosen_ids)
    rejected_logprobs = get_log_probs(model, input_ids, rejected_ids)

    # 참조 정책의 로그 확률
    with torch.no_grad():
        ref_chosen_logprobs = get_log_probs(ref_model, input_ids, chosen_ids)
        ref_rejected_logprobs = get_log_probs(ref_model, input_ids, rejected_ids)

    # 로그 비율 계산
    chosen_rewards = beta * (chosen_logprobs - ref_chosen_logprobs)
    rejected_rewards = beta * (rejected_logprobs - ref_rejected_logprobs)

    # DPO 손실
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss


def get_log_probs(model, input_ids, target_ids):
    """토큰별 로그 확률 계산"""
    logits = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)

    # 타겟 토큰의 로그 확률 추출
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    return target_log_probs.sum(dim=-1)
```

---

## PPO (Proximal Policy Optimization)

PPO는 **클리핑 기반 정책 최적화**로 안정적인 학습을 제공합니다.

### PPO 구성요소

```
┌─────────────────────────────────────────────────────────────────┐
│                    PPO Components                                │
│                                                                  │
│   Actor (정책 모델) ◀───────────────────────────▶ Critic (가치 모델) │
│          │                                              │       │
│          ▼                                              ▼       │
│   Action (응답 생성)                           Value (가치 추정) │
│          │                                              │       │
│          └──────────────▶ Reward ◀──────────────────────┘       │
│                            │                                    │
│                            ▼                                    │
│                    Advantage 계산                               │
│                            │                                    │
│                            ▼                                    │
│                    정책 업데이트 (클리핑)                        │
└─────────────────────────────────────────────────────────────────┘
```

### PPO 구현

```python
# trainer/train_ppo.py

class PPOTrainer:
    def __init__(
        self,
        actor_model,
        critic_model,
        ref_model,
        reward_model,
        clip_eps: float = 0.2,
        value_clip: float = 0.2,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.actor = actor_model
        self.critic = critic_model
        self.ref = ref_model
        self.reward = reward_model
        self.clip_eps = clip_eps
        self.value_clip = value_clip
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages(self, rewards, values, dones):
        """GAE(Generalized Advantage Estimation) 계산"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages)

    def ppo_step(self, batch):
        """PPO 업데이트 단계"""
        # 현재 정책의 로그 확률
        log_probs = self.actor.get_log_probs(batch['states'], batch['actions'])

        # 비율 계산
        ratio = torch.exp(log_probs - batch['old_log_probs'])

        # 클리핑된 목표 함수
        surr1 = ratio * batch['advantages']
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch['advantages']

        actor_loss = -torch.min(surr1, surr2).mean()

        # 가치 손실 (클리핑)
        values = self.critic(batch['states'])
        value_loss = F.mse_loss(values, batch['returns'])

        return actor_loss, value_loss
```

---

## GRPO (Group Relative Policy Optimization)

GRPO는 **그룹 내 상대적 비교**를 통해 최적화합니다.

```python
# trainer/train_grpo.py

def compute_grpo_loss(
    model,
    ref_model,
    prompts,
    group_size: int = 4,
    beta: float = 0.1,
):
    """GRPO 손실 계산"""
    all_losses = []

    for prompt in prompts:
        # 그룹 내 여러 응답 생성
        responses = []
        for _ in range(group_size):
            response = model.generate(prompt, temperature=1.0)
            responses.append(response)

        # 각 응답의 보상 계산
        rewards = [compute_reward(prompt, r) for r in responses]

        # 그룹 내 상대적 순위 계산
        ranks = torch.argsort(torch.tensor(rewards), descending=True)

        # 상대적 손실 계산
        for i, response in enumerate(responses):
            log_prob = get_log_prob(model, prompt, response)
            ref_log_prob = get_log_prob(ref_model, prompt, response)

            # 순위 기반 가중치
            weight = (group_size - ranks[i].item()) / group_size

            loss = -weight * beta * (log_prob - ref_log_prob)
            all_losses.append(loss)

    return torch.stack(all_losses).mean()
```

---

## RLAIF (AI Feedback)

MiniMind는 인간 피드백 대신 **AI 피드백**을 활용합니다.

```python
# trainer/train_reason.py

def generate_ai_feedback(model, prompt, response):
    """AI로 응답 품질 평가"""
    eval_prompt = f"""다음 응답을 1-10점으로 평가해주세요.

질문: {prompt}
응답: {response}

평가 기준:
1. 정확성
2. 관련성
3. 완전성

점수:"""

    score = model.generate(eval_prompt)
    return float(score.strip())
```

---

## 훈련 설정

```yaml
# config/rlhf.yaml

dpo:
  beta: 0.1
  learning_rate: 5e-7
  epochs: 3

ppo:
  clip_eps: 0.2
  value_clip: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  epochs: 4
  mini_batch_size: 8

grpo:
  group_size: 4
  beta: 0.1
  temperature: 1.0
```

---

## 알고리즘 비교

| 알고리즘 | 복잡도 | 보상 모델 | 메모리 | 안정성 |
|----------|--------|----------|--------|--------|
| **DPO** | 낮음 | 불필요 | 낮음 | 높음 |
| **PPO** | 높음 | 필요 | 높음 | 중간 |
| **GRPO** | 중간 | 불필요 | 중간 | 높음 |

---

*다음 글에서는 추론 및 배포를 살펴봅니다.*
