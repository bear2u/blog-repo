---
layout: post
title: "microgpt.py 가이드 (03) - Value 클래스 오토그라드: 체인룰을 직접 구현하기"
date: 2026-02-19
permalink: /microgpt-guide-03-value-class-and-autograd/
author: Andrej Karpathy
categories: ['LLM 학습', '수학/오토그라드']
tags: [Autograd, Value, Backpropagation, Chain Rule, microgpt]
original_url: "https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py"
excerpt: "Value 객체가 연산 그래프를 저장하고 backward에서 gradient를 전파하는 과정을 상세히 풉니다."
---

## Value 객체의 4개 핵심 필드

`__slots__ = ('data', 'grad', '_children', '_local_grads')`

- `data`: 순전파 결과 스칼라
- `grad`: 손실 기준 미분값
- `_children`: 현재 노드 입력 노드들
- `_local_grads`: 현재 노드가 각 child에 미치는 local 미분

---

## 연산자 오버로딩의 의미

예: `__mul__`

```python
return Value(self.data * other.data, (self, other), (other.data, self.data))
```

`z = x * y`일 때

- dz/dx = y
- dz/dy = x

를 `_local_grads`에 저장해 두는 구조입니다.

---

## backward 핵심 로직 (59~72행)

1. DFS로 토폴로지 정렬(`build_topo`)
2. 최종 loss.grad = 1 설정
3. 역순 순회하며 `child.grad += local_grad * v.grad`

즉 수식으로 쓰면:

`dL/dchild += dL/dv * dv/dchild`

---

## 미니멀하지만 강력한 이유

이 구현은 텐서 브로드캐스팅/배치 연산은 없지만,
GPT forward에 필요한 모든 스칼라 연산 체인을 다룰 수 있습니다.

그래서 attention/log-softmax/optimizer 업데이트까지 전부 같은 추상화 위에서 동작합니다.

---

## 디버깅 팁

학습이 안 되면 먼저 확인할 것:

- `loss.backward()` 뒤 특정 파라미터 grad가 0만 나오는지
- `log()` 입력이 0 이하로 가는지
- `exp()`가 과도하게 커지는지

다음 장에서 이 `Value`를 파라미터 저장 구조에 어떻게 연결하는지 봅니다.
