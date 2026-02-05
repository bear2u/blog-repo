---
layout: post
title: "WrenAI 완벽 가이드 (4) - MDL (Metadata Definition Language)"
date: 2025-02-05
permalink: /wrenai-guide-04-mdl/
author: Canner
categories: [AI 에이전트, WrenAI]
tags: [WrenAI, MDL, Schema, Semantic Layer, Data Modeling]
original_url: "https://github.com/Canner/WrenAI"
excerpt: "WrenAI의 핵심인 MDL(Metadata Definition Language)로 비즈니스 로직을 정의하는 방법을 알아봅니다."
---

## MDL이란?

**MDL (Metadata Definition Language)**은 WrenAI의 핵심 개념으로, 데이터베이스 스키마 위에 비즈니스 로직과 의미를 정의하는 의미론적 계층입니다.

```
┌─────────────────────────────────────────┐
│            사용자 질문                   │
│        "지난 분기 매출은?"              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│           MDL (의미론적 계층)            │
│  • 비즈니스 용어 정의                    │
│  • 관계 매핑                            │
│  • 계산 필드                            │
│  • 메트릭 정의                          │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│         물리적 데이터베이스              │
│    테이블, 컬럼, 조인                    │
└─────────────────────────────────────────┘
```

---

## MDL 구조

### 기본 스키마

```json
{
  "catalog": "my_database",
  "schema": "public",
  "models": [...],
  "relationships": [...],
  "metrics": [...],
  "views": [...]
}
```

---

## Models (모델)

모델은 데이터베이스 테이블을 의미론적으로 표현합니다.

```json
{
  "models": [
    {
      "name": "customers",
      "refSql": "SELECT * FROM public.customers",
      "columns": [
        {
          "name": "id",
          "type": "int",
          "notNull": true,
          "isHidden": false
        },
        {
          "name": "name",
          "type": "varchar",
          "properties": {
            "description": "고객 이름",
            "displayName": "고객명"
          }
        },
        {
          "name": "email",
          "type": "varchar"
        },
        {
          "name": "created_at",
          "type": "timestamp"
        }
      ],
      "primaryKey": ["id"],
      "properties": {
        "description": "고객 정보 테이블",
        "displayName": "고객"
      }
    }
  ]
}
```

### 컬럼 속성

| 속성 | 설명 | 필수 |
|-----|------|------|
| `name` | 컬럼 이름 | ✅ |
| `type` | 데이터 타입 | ✅ |
| `notNull` | NULL 허용 여부 | ❌ |
| `isHidden` | UI에서 숨김 | ❌ |
| `expression` | 계산 필드 수식 | ❌ |
| `isCalculated` | 계산 필드 여부 | ❌ |
| `properties` | 메타데이터 | ❌ |

---

## Calculated Fields (계산 필드)

SQL 표현식으로 가상 컬럼을 정의합니다.

```json
{
  "models": [
    {
      "name": "orders",
      "columns": [
        {
          "name": "id",
          "type": "int"
        },
        {
          "name": "quantity",
          "type": "int"
        },
        {
          "name": "unit_price",
          "type": "decimal"
        },
        {
          "name": "total_amount",
          "type": "decimal",
          "isCalculated": true,
          "expression": "quantity * unit_price",
          "properties": {
            "description": "주문 총액 (수량 × 단가)"
          }
        },
        {
          "name": "discounted_amount",
          "type": "decimal",
          "isCalculated": true,
          "expression": "quantity * unit_price * 0.9",
          "properties": {
            "description": "할인 적용 금액 (10% 할인)"
          }
        }
      ]
    }
  ]
}
```

---

## Relationships (관계)

테이블 간의 조인 관계를 정의합니다.

```json
{
  "relationships": [
    {
      "name": "customer_orders",
      "models": ["customers", "orders"],
      "joinType": "ONE_TO_MANY",
      "condition": "customers.id = orders.customer_id",
      "properties": {
        "description": "고객과 주문의 관계"
      }
    },
    {
      "name": "order_items",
      "models": ["orders", "order_items"],
      "joinType": "ONE_TO_MANY",
      "condition": "orders.id = order_items.order_id"
    },
    {
      "name": "item_product",
      "models": ["order_items", "products"],
      "joinType": "MANY_TO_ONE",
      "condition": "order_items.product_id = products.id"
    }
  ]
}
```

### 조인 타입

| 타입 | 설명 |
|------|------|
| `ONE_TO_ONE` | 1:1 관계 |
| `ONE_TO_MANY` | 1:N 관계 |
| `MANY_TO_ONE` | N:1 관계 |
| `MANY_TO_MANY` | N:M 관계 |

---

## Metrics (메트릭)

비즈니스 KPI와 집계 지표를 정의합니다.

```json
{
  "metrics": [
    {
      "name": "total_revenue",
      "type": "number",
      "baseModel": "orders",
      "expression": "SUM(total_amount)",
      "properties": {
        "description": "총 매출",
        "unit": "KRW",
        "format": "#,###"
      }
    },
    {
      "name": "average_order_value",
      "type": "number",
      "baseModel": "orders",
      "expression": "AVG(total_amount)",
      "properties": {
        "description": "평균 주문 금액"
      }
    },
    {
      "name": "customer_count",
      "type": "number",
      "baseModel": "customers",
      "expression": "COUNT(DISTINCT id)",
      "properties": {
        "description": "총 고객 수"
      }
    },
    {
      "name": "conversion_rate",
      "type": "number",
      "baseModel": "orders",
      "expression": "COUNT(DISTINCT customer_id) * 100.0 / (SELECT COUNT(*) FROM customers)",
      "properties": {
        "description": "전환율 (%)"
      }
    }
  ]
}
```

---

## Views (뷰)

복잡한 쿼리를 재사용 가능한 뷰로 정의합니다.

```json
{
  "views": [
    {
      "name": "monthly_sales",
      "statement": "SELECT DATE_TRUNC('month', order_date) as month, SUM(total_amount) as revenue FROM orders GROUP BY 1",
      "properties": {
        "description": "월별 매출 집계"
      }
    },
    {
      "name": "top_customers",
      "statement": "SELECT c.id, c.name, SUM(o.total_amount) as total_spent FROM customers c JOIN orders o ON c.id = o.customer_id GROUP BY c.id, c.name ORDER BY total_spent DESC LIMIT 100",
      "properties": {
        "description": "상위 100명 고객"
      }
    }
  ]
}
```

---

## 전체 MDL 예제

```json
{
  "catalog": "ecommerce",
  "schema": "public",
  "models": [
    {
      "name": "customers",
      "refSql": "SELECT * FROM customers",
      "columns": [
        { "name": "id", "type": "int", "notNull": true },
        { "name": "name", "type": "varchar" },
        { "name": "email", "type": "varchar" },
        { "name": "tier", "type": "varchar", "properties": { "description": "고객 등급 (Gold/Silver/Bronze)" } },
        { "name": "created_at", "type": "timestamp" }
      ],
      "primaryKey": ["id"],
      "properties": { "displayName": "고객" }
    },
    {
      "name": "orders",
      "refSql": "SELECT * FROM orders",
      "columns": [
        { "name": "id", "type": "int", "notNull": true },
        { "name": "customer_id", "type": "int" },
        { "name": "order_date", "type": "date" },
        { "name": "status", "type": "varchar" },
        { "name": "total_amount", "type": "decimal" },
        {
          "name": "quarter",
          "type": "varchar",
          "isCalculated": true,
          "expression": "CONCAT('Q', EXTRACT(QUARTER FROM order_date))"
        }
      ],
      "primaryKey": ["id"],
      "properties": { "displayName": "주문" }
    }
  ],
  "relationships": [
    {
      "name": "customer_orders",
      "models": ["customers", "orders"],
      "joinType": "ONE_TO_MANY",
      "condition": "customers.id = orders.customer_id"
    }
  ],
  "metrics": [
    {
      "name": "total_revenue",
      "type": "number",
      "baseModel": "orders",
      "expression": "SUM(total_amount)"
    },
    {
      "name": "order_count",
      "type": "number",
      "baseModel": "orders",
      "expression": "COUNT(*)"
    }
  ]
}
```

---

## MDL 배포

### UI에서 배포

1. **Modeling** 페이지에서 모델 정의
2. **Deploy** 버튼 클릭
3. 자동으로 Wren Engine에 배포

### API로 배포

```bash
# GraphQL Mutation
mutation {
  deployProject(projectId: 1) {
    success
    message
  }
}
```

---

## MDL 활용 팁

### 1. 비즈니스 용어 사용

```json
// ❌ 기술적 이름
{ "name": "cust_rev_ttl", "type": "decimal" }

// ✅ 비즈니스 용어
{
  "name": "total_customer_revenue",
  "type": "decimal",
  "properties": {
    "displayName": "고객별 총 매출",
    "description": "해당 고객의 모든 주문 금액 합계"
  }
}
```

### 2. 상세한 설명 추가

```json
{
  "name": "churn_rate",
  "expression": "...",
  "properties": {
    "description": "지난 90일간 주문이 없는 고객 비율",
    "formula": "(90일 비활성 고객 수 / 전체 고객 수) × 100",
    "unit": "%"
  }
}
```

### 3. 숨김 필드 활용

```json
// 내부 키는 숨기고 의미있는 필드만 노출
{ "name": "internal_id", "type": "int", "isHidden": true },
{ "name": "display_name", "type": "varchar", "isHidden": false }
```

---

*다음 글에서는 RAG 파이프라인의 작동 원리를 살펴봅니다.*
