---
layout: post
title: "Claude Skills 완벽 가이드 (08) - 실전 패턴"
date: 2026-02-07
permalink: /claude-skills-guide-08-patterns/
author: Anthropic
categories: [AI 에이전트, 개발 도구]
tags: [Claude, Skills, Patterns, Best Practices, Architecture]
original_url: "https://github.com/corca-ai/claude-plugins/tree/main/references/anthropic-skills-guide"
excerpt: "실전에서 검증된 5가지 스킬 디자인 패턴"
---

## 패턴 소개

이 패턴들은 **얼리 어답터와 내부 팀**이 만든 스킬에서 나타난 것입니다.

**중요:** 이것은 강제적인 템플릿이 아니라, 실제로 잘 작동하는 것으로 입증된 일반적인 접근 방식입니다.

---

## 접근법 선택: 문제 우선 vs. 도구 우선

### 문제 우선 (Problem-First)

**시작점:** "프로젝트 워크스페이스를 설정해야 해"

**특징:**
- 사용자는 결과를 설명
- 스킬이 올바른 순서로 올바른 MCP 호출을 오케스트레이션
- 도구는 수단, 결과가 목적

**예시:** "Q4 계획 프로젝트 만들어줘"
→ 스킬이 모든 단계를 자동 실행

---

### 도구 우선 (Tool-First)

**시작점:** "Notion MCP가 연결되어 있어"

**특징:**
- 사용자는 액세스 권한 보유
- 스킬이 최적 워크플로우와 베스트 프랙티스 제공
- 도구는 있지만 전문성이 필요

**예시:** "Notion으로 뭘 할 수 있어?"
→ 스킬이 가능한 워크플로우 안내

---

**대부분의 스킬은 한 방향으로 치우칩니다.** 자신의 유스케이스가 어느 쪽인지 아는 것이 올바른 패턴을 선택하는 데 도움이 됩니다.

---

## Pattern 1: Sequential Workflow Orchestration
## 순차 워크플로우 오케스트레이션

**언제 사용:** 사용자가 **특정 순서로 다단계 프로세스**가 필요할 때

---

### 구조

```markdown
## Workflow: Onboard New Customer

### Step 1: Create Account
Call MCP tool: `create_customer`
Parameters: name, email, company

**Expected result:**
- Customer ID generated
- Account status: pending

---

### Step 2: Setup Payment
Call MCP tool: `setup_payment_method`
Parameters: customer_id (from Step 1)

**Wait for:** Payment method verification
**Timeout:** 30 seconds

---

### Step 3: Create Subscription
Call MCP tool: `create_subscription`
Parameters:
- plan_id (from user input or default: "standard")
- customer_id (from Step 1)

**Validation:**
- Payment method verified in Step 2
- Plan ID exists and is active

---

### Step 4: Send Welcome Email
Call MCP tool: `send_email`
Template: welcome_email_template
Parameters:
- to: customer email
- subject: "Welcome to [Company]"
- body: Use template with customer_name, login_link

---

### Rollback on Failure

If any step fails:
1. Log the error
2. Reverse completed steps:
   - Delete subscription (if created)
   - Remove payment method (if added)
   - Mark account as "failed onboarding"
3. Notify user with specific error message
```

---

### 핵심 기법

- ✅ **명시적 단계 순서**
  - 각 단계가 명확히 구분됨
  - 의존성이 명확함

- ✅ **단계 간 의존성**
  - Step 2는 Step 1의 customer_id 필요
  - Step 3는 Step 2의 검증 결과 필요

- ✅ **각 단계 검증**
  - 예상 결과 명시
  - 성공 조건 확인

- ✅ **실패 시 롤백 지침**
  - 어떤 단계를 되돌릴지 명확
  - 데이터 일관성 보장

---

## Pattern 2: Multi-MCP Coordination
## 다중 MCP 조율

**언제 사용:** 워크플로우가 **여러 서비스에 걸쳐** 있을 때

---

### 예시: Design-to-Development Handoff

```markdown
# Design-to-Development Handoff Workflow

## Phase 1: Design Export (Figma MCP)

### Step 1.1: Export Assets
```bash
mcp-tool call figma export_assets \
  --file-key ${FIGMA_FILE_KEY} \
  --format png \
  --scale 2x
```

**Output:** Array of asset URLs

### Step 1.2: Generate Specifications
```bash
mcp-tool call figma get_file_components \
  --file-key ${FIGMA_FILE_KEY}
```

**Output:** Component specifications (colors, spacing, typography)

### Step 1.3: Create Asset Manifest
```json
{
  "project": "Q4 Dashboard",
  "assets": [...],
  "specifications": {...}
}
```

---

## Phase 2: Asset Storage (Google Drive MCP)

### Step 2.1: Create Project Folder
```bash
mcp-tool call drive create_folder \
  --name "Q4 Dashboard - Design Assets" \
  --parent-id ${TEAM_DRIVE_ID}
```

**Output:** Folder ID

### Step 2.2: Upload Assets
For each asset from Phase 1:
```bash
mcp-tool call drive upload_file \
  --file ${asset_path} \
  --folder-id ${FOLDER_ID}
  --sharing "anyone_with_link"
```

**Output:** Array of shareable links

### Step 2.3: Upload Manifest
```bash
mcp-tool call drive upload_file \
  --file "manifest.json" \
  --folder-id ${FOLDER_ID}
```

---

## Phase 3: Task Creation (Linear MCP)

### Step 3.1: Create Parent Issue
```bash
mcp-tool call linear create_issue \
  --title "Q4 Dashboard - Implementation" \
  --description "Design assets ready for development" \
  --label "design-ready"
```

**Output:** Parent issue ID

### Step 3.2: Create Sub-Tasks
For each component in manifest:
```bash
mcp-tool call linear create_issue \
  --title "Implement ${component.name}" \
  --description "Design: ${drive_link}\nSpecs: ${specifications}" \
  --parent-id ${PARENT_ISSUE_ID} \
  --estimate ${calculate_estimate(component)}
```

### Step 3.3: Assign to Team
```bash
mcp-tool call linear update_issue \
  --issue-id ${ISSUE_ID} \
  --assignee-id ${FRONTEND_ENGINEER_ID}
```

---

## Phase 4: Notification (Slack MCP)

### Step 4.1: Format Summary
```markdown
🎨 **Design Handoff Complete: Q4 Dashboard**

📁 Assets: ${drive_folder_link}
📋 Tasks: ${linear_project_link}

**Components:**
- Header (8 points) → @alice
- Sidebar (5 points) → @bob
- Dashboard Grid (13 points) → @charlie

**Next Steps:**
1. Review design specs
2. Technical planning meeting (tomorrow 2pm)
3. Implementation starts Monday
```

### Step 4.2: Post to Channel
```bash
mcp-tool call slack post_message \
  --channel "#engineering" \
  --message "${summary}" \
  --mentions "${ENGINEERING_TEAM_ID}"
```

---

## Error Handling

### Phase 1 Fails (Figma)
- Verify Figma file access
- Check API key permissions
- Retry with exponential backoff

### Phase 2 Fails (Drive)
- Verify Drive folder permissions
- Check storage quota
- DO NOT proceed to Phase 3 (no assets to link)

### Phase 3 Fails (Linear)
- Assets are safe in Drive
- Can retry task creation independently
- Manual fallback: Share Drive link directly

### Phase 4 Fails (Slack)
- Workflow still succeeded (assets + tasks created)
- Notify via email as fallback
- Log for manual follow-up
```

---

### 핵심 기법

- ✅ **명확한 페이즈 분리**
  - 각 서비스가 독립된 페이즈
  - 페이즈 간 경계 명확

- ✅ **페이즈 간 데이터 전달**
  - Phase 1 출력 → Phase 2 입력
  - Phase 2 출력 → Phase 3 입력

- ✅ **다음 페이즈 전 검증**
  - 각 페이즈 완료 확인
  - 필요한 데이터 준비 확인

- ✅ **중앙 집중식 에러 처리**
  - 각 페이즈 실패 시나리오 정의
  - 폴백 전략 명확

---

## Pattern 3: Iterative Refinement
## 반복 개선

**언제 사용:** 반복을 통해 **출력 품질이 향상**될 때

---

### 예시: Report Generation

```markdown
# Iterative Report Creation

## Initial Draft

### Step 1: Fetch Data
```bash
mcp-tool call analytics fetch_data \
  --start-date "2025-01-01" \
  --end-date "2025-01-31" \
  --metrics "revenue,users,conversion"
```

### Step 2: Generate First Draft
```python
python scripts/generate_report.py \
  --data data.json \
  --template quarterly_report \
  --output draft_report.md
```

**Save to:** `temp/draft_v1.md`

---

## Quality Check

### Step 3: Run Validation
```bash
python scripts/check_report.py \
  --input temp/draft_v1.md \
  --output issues.json
```

**Checks:**
1. **Structure:**
   - Executive summary present?
   - All required sections included?
   - Proper heading hierarchy?

2. **Data:**
   - All metrics calculated?
   - No missing values?
   - Trends analyzed?

3. **Formatting:**
   - Tables formatted correctly?
   - Charts embedded?
   - Citations present?

**Output:** List of issues with severity

---

## Refinement Loop

### Step 4: Address Issues
```python
issues = load_issues("issues.json")

for issue in issues.sorted_by_severity():
    if issue.type == "missing_section":
        # Regenerate missing section
        section = generate_section(issue.section_name, data)
        insert_section(draft, section, issue.position)

    elif issue.type == "data_validation":
        # Recalculate metric
        correct_value = recalculate_metric(issue.metric, data)
        replace_value(draft, issue.location, correct_value)

    elif issue.type == "formatting":
        # Fix formatting
        apply_format(draft, issue.location, issue.correct_format)
```

### Step 5: Re-validate
```bash
python scripts/check_report.py \
  --input temp/draft_v2.md \
  --output issues_v2.json
```

### Step 6: Repeat Until Quality Threshold
```python
iteration = 2
max_iterations = 5

while issues_remaining() and iteration < max_iterations:
    address_issues()
    re_validate()
    iteration += 1

    if critical_issues() == 0 and minor_issues() <= 3:
        break  # Quality threshold met
```

---

## Finalization

### Step 7: Apply Final Formatting
```bash
python scripts/format_final.py \
  --input temp/draft_v${iteration}.md \
  --output reports/Q1_2025_Report.pdf \
  --style corporate_template
```

### Step 8: Generate Summary
```markdown
## Report Generation Summary

- **Total iterations:** ${iteration}
- **Issues addressed:** ${total_issues_fixed}
- **Final quality score:** ${quality_score}/100
- **Time elapsed:** ${elapsed_time}

**Sections:**
✓ Executive Summary
✓ Revenue Analysis
✓ User Growth Trends
✓ Conversion Funnel
✓ Recommendations
```

### Step 9: Save and Deliver
```bash
# Save final version
mv reports/Q1_2025_Report.pdf /output/

# Upload to storage
mcp-tool call drive upload_file \
  --file /output/Q1_2025_Report.pdf \
  --folder-id ${REPORTS_FOLDER}
```
```

---

### 핵심 기법

- ✅ **명시적 품질 기준**
  - 무엇이 "좋은" 출력인지 정의
  - 측정 가능한 메트릭

- ✅ **반복적 개선**
  - 초안 → 검증 → 수정 → 재검증
  - 점진적 품질 향상

- ✅ **검증 스크립트**
  - 자동화된 품질 체크
  - 일관된 기준 적용

- ✅ **반복 중단 조건**
  - 최대 반복 횟수 설정
  - 품질 임계값 도달 시 중단

---

## Pattern 4: Context-Aware Tool Selection
## 컨텍스트 인식 도구 선택

**언제 사용:** 동일한 결과를 위해 **컨텍스트에 따라 다른 도구** 사용

---

### 예시: Smart File Storage

```markdown
# Smart File Storage

## Decision Tree

### Step 1: Analyze File
```python
file_info = {
    "type": detect_file_type(filename),
    "size": get_file_size(filepath),
    "content": analyze_content(filepath),
    "context": get_user_context()
}
```

### Step 2: Determine Best Storage

```python
def select_storage(file_info):
    # Large files → Cloud storage
    if file_info["size"] > 10_000_000:  # 10MB
        return "cloud_storage_mcp"

    # Collaborative documents → Docs platform
    if file_info["type"] in ["docx", "gdoc", "notion"]:
        if file_info["context"]["team_collaboration"]:
            return "docs_mcp"

    # Code files → Version control
    if file_info["type"] in ["py", "js", "ts", "go", "rs"]:
        return "github_mcp"

    # Temporary files → Local storage
    if file_info["context"]["temporary"]:
        return "local_storage"

    # Default → Cloud storage
    return "cloud_storage_mcp"
```

---

## Execute Storage

```python
storage_choice = select_storage(file_info)

if storage_choice == "cloud_storage_mcp":
    result = store_in_cloud(filepath, file_info)

elif storage_choice == "docs_mcp":
    result = store_in_docs(filepath, file_info)
    # Apply service-specific metadata
    add_metadata(result.file_id, {
        "team": file_info["context"]["team"],
        "project": file_info["context"]["project"],
        "sharing": "team"
    })

elif storage_choice == "github_mcp":
    result = store_in_github(filepath, file_info)
    # Commit with context
    commit_message = f"Add {filename} for {file_info['context']['feature']}"

elif storage_choice == "local_storage":
    result = store_locally(filepath, file_info)
```

---

## Provide Context to User

```markdown
📁 **File Stored:** ${filename}

**Location:** ${storage_choice}
**Reason:** ${explain_choice(storage_choice, file_info)}
**Access:** ${result.access_link}

**Why this storage?**
${detailed_explanation}
```

**Examples:**

```
✓ Stored in Google Drive
  Reason: Collaborative document (20 team members need access)

✓ Stored in GitHub
  Reason: Code file in active project repository

✓ Stored locally
  Reason: Temporary analysis file (will be deleted after use)
```
```

---

### 핵심 기법

- ✅ **명확한 결정 기준**
  - 각 선택의 조건 명시
  - 우선순위 정의

- ✅ **폴백 옵션**
  - 기본 선택 제공
  - 실패 시 대안

- ✅ **선택에 대한 투명성**
  - 왜 그 도구를 선택했는지 설명
  - 사용자가 이해할 수 있도록

---

## Pattern 5: Domain-Specific Intelligence
## 도메인 특화 지능

**언제 사용:** 스킬이 **도구 액세스를 넘어 전문 지식**을 추가할 때

---

### 예시: Financial Compliance

```markdown
# Payment Processing with Compliance

## Before Processing: Compliance Check

### Step 1: Fetch Transaction Details
```bash
mcp-tool call payments get_transaction \
  --transaction-id ${TX_ID}
```

**Output:**
```json
{
  "amount": 50000,
  "currency": "USD",
  "from_country": "US",
  "to_country": "IR",
  "customer_id": "cust_123"
}
```

---

### Step 2: Apply Compliance Rules

```python
compliance_result = check_compliance(transaction)

# Rule 1: Sanctions Check
if transaction["to_country"] in SANCTIONED_COUNTRIES:
    compliance_result.block(
        reason="Destination country under sanctions",
        regulation="OFAC regulations",
        action="automatic_block"
    )

# Rule 2: Amount Threshold
if transaction["amount"] > 10000:
    if not customer_has_kyc_verified(transaction["customer_id"]):
        compliance_result.flag(
            reason="Large transaction without KYC verification",
            regulation="BSA/AML requirements",
            action="manual_review"
        )

# Rule 3: Jurisdiction Check
if not jurisdiction_allowed(
    transaction["from_country"],
    transaction["to_country"]
):
    compliance_result.block(
        reason="Cross-border transaction not permitted",
        regulation="Local banking regulations",
        action="automatic_block"
    )

# Rule 4: Risk Assessment
risk_score = assess_transaction_risk(transaction)
if risk_score > 75:
    compliance_result.flag(
        reason=f"High risk score: {risk_score}",
        regulation="Internal risk policy",
        action="enhanced_review"
    )
```

---

### Step 3: Document Decision

```python
compliance_record = {
    "transaction_id": transaction["id"],
    "timestamp": now(),
    "checks_performed": [
        "sanctions_screening",
        "kyc_verification",
        "jurisdiction_validation",
        "risk_assessment"
    ],
    "result": compliance_result.status,  # passed | flagged | blocked
    "details": compliance_result.details,
    "reviewed_by": "automated_compliance_skill",
    "regulations_applied": compliance_result.regulations
}

# Store compliance record
mcp-tool call compliance create_record \
  --data ${compliance_record}
```

---

## Processing

### If Compliance Passed
```bash
# Proceed with transaction
mcp-tool call payments process_transaction \
  --transaction-id ${TX_ID} \
  --compliance-check-id ${compliance_record.id}

# Apply appropriate fraud checks
if transaction["amount"] > 1000:
    mcp-tool call fraud_detection enhanced_check \
      --transaction-id ${TX_ID}
```

### If Flagged for Review
```bash
# Create compliance case
mcp-tool call compliance create_case \
  --transaction-id ${TX_ID} \
  --priority "high" \
  --assigned-to "compliance_team"

# Notify compliance team
mcp-tool call slack post_message \
  --channel "#compliance-alerts" \
  --message "Transaction ${TX_ID} requires manual review"

# Hold transaction
mcp-tool call payments hold_transaction \
  --transaction-id ${TX_ID} \
  --reason "pending_compliance_review"
```

### If Blocked
```bash
# Block transaction
mcp-tool call payments block_transaction \
  --transaction-id ${TX_ID} \
  --reason ${compliance_result.reason}

# Notify customer
mcp-tool call notifications send_email \
  --to ${customer_email} \
  --template "transaction_blocked" \
  --data ${compliance_result.customer_message}

# Create audit log
mcp-tool call audit log_event \
  --type "transaction_blocked" \
  --transaction-id ${TX_ID} \
  --reason ${compliance_result.reason} \
  --regulation ${compliance_result.regulation}
```

---

## Audit Trail

```markdown
## Compliance Audit Trail

**Transaction:** ${TX_ID}
**Date:** ${timestamp}
**Status:** ${compliance_result.status}

**Checks Performed:**
✓ Sanctions screening (passed)
✓ KYC verification (passed)
✗ Jurisdiction validation (failed)
  → Cross-border transaction not permitted
  → Regulation: Local banking regulations
  → Action: Automatic block

**Decision:** BLOCKED
**Reviewed by:** Automated Compliance Skill v2.1
**Regulations Applied:** OFAC, BSA/AML, Local banking laws

**Audit Record ID:** ${compliance_record.id}
**Retrievable:** Yes (7 years retention)
```
```

---

### 핵심 기법

- ✅ **도메인 전문 지식 내장**
  - 규정 준수 규칙 내장
  - 산업 베스트 프랙티스 적용

- ✅ **실행 전 검증**
  - 중요한 작업 전 확인
  - 규제 요구사항 충족

- ✅ **포괄적인 문서화**
  - 모든 결정 기록
  - 감사 추적 생성

- ✅ **명확한 거버넌스**
  - 누가, 무엇을, 언제, 왜 명확
  - 책임 소재 분명

---

## 패턴 선택 가이드

| 유스케이스 | 권장 패턴 |
|----------|---------|
| 단계별 프로세스 자동화 | Pattern 1: Sequential Workflow |
| 여러 서비스 통합 | Pattern 2: Multi-MCP Coordination |
| 품질이 중요한 출력 | Pattern 3: Iterative Refinement |
| 상황별 도구 선택 | Pattern 4: Context-Aware Selection |
| 규제/전문성 필요 | Pattern 5: Domain-Specific Intelligence |

**혼합 가능:** 여러 패턴을 조합하여 사용할 수 있습니다.

---

## 다음 단계

패턴을 선택했다면:

1. 자신의 유스케이스에 맞게 조정
2. 테스트 케이스 정의
3. 구현 및 반복
4. 사용자 피드백 수집

---

*다음 글에서는 일반적인 문제 해결 방법을 다룹니다.*
