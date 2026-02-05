---
layout: post
title: "TrendRadar 완벽 가이드 (4) - 알림 시스템"
date: 2025-02-04
permalink: /trendradar-guide-04-notification/
author: sansan0
categories: [개발 도구, TrendRadar]
tags: [TrendRadar, Notification, Telegram, WeChat, Slack]
original_url: "https://github.com/sansan0/TrendRadar"
excerpt: "TrendRadar의 다중 채널 알림 시스템을 분석합니다."
---

## 알림 시스템 개요

TrendRadar는 **10개 이상의 알림 채널**을 지원합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│                  Notification Channels                           │
│                                                                  │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
│   │Telegram │ │ WeChat  │ │DingTalk │ │ Feishu  │             │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
│                                                                  │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐             │
│   │  Slack  │ │  Email  │ │  ntfy   │ │  Bark   │             │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘             │
│                                                                  │
│   ┌─────────┐                                                   │
│   │ Webhook │                                                   │
│   └─────────┘                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Telegram

가장 널리 사용되는 알림 채널입니다.

### 설정

```yaml
# config/config.yaml

notifications:
  telegram:
    enabled: true
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}
    # 선택적 설정
    parse_mode: HTML  # HTML 또는 Markdown
    disable_preview: false
```

### 봇 생성 방법

1. @BotFather에게 `/newbot` 명령
2. 봇 이름과 username 설정
3. 발급된 `bot_token` 저장
4. 봇과 대화 시작 후 `chat_id` 확인

### 구현

```python
# trendradar/notification/telegram.py

class TelegramNotifier(BaseNotifier):
    """Telegram 봇 알림"""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, context: Context):
        super().__init__(context)
        self.token = context.config.notifications.telegram.bot_token
        self.chat_id = context.config.notifications.telegram.chat_id
        self.parse_mode = context.config.notifications.telegram.parse_mode

    async def send(self, news_items: List[NewsItem]) -> bool:
        message = self._format_message(news_items)

        url = f"{self.BASE_URL.format(token=self.token)}/sendMessage"

        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": True,
        }

        async with self.context.http_client.post(url, json=payload) as resp:
            return resp.status == 200

    def _format_message(self, items: List[NewsItem]) -> str:
        lines = ["📰 <b>TrendRadar 알림</b>\n"]

        for item in items[:10]:  # 최대 10개
            lines.append(f"• <a href='{item.url}'>{item.title}</a>")

            if item.summary:
                lines.append(f"  💡 {item.summary[:100]}...")

        lines.append(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        return "\n".join(lines)

    def is_enabled(self) -> bool:
        cfg = self.context.config.notifications.telegram
        return cfg.enabled and cfg.bot_token and cfg.chat_id
```

---

## WeChat (기업/개인)

### 기업 WeChat

```yaml
notifications:
  wechat_work:
    enabled: true
    corp_id: ${WECHAT_CORP_ID}
    agent_id: ${WECHAT_AGENT_ID}
    secret: ${WECHAT_SECRET}
```

```python
# trendradar/notification/wechat.py

class WeChatWorkNotifier(BaseNotifier):
    """기업 WeChat 알림"""

    TOKEN_URL = "https://qyapi.weixin.qq.com/cgi-bin/gettoken"
    SEND_URL = "https://qyapi.weixin.qq.com/cgi-bin/message/send"

    async def get_access_token(self) -> str:
        params = {
            "corpid": self.corp_id,
            "corpsecret": self.secret,
        }

        async with self.context.http_client.get(self.TOKEN_URL, params=params) as resp:
            data = await resp.json()
            return data["access_token"]

    async def send(self, news_items: List[NewsItem]) -> bool:
        token = await self.get_access_token()

        payload = {
            "touser": "@all",
            "msgtype": "textcard",
            "agentid": self.agent_id,
            "textcard": {
                "title": "TrendRadar 알림",
                "description": self._format_description(news_items),
                "url": news_items[0].url if news_items else "",
            }
        }

        url = f"{self.SEND_URL}?access_token={token}"
        async with self.context.http_client.post(url, json=payload) as resp:
            return resp.status == 200
```

### 개인 WeChat (WxPusher)

```yaml
notifications:
  wechat_personal:
    enabled: true
    app_token: ${WXPUSHER_APP_TOKEN}
    uid: ${WXPUSHER_UID}
```

---

## Slack

### 설정

```yaml
notifications:
  slack:
    enabled: true
    webhook_url: ${SLACK_WEBHOOK_URL}
    channel: "#news"
```

### 구현

```python
# trendradar/notification/slack.py

class SlackNotifier(BaseNotifier):
    """Slack Webhook 알림"""

    async def send(self, news_items: List[NewsItem]) -> bool:
        blocks = self._build_blocks(news_items)

        payload = {
            "channel": self.channel,
            "blocks": blocks,
        }

        async with self.context.http_client.post(
            self.webhook_url,
            json=payload
        ) as resp:
            return resp.status == 200

    def _build_blocks(self, items: List[NewsItem]) -> List[dict]:
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📰 TrendRadar 알림"
                }
            },
            {"type": "divider"}
        ]

        for item in items[:5]:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<{item.url}|{item.title}>*"
                }
            })

            if item.summary:
                blocks.append({
                    "type": "context",
                    "elements": [{
                        "type": "mrkdwn",
                        "text": f"💡 {item.summary[:200]}"
                    }]
                })

        return blocks
```

---

## Email

### 설정

```yaml
notifications:
  email:
    enabled: true
    smtp_server: smtp.gmail.com
    smtp_port: 587
    username: ${EMAIL_USERNAME}
    password: ${EMAIL_PASSWORD}
    from_addr: news@example.com
    to_addrs:
      - user1@example.com
      - user2@example.com
```

### 구현

```python
# trendradar/notification/email.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailNotifier(BaseNotifier):
    """Email 알림"""

    async def send(self, news_items: List[NewsItem]) -> bool:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"TrendRadar 알림 - {len(news_items)}개 새 뉴스"
        msg['From'] = self.from_addr
        msg['To'] = ", ".join(self.to_addrs)

        # HTML 본문
        html = self._build_html(news_items)
        msg.attach(MIMEText(html, 'html'))

        # SMTP 전송 (동기이므로 executor 사용)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_smtp, msg)

        return True

    def _send_smtp(self, msg):
        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

    def _build_html(self, items: List[NewsItem]) -> str:
        html = """
        <html>
        <body>
        <h2>📰 TrendRadar 알림</h2>
        <ul>
        """

        for item in items:
            html += f"""
            <li>
                <a href="{item.url}">{item.title}</a>
                <p>{item.summary or ''}</p>
            </li>
            """

        html += """
        </ul>
        </body>
        </html>
        """

        return html
```

---

## ntfy

오픈소스 푸시 알림 서비스입니다.

```yaml
notifications:
  ntfy:
    enabled: true
    server: https://ntfy.sh
    topic: my-trendradar
    priority: default  # min, low, default, high, urgent
```

```python
# trendradar/notification/ntfy.py

class NtfyNotifier(BaseNotifier):
    """ntfy 알림"""

    async def send(self, news_items: List[NewsItem]) -> bool:
        url = f"{self.server}/{self.topic}"

        for item in news_items[:5]:
            payload = {
                "title": "TrendRadar",
                "message": item.title,
                "click": item.url,
                "priority": self.priority,
            }

            await self.context.http_client.post(url, json=payload)

        return True
```

---

## 커스텀 Webhook

모든 HTTP 엔드포인트와 연동 가능합니다.

```yaml
notifications:
  webhook:
    enabled: true
    url: https://your-api.com/webhook
    method: POST
    headers:
      Authorization: "Bearer ${WEBHOOK_TOKEN}"
      Content-Type: "application/json"
```

```python
# trendradar/notification/webhook.py

class WebhookNotifier(BaseNotifier):
    """커스텀 Webhook 알림"""

    async def send(self, news_items: List[NewsItem]) -> bool:
        payload = {
            "event": "new_news",
            "timestamp": datetime.now().isoformat(),
            "items": [
                {
                    "title": item.title,
                    "url": item.url,
                    "source": item.source,
                    "summary": item.summary,
                }
                for item in news_items
            ]
        }

        async with self.context.http_client.request(
            method=self.method,
            url=self.url,
            headers=self.headers,
            json=payload
        ) as resp:
            return 200 <= resp.status < 300
```

---

## 알림 관리자

모든 알림 채널을 관리합니다.

```python
# trendradar/notification/manager.py

class NotifierManager:
    """알림 관리자"""

    NOTIFIER_CLASSES = {
        'telegram': TelegramNotifier,
        'wechat_work': WeChatWorkNotifier,
        'slack': SlackNotifier,
        'email': EmailNotifier,
        'ntfy': NtfyNotifier,
        'webhook': WebhookNotifier,
    }

    def __init__(self, context: Context):
        self.context = context
        self.notifiers: List[BaseNotifier] = []
        self._init_notifiers()

    def _init_notifiers(self):
        for name, cls in self.NOTIFIER_CLASSES.items():
            notifier = cls(self.context)
            if notifier.is_enabled():
                self.notifiers.append(notifier)
                self.context.logger.info(f"Notifier enabled: {name}")

    async def notify_all(self, news_items: List[NewsItem]):
        """모든 활성화된 채널로 알림 전송"""
        if not news_items:
            return

        tasks = [
            notifier.send(news_items)
            for notifier in self.notifiers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for notifier, result in zip(self.notifiers, results):
            if isinstance(result, Exception):
                self.context.logger.error(
                    f"Notifier {notifier.__class__.__name__} failed: {result}"
                )
```

---

*다음 글에서는 AI 분석과 MCP 통합을 살펴봅니다.*
