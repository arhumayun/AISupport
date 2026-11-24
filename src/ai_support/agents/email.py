"""Confirmation-gated Mailpit SMTP delivery adapter."""

import smtplib
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Protocol

from ai_support.agents.schema import EmailDraft, EmailReceipt


class EmailSender(Protocol):
    def send(self, draft: EmailDraft) -> EmailReceipt: ...


class MailpitEmailSender:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 1025,
        sender: str = "assistant@example.test",
        timeout: float = 10.0,
    ) -> None:
        self.host = host
        self.port = port
        self.sender = sender
        self.timeout = timeout

    def send(self, draft: EmailDraft) -> EmailReceipt:
        message = EmailMessage()
        delivery_id = make_msgid(domain="example.test")
        message["Message-ID"] = delivery_id
        message["From"] = self.sender
        message["To"] = draft.recipient
        message["Subject"] = draft.subject
        message.set_content(draft.body)
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as client:
            client.send_message(message)
        return EmailReceipt(action_id=draft.action_id, delivery_id=delivery_id, delivered=True)
