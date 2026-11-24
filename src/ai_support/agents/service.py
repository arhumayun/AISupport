"""Application service for chat, email confirmation, and feedback."""

from ai_support.agents.actions import (
    FeedbackStore,
    JsonlAuditStore,
    PendingActionStore,
)
from ai_support.agents.email import EmailSender
from ai_support.agents.schema import AgentReply, EmailReceipt
from ai_support.agents.workflow import AgentService
from ai_support.security.auth import Principal


class EmailDeliveryError(RuntimeError):
    """Raised when confirmed synthetic email delivery fails."""


class ChatService:
    def __init__(
        self,
        agent: AgentService,
        actions: PendingActionStore,
        sender: EmailSender,
        feedback: FeedbackStore,
        audit: JsonlAuditStore,
    ) -> None:
        self.agent = agent
        self.actions = actions
        self.sender = sender
        self.feedback = feedback
        self.audit = audit

    def respond(self, principal: Principal, message: str) -> AgentReply:
        reply = self.agent.respond(principal, message)
        self.feedback.register(reply.message_id, principal)
        return reply

    def confirm_email(self, principal: Principal, action_id: str) -> EmailReceipt:
        draft = self.actions.get(action_id, principal)
        try:
            receipt = self.sender.send(draft)
        except (OSError, TimeoutError) as error:
            self.audit.record(
                {
                    "event": "department_email",
                    "audience": principal.audience,
                    "department": draft.department,
                    "action_id": draft.action_id,
                    "delivered": False,
                }
            )
            raise EmailDeliveryError("synthetic email delivery failed") from error
        self.actions.consume(action_id, principal)
        self.audit.record(
            {
                "event": "department_email",
                "audience": principal.audience,
                "department": draft.department,
                "action_id": draft.action_id,
                "delivery_id": receipt.delivery_id,
                "delivered": receipt.delivered,
            }
        )
        return receipt

    def record_feedback(self, principal: Principal, message_id: str, helpful: bool) -> None:
        self.feedback.record(message_id, helpful, principal)
