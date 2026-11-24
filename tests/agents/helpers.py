"""Agent test doubles."""

from dataclasses import dataclass

from ai_support.rag.schema import SearchResult
from ai_support.router.service import DepartmentScore, RouterPrediction


class FakeRouter:
    def __init__(self, department: str = "Registry", confidence: float = 0.9) -> None:
        self.department = department
        self.confidence = confidence
        self.audiences: list[str] = []

    def predict(self, audience: str, subject: str, description: str, top_k: int = 3):
        self.audiences.append(audience)
        return RouterPrediction(
            audience=audience,
            department=self.department,
            confidence=self.confidence,
            top_departments=(DepartmentScore(self.department, self.confidence),),
            model_name=f"{audience}-department-router",
            model_reference="test",
        )


class FakeKnowledge:
    def __init__(self, content: str = "Use the timetable page in MyUOWD.") -> None:
        self.content = content
        self.calls: list[tuple[str, str, str | None]] = []

    def search(self, audience: str, query: str, department: str | None = None):
        self.calls.append((audience, query, department))
        return [
            SearchResult(
                chunk_id=f"{audience}-chunk",
                document_id=f"{audience}-registry",
                audience=audience,
                department=department or "Registry",
                source=f"synthetic://{audience}/registry",
                content=self.content,
                score=1.0,
                citation=f"synthetic://{audience}/registry#{audience}-chunk",
            )
        ]


class EmptyKnowledge(FakeKnowledge):
    def search(self, audience: str, query: str, department: str | None = None):
        self.calls.append((audience, query, department))
        return []


class FakeLLM:
    def __init__(self, answer: str = "Open MyUOWD and check the timetable page.") -> None:
        self.response = answer
        self.calls: list[tuple[str, list[str]]] = []

    def answer(self, question: str, evidence: list[str]) -> str:
        self.calls.append((question, evidence))
        return self.response


@dataclass
class FakeEmailSender:
    sent: list

    def send(self, draft):
        from ai_support.agents.schema import EmailReceipt

        self.sent.append(draft)
        return EmailReceipt(action_id=draft.action_id, delivery_id="mailpit-test", delivered=True)
