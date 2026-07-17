import os
from typing import Self

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backend.app.agent.state import PlannerIntent, SmallTalkTopic


DEFAULT_PLANNER_MODEL = "gpt-5.4-mini"


class PlannerSlots(BaseModel):
    """Appointment details stated explicitly in the user's request."""

    model_config = ConfigDict(extra="forbid")

    service: str | None
    date: str | None
    time: str | None
    escalation_reason: str | None


class PlannerDecision(BaseModel):
    """A single supported route and any slots found by the planner."""

    model_config = ConfigDict(extra="forbid")

    intent: PlannerIntent
    confidence: float
    small_talk_topic: SmallTalkTopic | None
    slots: PlannerSlots

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def validate_small_talk_topic(self) -> Self:
        if self.intent == "small_talk" and self.small_talk_topic is None:
            raise ValueError(
                "small_talk_topic is required for the small_talk intent"
            )

        if self.intent != "small_talk" and self.small_talk_topic is not None:
            raise ValueError(
                "small_talk_topic is only valid for the small_talk intent"
            )

        return self


PlannerRunnable = Runnable[dict[str, str], PlannerDecision]


PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
"""You are the constrained intent planner for an enterprise receptionist.

Selected demo business type: {business_type}
Supported bookable services: {supported_services}

Choose exactly one route:
- small_talk: greetings, assistant identity, capabilities, or courtesy replies.
- rag: services, pricing, policies, or other business knowledge.
- book_appointment: create a new appointment.
- cancel_appointment: cancel an existing appointment.
- reschedule_appointment: move an existing appointment.
- human_escalation: the user explicitly asks for a person or human handoff,
  or says they are dissatisfied, unhappy, want to complain, or the resolution
  was not acceptable for this selected business or its appointment workflow.
- clarification: the request is unclear or no single route can be selected.

Do not choose human_escalation for unrelated technical support, family,
personal, general-knowledge, or outside-business problems, even if the user
uses words like issue, problem, help, support, or setup. Those are clarification
unless they directly concern the selected business profile or the user explicitly
asks for a human without adding an unrelated topic.

For small_talk, set small_talk_topic to greeting, assistant_identity,
capabilities, or courtesy. For every other route, set small_talk_topic to null.

Extract only details explicitly present in the current request or its relevant
unfinished prior request. Preserve date and time phrases as spoken. Use prior
conversation only to resolve a direct follow-up. Do not carry details into an
unrelated request. Cancellation and rescheduling use the current active
appointment in the conversation. Do not answer the request and do not call any
tool. Return a confidence between 0 and 1 for the route selection.""",
        ),
        (
            "human",
            """Prior conversation:
{conversation_history}

Classify the current request:
{user_message}""",
        ),
    ]
)


def create_openai_planner(model: str | None = None) -> PlannerRunnable:
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_PLANNER_MODEL)
    chat_model = ChatOpenAI(model=selected_model, max_retries=2)
    structured_model = chat_model.with_structured_output(
        PlannerDecision,
        method="json_schema",
        strict=True,
    )
    return PLANNER_PROMPT | structured_model
