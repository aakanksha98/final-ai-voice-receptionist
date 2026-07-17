from typing import Literal

from backend.app.agent.state import AgentState


RouteDecision = Literal["ready", "invalid"]
PlannedRouteDecision = Literal[
    "small_talk",
    "rag",
    "booking",
    "cancellation",
    "reschedule",
    "escalation",
    "deferred",
]


def route_validated_input(state: AgentState) -> RouteDecision:
    input_status = state.get("input_status")

    if input_status == "valid":
        return "ready"

    if input_status == "invalid":
        return "invalid"

    raise ValueError("Input must be validated before routing")


def route_planned_intent(state: AgentState) -> PlannedRouteDecision:
    detected_intent = state.get("detected_intent")
    if detected_intent is None:
        raise ValueError("A planned intent is required before routing")

    if detected_intent == "small_talk":
        return "small_talk"

    if detected_intent == "rag":
        if state.get("query_scope") not in {None, "business_question"}:
            return "deferred"
        return "rag"

    if detected_intent == "book_appointment":
        return "booking"

    if detected_intent == "cancel_appointment":
        return "cancellation"

    if detected_intent == "reschedule_appointment":
        return "reschedule"

    if detected_intent == "human_escalation":
        return "escalation"

    return "deferred"
