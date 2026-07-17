import re
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.tools.escalation import (
    HumanEscalationRequest,
    mock_human_escalation_tool,
)


def escalation_decision(
    reason: str | None = "The billing issue is unresolved",
) -> PlannerDecision:
    return PlannerDecision(
        intent="human_escalation",
        confidence=0.99,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=reason,
        ),
    )


def create_recording_escalation_tool(
    calls: list[dict[str, str | None]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_human_escalation", args_schema=HumanEscalationRequest)
    def record_human_escalation(reason: str | None = None) -> Any:
        """Record a human escalation request for a deterministic test."""
        calls.append({"reason": reason})
        if result is not None:
            return result

        return {
            "escalation_id": "ESC-1234ABCD",
            "status": "queued",
            "reason": reason,
        }

    return record_human_escalation


def test_mock_escalation_tool_returns_queued_confirmation() -> None:
    result = mock_human_escalation_tool.invoke(
        {"reason": " The billing issue is unresolved "}
    )

    assert mock_human_escalation_tool.name == "escalate_to_human"
    assert re.fullmatch(r"ESC-[0-9A-F]{8}", result["escalation_id"])
    assert result == {
        "reason": "The billing issue is unresolved",
        "escalation_id": result["escalation_id"],
        "status": "queued",
    }


def test_escalation_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str | None]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision()),
        escalation_tool=create_recording_escalation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Let me speak to someone about this billing issue"},
        context=context,
    )

    assert tool_calls == [{"reason": "The billing issue is unresolved"}]
    assert result["workflow_stage"] == "human_escalation_queued"
    assert result["escalation_result"] == {
        "reason": "The billing issue is unresolved",
        "escalation_id": "ESC-1234ABCD",
        "status": "queued",
    }
    assert result["final_response"] == (
        "I have queued your request for a human specialist. "
        "Your escalation ID is ESC-1234ABCD."
    )


def test_escalation_without_reason_still_queues_handoff() -> None:
    tool_calls: list[dict[str, str | None]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision(None)),
        escalation_tool=create_recording_escalation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "I want to speak to a person"},
        context=context,
    )

    assert tool_calls == [{"reason": None}]
    assert result["escalation_result"]["reason"] is None
    assert result["escalation_result"]["status"] == "queued"


@pytest.mark.parametrize(
    "message",
    [
        "Cool! I need some help regarding an issue with my grand kid",
        "I am facing issue with my keyboard and mouse setup",
        "I'm unhappy with my keyboard setup",
        "My laptop is not working, can I speak to someone?",
        "I need help with my internet router problem",
    ],
)
def test_off_domain_support_issues_do_not_create_human_escalations(
    message: str,
) -> None:
    tool_calls: list[dict[str, str | None]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision(message)),
        escalation_tool=create_recording_escalation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": message},
        context=context,
    )

    assert tool_calls == []
    assert result["detected_intent"] == "clarification"
    assert result["workflow_stage"] == "planned"
    assert result["query_scope"] == "off_domain"
    assert result.get("escalation_result") is None
    assert result["final_response"] == (
        "I'm here to help with this business profile's services, "
        "policies, hours, pricing, and appointments."
    )


def test_business_related_issue_can_still_escalate_to_human() -> None:
    tool_calls: list[dict[str, str | None]] = []
    message = "I have an issue with my dental cleaning and need a person"
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision(message)),
        escalation_tool=create_recording_escalation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": message},
        context=context,
    )

    assert tool_calls == [{"reason": message}]
    assert result["detected_intent"] == "human_escalation"
    assert result["workflow_stage"] == "human_escalation_queued"
    assert result["query_scope"] == "business_question"


def test_escalation_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision()),
        escalation_tool=create_recording_escalation_tool([]),
    )
    updates = agent_graph.stream(
        {"user_message": "Let me speak to someone"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "escalation",
        "response",
    ]


def test_dissatisfaction_overrides_planner_to_escalation() -> None:
    tool_calls: list[dict[str, str | None]] = []
    planner_decision = PlannerDecision(
        intent="clarification",
        confidence=0.71,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(
        planner=RunnableLambda(lambda _: planner_decision),
        escalation_tool=create_recording_escalation_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "I am not satisfied with your resolution"},
        context=context,
    )

    assert result["detected_intent"] == "human_escalation"
    assert result["workflow_stage"] == "human_escalation_queued"
    assert tool_calls == [{"reason": "I am not satisfied with your resolution"}]


def test_escalation_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision()),
        escalation_tool=create_recording_escalation_tool(
            [],
            {
                "escalation_id": "ESC-1234ABCD",
                "status": "connected",
                "reason": "The billing issue is unresolved",
            },
        ),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {"user_message": "Let me speak to someone"},
            context=context,
        )


def test_escalation_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: escalation_decision()),
    )

    with pytest.raises(RuntimeError, match="escalation tool context is required"):
        agent_graph.invoke(
            {"user_message": "Let me speak to someone"},
            context=context,
        )


def test_planned_intent_router_selects_escalation_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Let me speak to someone",
            "detected_intent": "human_escalation",
        }
    ) == "escalation"
