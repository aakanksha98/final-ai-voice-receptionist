from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.tools.rescheduling import (
    RescheduleRequest,
    mock_reschedule_tool,
)


def reschedule_decision(
    *,
    date: str | None = "Friday",
    time: str | None = "4 PM",
) -> PlannerDecision:
    return PlannerDecision(
        intent="reschedule_appointment",
        confidence=0.95,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=date,
            time=time,
            escalation_reason=None,
        ),
    )


def create_recording_reschedule_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_reschedule", args_schema=RescheduleRequest)
    def record_reschedule(
        service: str,
        new_date: str,
        new_time: str,
    ) -> Any:
        """Record a reschedule request for a deterministic test."""
        calls.append(
            {
                "service": service,
                "new_date": new_date,
                "new_time": new_time,
            }
        )
        if result is not None:
            return result

        return {
            "service": service,
            "status": "rescheduled",
            "date": new_date,
            "time": new_time,
        }

    return record_reschedule


def test_mock_reschedule_tool_returns_structured_confirmation() -> None:
    result = mock_reschedule_tool.invoke(
        {
            "service": " dental cleaning ",
            "new_date": " Friday ",
            "new_time": " 4 PM ",
        }
    )

    assert mock_reschedule_tool.name == "reschedule_appointment"
    assert result == {
        "service": "dental cleaning",
        "date": "Friday",
        "time": "4 PM",
        "status": "rescheduled",
    }


def test_reschedule_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {
            "user_message": "Move my appointment to Friday at 4 PM",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
    )

    assert tool_calls == [
        {
            "service": "dental cleaning",
            "new_date": "Friday",
            "new_time": "4 PM",
        }
    ]
    assert result["workflow_stage"] == "appointment_rescheduled"
    assert result["missing_reschedule_slots"] == []
    assert result["reschedule_result"] == {
        "service": "dental cleaning",
        "date": "Friday",
        "time": "4 PM",
        "status": "rescheduled",
    }
    assert result["active_appointment"] == {
        "service": "dental cleaning",
        "date": "Friday",
        "time": "4 PM",
        "status": "rescheduled",
    }
    assert result["final_response"] == (
        "Your dental cleaning appointment has been rescheduled to Friday at 4 PM."
    )


def test_reschedule_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool([]),
    )
    updates = agent_graph.stream(
        {
            "user_message": "Move my appointment to Friday at 4 PM",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "reschedule",
        "response",
    ]


def test_missing_reschedule_slots_skip_tool_invocation() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: reschedule_decision(date=None, time=None)
        ),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {
            "user_message": "Reschedule my appointment",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "reschedule_information_required"
    assert result["missing_reschedule_slots"] == ["date", "time"]
    assert result["reschedule_result"] is None
    assert result["final_response"] == (
        "To reschedule the appointment, please provide date and time."
    )


@pytest.mark.parametrize(
    ("date_value", "time_value", "expected_violation"),
    [
        (
            "yesterday",
            "7 PM",
            (
                "yesterday is in the past. Please choose a future date. "
                "7 PM is outside business hours. Available booking hours are "
                "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
            ),
        ),
        (
            "February 30, 2027",
            "2 PM",
            (
                "I could not validate February 30, 2027 as a real appointment "
                "date. Please choose a valid future date."
            ),
        ),
        (
            "Monday",
            "7",
            (
                "I could not validate 7 as an appointment time. "
                "Please provide a specific time, such as 2 PM."
            ),
        ),
        (
            "next available day",
            "19:00",
            (
                "I could not validate next available day as a specific appointment "
                "date. Please choose a specific future date. "
                "19:00 is outside business hours. Available booking hours are "
                "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
            ),
        ),
        (
            "someday",
            "2 PM",
            (
                "I could not validate someday as a specific appointment date. "
                "Please choose a specific future date. Available booking hours "
                "are Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
            ),
        ),
    ],
)
def test_invalid_reschedule_dates_and_times_skip_tool_invocation(
    date_value: str,
    time_value: str,
    expected_violation: str,
) -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: reschedule_decision(date=date_value, time=time_value)
        ),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {
            "user_message": f"Move my appointment to {date_value} at {time_value}",
            "active_appointment": {
                "service": "dental cleaning",
                "date": "tomorrow",
                "time": "2 PM",
                "status": "confirmed",
            },
        },
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["missing_reschedule_slots"] == []
    assert result["reschedule_result"] is None
    assert result["schedule_violation"] == expected_violation


def test_reschedule_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool(
            [],
            {
                "service": "dental cleaning",
                "date": "Friday",
                "time": "4 PM",
                "status": "confirmed",
            },
        ),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {
                "user_message": "Move my appointment to Friday at 4 PM",
                "active_appointment": {
                    "service": "dental cleaning",
                    "date": "tomorrow",
                    "time": "2 PM",
                    "status": "confirmed",
                },
            },
            context=context,
        )


def test_complete_reschedule_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
    )

    with pytest.raises(RuntimeError, match="Reschedule tool context is required"):
        agent_graph.invoke(
            {
                "user_message": "Move my appointment to Friday at 4 PM",
                "active_appointment": {
                    "service": "dental cleaning",
                    "date": "tomorrow",
                    "time": "2 PM",
                    "status": "confirmed",
                },
            },
            context=context,
        )


def test_planned_intent_router_selects_reschedule_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Move my appointment to Friday at 4 PM",
            "detected_intent": "reschedule_appointment",
        }
    ) == "reschedule"


def test_reschedule_without_active_appointment_skips_tool() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: reschedule_decision()),
        reschedule_tool=create_recording_reschedule_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Move my appointment to Friday at 4 PM"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "no_active_appointment"
    assert result["reschedule_result"] is None
    assert result["final_response"] == (
        "I don't have an active appointment in this conversation to reschedule yet."
    )
