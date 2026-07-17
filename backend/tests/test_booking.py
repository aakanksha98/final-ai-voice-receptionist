from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool, tool
from pydantic import ValidationError

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.routing import route_planned_intent
from backend.app.business_profiles import get_business_profile
from backend.app.tools.booking import BookingRequest, mock_booking_tool


def booking_decision(
    *,
    service: str | None = "dental cleaning",
    date: str | None = "Monday",
    time: str | None = "2 PM",
) -> PlannerDecision:
    return PlannerDecision(
        intent="book_appointment",
        confidence=0.96,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=service,
            date=date,
            time=time,
            escalation_reason=None,
        ),
    )


def clarification_decision() -> PlannerDecision:
    return PlannerDecision(
        intent="clarification",
        confidence=0.72,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )


def create_recording_booking_tool(
    calls: list[dict[str, str]],
    result: object | None = None,
) -> BaseTool:
    @tool("record_booking", args_schema=BookingRequest)
    def record_booking(
        service: str,
        date: str,
        time: str,
    ) -> Any:
        """Record a booking request for a deterministic test."""
        calls.append({"service": service, "date": date, "time": time})
        if result is not None:
            return result

        return {
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    return record_booking


def test_mock_booking_tool_returns_structured_confirmation() -> None:
    result = mock_booking_tool.invoke(
        {"service": " dental cleaning ", "date": " Monday ", "time": " 2 PM "}
    )

    assert mock_booking_tool.name == "book_appointment"
    assert result == {
        "service": "dental cleaning",
        "date": "Monday",
        "time": "2 PM",
        "status": "confirmed",
    }


def test_booking_intent_invokes_tool_and_persists_result() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning Monday at 2 PM"},
        context=context,
    )

    assert tool_calls == [
        {"service": "dental cleaning", "date": "Monday", "time": "2 PM"}
    ]
    assert result == {
        "user_message": "Book a dental cleaning Monday at 2 PM",
        "conversation_history": [
            {
                "role": "user",
                "content": "Book a dental cleaning Monday at 2 PM",
            },
            {
                "role": "assistant",
                "content": (
                    "Your dental cleaning appointment is booked for Monday at 2 PM."
                ),
            },
        ],
        "normalized_message": "Book a dental cleaning Monday at 2 PM",
        "input_status": "valid",
        "workflow_stage": "appointment_booked",
        "detected_intent": "book_appointment",
        "planner_confidence": 0.96,
        "extracted_slots": {
            "service": "dental cleaning",
            "date": "Monday",
            "time": "2 PM",
        },
        "small_talk_topic": None,
        "missing_booking_slots": [],
        "booking_result": {
            "service": "dental cleaning",
            "date": "Monday",
            "time": "2 PM",
            "status": "confirmed",
        },
        "active_appointment": {
            "service": "dental cleaning",
            "date": "Monday",
            "time": "2 PM",
            "status": "confirmed",
        },
        "final_response": (
            "Your dental cleaning appointment is booked for Monday at 2 PM."
        ),
    }


def test_booking_route_runs_after_planning() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool([]),
    )
    updates = agent_graph.stream(
        {"user_message": "Book a dental cleaning Monday at 2 PM"},
        context=context,
        stream_mode="updates",
    )

    assert [next(iter(update)) for update in updates] == [
        "validate_input",
        "ready_for_planning",
        "planner",
        "booking",
        "response",
    ]


def test_missing_booking_slots_skip_tool_invocation() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision(time=None)),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning tomorrow"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "booking_information_required"
    assert result["missing_booking_slots"] == ["time"]
    assert result["booking_result"] is None
    assert result["final_response"] == (
        "To book the appointment, please provide time."
    )


def test_booking_node_rejects_invalid_tool_result() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
        booking_tool=create_recording_booking_tool([], {"status": "confirmed"}),
    )

    with pytest.raises(ValidationError):
        agent_graph.invoke(
            {"user_message": "Book a dental cleaning Monday at 2 PM"},
            context=context,
        )


def test_complete_booking_requires_tool_context() -> None:
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision()),
    )

    with pytest.raises(RuntimeError, match="Booking tool context is required"):
        agent_graph.invoke(
            {"user_message": "Book a dental cleaning Monday at 2 PM"},
            context=context,
        )


def test_planned_intent_router_selects_booking_route() -> None:
    assert route_planned_intent(
        {
            "user_message": "Book a dental cleaning Monday at 2 PM",
            "detected_intent": "book_appointment",
        }
    ) == "booking"


def test_unsupported_service_skips_booking_tool() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: booking_decision(service="oil change")),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book an oil change tomorrow at 2 PM"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "unsupported_service_requested"
    assert result["unsupported_service"] == "oil change"
    assert result["supported_services"] == [
        "dental cleaning",
        "dental exam",
        "teeth whitening",
        "filling",
        "emergency dental visit",
    ]
    assert result["final_response"] == (
        "I can't book oil change for this demo profile. I can help with "
        "dental cleaning, dental exam, teeth whitening, filling, and "
        "emergency dental visit. Which service would you like?"
    )


def test_auto_repair_battery_diagnostic_booking_uses_tool_not_rag() -> None:
    tool_calls: list[dict[str, str]] = []
    retrieval_queries: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(
                service="battery diagnostics",
                date="Monday",
                time="2 PM",
            )
        ),
        business_profile=get_business_profile("auto_repair"),
        rag_retriever=RunnableLambda(lambda request: retrieval_queries.append(request)),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book battery diagnostics Monday 2 PM"},
        context=context,
    )

    assert retrieval_queries == []
    assert tool_calls == [
        {"service": "battery diagnostic", "date": "Monday", "time": "2 PM"}
    ]
    assert result["workflow_stage"] == "appointment_booked"
    assert result["extracted_slots"] == {
        "service": "battery diagnostic",
        "date": "Monday",
        "time": "2 PM",
    }
    assert result["booking_result"] == {
        "service": "battery diagnostic",
        "date": "Monday",
        "time": "2 PM",
        "status": "confirmed",
    }


@pytest.mark.parametrize(
    "message",
    [
        "Ok aster ! Help me with appointment book for yesterday for my dental cleaning",
        "Ok, help me with tomorrow's for my teeth cleaning",
    ],
)
def test_cross_profile_service_mentions_are_rejected_even_if_planner_misses_service(
    message: str,
) -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(service=None, date="tomorrow", time=None)
        ),
        business_profile=get_business_profile("auto_repair"),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": message},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "unsupported_service_requested"
    assert result["unsupported_service"] == "dental cleaning"
    assert result["supported_services"] == [
        "oil change",
        "brake inspection",
        "tire rotation",
        "battery diagnostic",
        "engine diagnostic",
    ]


def test_supported_service_mention_can_recover_missing_planner_service_slot() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(service=None, date="Monday", time="2 PM")
        ),
        business_profile=get_business_profile("auto_repair"),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book an oil change Monday at 2 PM"},
        context=context,
    )

    assert tool_calls == [
        {"service": "oil change", "date": "Monday", "time": "2 PM"}
    ]
    assert result["workflow_stage"] == "appointment_booked"
    assert result["extracted_slots"] == {
        "service": "oil change",
        "date": "Monday",
        "time": "2 PM",
    }


def test_booking_outside_business_hours_skips_tool_invocation() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(date="Sunday", time="4 PM")
        ),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning Sunday at 4 PM"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["booking_result"] is None
    assert result["schedule_violation"] == (
        "Sunday at 4 PM is outside business hours. Available booking hours are "
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
    )
    assert result["business_hours"] == (
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM"
    )
    assert result["final_response"] == (
        "Sunday at 4 PM is outside business hours. Available booking hours are "
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM. "
        "Please choose another time within business hours."
    )


def test_past_date_and_outside_hours_are_rejected_before_missing_service() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(service=None, date="yesterday", time="7PM")
        ),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Ok, go for 7PM yesterday"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["missing_booking_slots"] == []
    assert result["booking_result"] is None
    assert result["schedule_violation"] == (
        "yesterday is in the past. Please choose a future date. "
        "7PM is outside business hours. Available booking hours are "
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
    )
    assert result["final_response"] == (
        "yesterday is in the past. Please choose a future date. "
        "7PM is outside business hours. Available booking hours are "
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM. "
        "Please choose another time within business hours."
    )


def test_booking_slot_fragment_runs_validation_even_when_planner_clarifies() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: clarification_decision()),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Ok, go for 7PM yesterday"},
        context=context,
    )

    assert tool_calls == []
    assert result["detected_intent"] == "book_appointment"
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["extracted_slots"] == {
        "date": "yesterday",
        "time": "7PM",
    }
    assert result["schedule_violation"] == (
        "yesterday is in the past. Please choose a future date. "
        "7PM is outside business hours. Available booking hours are "
        "Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
    )


def test_booking_follow_up_service_fragment_can_complete_prior_request() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(lambda _: clarification_decision()),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {
            "user_message": "dental cleaning",
            "conversation_history": [
                {"role": "user", "content": "Ok, go for Monday at 2 PM"},
                {
                    "role": "assistant",
                    "content": "I have the date and time, but I need the service.",
                },
            ],
            "workflow_stage": "booking_information_required",
            "detected_intent": "book_appointment",
            "extracted_slots": {"date": "Monday", "time": "2 PM"},
        },
        context=context,
    )

    assert tool_calls == [
        {"service": "dental cleaning", "date": "Monday", "time": "2 PM"}
    ]
    assert result["detected_intent"] == "book_appointment"
    assert result["workflow_stage"] == "appointment_booked"
    assert result["extracted_slots"] == {
        "service": "dental cleaning",
        "date": "Monday",
        "time": "2 PM",
    }


@pytest.mark.parametrize(
    ("date_value", "time_value", "expected_violation"),
    [
        (
            "Monday",
            "7",
            (
                "I could not validate 7 as an appointment time. "
                "Please provide a specific time, such as 2 PM."
            ),
        ),
        (
            "Monday",
            "25:00",
            (
                "I could not validate 25:00 as an appointment time. "
                "Please provide a specific time, such as 2 PM."
            ),
        ),
        (
            "Monday",
            "midnight",
            (
                "Monday at midnight is outside business hours. Available booking "
                "hours are Monday through Friday from 8 AM to 5 PM, and Saturday "
                "from 9 AM to 1 PM."
            ),
        ),
        (
            "last Monday",
            "2 PM",
            (
                "last Monday is in the past. Please choose a future date. "
                "Available booking hours are Monday through Friday from 8 AM "
                "to 5 PM, and Saturday from 9 AM to 1 PM."
            ),
        ),
        (
            "2026-07-16",
            "2 PM",
            (
                "2026-07-16 is in the past. Please choose a future date. "
                "Available booking hours are Monday through Friday from 8 AM "
                "to 5 PM, and Saturday from 9 AM to 1 PM."
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
        (
            "ASAP",
            "2 PM",
            (
                "I could not validate ASAP as a specific appointment date. "
                "Please choose a specific future date. Available booking hours "
                "are Monday through Friday from 8 AM to 5 PM, and Saturday from 9 AM to 1 PM."
            ),
        ),
    ],
)
def test_invalid_booking_dates_and_times_skip_tool_invocation(
    date_value: str,
    time_value: str,
    expected_violation: str,
) -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(date=date_value, time=time_value)
        ),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": f"Book a dental cleaning {date_value} at {time_value}"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["schedule_violation"] == expected_violation
    assert result["booking_result"] is None


def test_noon_booking_is_treated_as_specific_valid_time() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(date="Monday", time="noon")
        ),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning Monday at noon"},
        context=context,
    )

    assert tool_calls == [
        {"service": "dental cleaning", "date": "Monday", "time": "noon"}
    ]
    assert result["workflow_stage"] == "appointment_booked"


def test_booking_outside_business_hours_is_checked_before_missing_service() -> None:
    tool_calls: list[dict[str, str]] = []
    context = AgentContext(
        planner=RunnableLambda(
            lambda _: booking_decision(service=None, date="Sunday", time="4 PM")
        ),
        booking_tool=create_recording_booking_tool(tool_calls),
    )

    result = agent_graph.invoke(
        {"user_message": "I would like to book for Sunday 4 PM"},
        context=context,
    )

    assert tool_calls == []
    assert result["workflow_stage"] == "appointment_outside_business_hours"
    assert result["missing_booking_slots"] == []
    assert result["booking_result"] is None
