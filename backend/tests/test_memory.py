import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import build_memory_agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.business_profiles import get_business_profile
from backend.app.tools.booking import BookingRequest


def booking_decision(
    *,
    service: str | None = None,
    date: str | None = None,
    time: str | None = None,
) -> PlannerDecision:
    return PlannerDecision(
        intent="book_appointment",
        confidence=0.95,
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
        confidence=0.82,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )


def test_memory_carries_slots_across_an_incomplete_booking() -> None:
    planner_inputs: list[dict[str, str]] = []
    tool_calls: list[dict[str, str]] = []

    def plan(planner_input: dict[str, str]) -> PlannerDecision:
        planner_inputs.append(planner_input)
        if planner_input["user_message"] == "Book a haircut tomorrow":
            return booking_decision(service="haircut", date="tomorrow")
        return booking_decision(time="2 PM")

    @tool("record_memory_booking", args_schema=BookingRequest)
    def record_memory_booking(
        service: str,
        date: str,
        time: str,
    ) -> dict[str, str]:
        """Record a booking assembled across conversation turns."""
        tool_calls.append({"service": service, "date": date, "time": time})
        return {
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    graph = build_memory_agent_graph()
    context = AgentContext(
        planner=RunnableLambda(plan),
        business_profile=get_business_profile("salon"),
        booking_tool=record_memory_booking,
    )
    config = {"configurable": {"thread_id": "booking-thread"}}

    first_result = graph.invoke(
        {"user_message": "Book a haircut tomorrow"},
        context=context,
        config=config,
    )
    second_result = graph.invoke(
        {"user_message": "At 2 PM"},
        context=context,
        config=config,
    )

    assert first_result["workflow_stage"] == "booking_information_required"
    assert first_result["missing_booking_slots"] == ["time"]
    assert planner_inputs == [
        {
            "user_message": "Book a haircut tomorrow",
            "conversation_history": "No prior conversation.",
            "business_type": "Salon",
            "supported_services": "haircut, blowout, hair color, manicure, facial",
        },
        {
            "user_message": "At 2 PM",
            "conversation_history": (
                "user: Book a haircut tomorrow\n"
                "assistant: To book the appointment, please provide time."
            ),
            "business_type": "Salon",
            "supported_services": "haircut, blowout, hair color, manicure, facial",
        },
    ]
    assert tool_calls == [
        {"service": "haircut", "date": "tomorrow", "time": "2 PM"}
    ]
    assert second_result["workflow_stage"] == "appointment_booked"
    assert second_result["conversation_history"] == [
        {"role": "user", "content": "Book a haircut tomorrow"},
        {
            "role": "assistant",
            "content": "To book the appointment, please provide time.",
        },
        {"role": "user", "content": "At 2 PM"},
        {
            "role": "assistant",
            "content": (
                "Your haircut appointment is booked for tomorrow at 2 PM."
            ),
        },
    ]


def test_pending_booking_context_survives_clarification_follow_ups() -> None:
    def plan(planner_input: dict[str, str]) -> PlannerDecision:
        message = planner_input["user_message"]
        if message == "I want to book Airbnb":
            return booking_decision(service="Airbnb")
        if message == "Help me book a flight":
            return booking_decision(service="flight")
        if message == "Book appointment for teeth cleaning":
            return booking_decision(service="teeth cleaning")
        return clarification_decision()

    graph = build_memory_agent_graph()
    context = AgentContext(
        planner=RunnableLambda(plan),
        business_profile=get_business_profile("dental"),
    )
    config = {"configurable": {"thread_id": "pending-booking-context"}}

    graph.invoke(
        {"user_message": "I want to book Airbnb"},
        context=context,
        config=config,
    )
    graph.invoke(
        {"user_message": "Help me book a flight"},
        context=context,
        config=config,
    )
    service_result = graph.invoke(
        {"user_message": "Book appointment for teeth cleaning"},
        context=context,
        config=config,
    )
    suggestion_result = graph.invoke(
        {"user_message": "what do you suggest"},
        context=context,
        config=config,
    )
    recall_result = graph.invoke(
        {"user_message": "What service was I looking for?"},
        context=context,
        config=config,
    )

    assert service_result["workflow_stage"] == "booking_information_required"
    assert service_result["missing_booking_slots"] == ["date", "time"]
    assert suggestion_result["detected_intent"] == "book_appointment"
    assert suggestion_result["workflow_stage"] == "booking_information_required"
    assert suggestion_result["missing_booking_slots"] == ["date", "time"]
    assert suggestion_result["extracted_slots"] == {"service": "teeth cleaning"}
    assert suggestion_result["final_response"] == (
        "To book the appointment, please provide date and time."
    )
    assert recall_result["detected_intent"] == "clarification"
    assert recall_result["workflow_stage"] == "planned"
    assert recall_result["extracted_slots"] == {"service": "teeth cleaning"}
    assert recall_result["final_response"] == (
        "You were looking to book teeth cleaning."
    )


def test_unsupported_booking_request_does_not_become_pending_context() -> None:
    decisions = iter(
        [
            booking_decision(service="flight"),
            clarification_decision(),
        ]
    )
    graph = build_memory_agent_graph()
    context = AgentContext(
        planner=RunnableLambda(lambda _: next(decisions)),
        business_profile=get_business_profile("dental"),
    )
    config = {"configurable": {"thread_id": "unsupported-context"}}

    graph.invoke(
        {"user_message": "Help me book a flight"},
        context=context,
        config=config,
    )
    result = graph.invoke(
        {"user_message": "What service was I looking for?"},
        context=context,
        config=config,
    )

    assert result["detected_intent"] == "clarification"
    assert result["workflow_stage"] == "planned"
    assert result["extracted_slots"] == {}
    assert result["final_response"] == (
        "Could you clarify whether you need business information, "
        "appointment help, or a human specialist?"
    )


def test_slot_fragment_after_context_recall_uses_remembered_service() -> None:
    decisions = iter(
        [
            booking_decision(service="teeth cleaning"),
            clarification_decision(),
            clarification_decision(),
        ]
    )
    tool_calls: list[dict[str, str]] = []

    @tool("record_context_recall_booking", args_schema=BookingRequest)
    def record_context_recall_booking(
        service: str,
        date: str,
        time: str,
    ) -> dict[str, str]:
        """Record a booking after a context recall turn."""
        tool_calls.append({"service": service, "date": date, "time": time})
        return {
            "status": "confirmed",
            "service": service,
            "date": date,
            "time": time,
        }

    graph = build_memory_agent_graph()
    context = AgentContext(
        planner=RunnableLambda(lambda _: next(decisions)),
        business_profile=get_business_profile("dental"),
        booking_tool=record_context_recall_booking,
    )
    config = {"configurable": {"thread_id": "context-recall-slot-fragment"}}

    graph.invoke(
        {"user_message": "Book appointment for teeth cleaning"},
        context=context,
        config=config,
    )
    graph.invoke(
        {"user_message": "What service was I looking for?"},
        context=context,
        config=config,
    )
    result = graph.invoke(
        {"user_message": "Monday at 2 PM"},
        context=context,
        config=config,
    )

    assert result["workflow_stage"] == "appointment_booked"
    assert result["extracted_slots"] == {
        "service": "dental cleaning",
        "date": "Monday",
        "time": "2 PM",
    }
    assert tool_calls == [
        {"service": "dental cleaning", "date": "Monday", "time": "2 PM"}
    ]


def test_memory_isolates_conversation_threads() -> None:
    graph = build_memory_agent_graph()
    decision = PlannerDecision(
        intent="clarification",
        confidence=0.8,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(planner=RunnableLambda(lambda _: decision))

    graph.invoke(
        {"user_message": "First thread message"},
        context=context,
        config={"configurable": {"thread_id": "thread-a"}},
    )
    second_thread = graph.invoke(
        {"user_message": "Second thread message"},
        context=context,
        config={"configurable": {"thread_id": "thread-b"}},
    )

    assert second_thread["conversation_history"] == [
        {"role": "user", "content": "Second thread message"},
        {
            "role": "assistant",
            "content": (
                "Could you clarify whether you need business information, "
                "appointment help, or a human specialist?"
            ),
        },
    ]


def test_memory_graph_requires_thread_identifier() -> None:
    graph = build_memory_agent_graph()

    with pytest.raises(ValueError, match="thread_id"):
        graph.invoke({"user_message": "Hello"})


def test_new_turn_clears_stale_terminal_outputs() -> None:
    decisions = iter(
        [
            PlannerDecision(
                intent="small_talk",
                confidence=0.98,
                small_talk_topic="greeting",
                slots=PlannerSlots(
                    service=None,
                    date=None,
                    time=None,
                    escalation_reason=None,
                ),
            ),
            PlannerDecision(
                intent="clarification",
                confidence=0.7,
                small_talk_topic=None,
                slots=PlannerSlots(
                    service=None,
                    date=None,
                    time=None,
                    escalation_reason=None,
                ),
            ),
        ]
    )
    graph = build_memory_agent_graph()
    context = AgentContext(planner=RunnableLambda(lambda _: next(decisions)))
    config = {"configurable": {"thread_id": "reset-thread"}}

    first_result = graph.invoke(
        {"user_message": "Hello"},
        context=context,
        config=config,
    )
    second_result = graph.invoke(
        {"user_message": "Something unclear"},
        context=context,
        config=config,
    )

    assert first_result["small_talk_topic"] == "greeting"
    assert second_result["workflow_stage"] == "planned"
    assert second_result["small_talk_topic"] is None
    assert second_result["retrieved_documents"] == []
    assert second_result["escalation_result"] is None
