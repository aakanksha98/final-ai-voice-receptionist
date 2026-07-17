import json
from datetime import date

from langchain_core.runnables import RunnableLambda

from backend.app.agent.context import AgentContext
from backend.app.agent.graph import agent_graph
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.tools.booking import mock_booking_tool


def test_response_generator_receives_missing_booking_slot_context() -> None:
    response_inputs: list[dict[str, str]] = []

    decision = PlannerDecision(
        intent="book_appointment",
        confidence=0.96,
        small_talk_topic=None,
        slots=PlannerSlots(
            service="dental cleaning",
            date="Monday",
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(
        planner=RunnableLambda(lambda _: decision),
        response_generator=RunnableLambda(
            lambda response_input: (
                response_inputs.append(response_input)
                or "I'd be happy to help book your dental cleaning appointment. What time works best Monday?"
            )
        ),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning Monday"},
        context=context,
    )

    assert result["workflow_stage"] == "booking_information_required"
    assert result["missing_booking_slots"] == ["time"]
    assert result["final_response"] == (
        "I'd be happy to help book your dental cleaning appointment. What time works best Monday?"
    )
    response_context = json.loads(response_inputs[0]["response_context"])
    assert response_context["intent"] == "book_appointment"
    assert response_context["workflow_stage"] == "booking_information_required"
    assert response_context["response_goal"] == (
        "Ask only for the missing appointment booking details."
    )
    assert response_context["facts"]["extracted_slots"] == {
        "service": "dental cleaning",
        "date": "Monday",
    }
    assert response_context["facts"]["missing_fields"] == ["time"]


def test_response_generator_receives_tool_result_context() -> None:
    response_inputs: list[dict[str, str]] = []

    decision = PlannerDecision(
        intent="book_appointment",
        confidence=0.97,
        small_talk_topic=None,
        slots=PlannerSlots(
            service="dental cleaning",
            date="Monday",
            time="4 PM",
            escalation_reason=None,
        ),
    )
    context = AgentContext(
        planner=RunnableLambda(lambda _: decision),
        booking_tool=mock_booking_tool,
        response_generator=RunnableLambda(
            lambda response_input: (
                response_inputs.append(response_input)
                or "Your dental cleaning appointment is booked for Monday at 4 PM. Let me know if you need anything else."
            )
        ),
    )

    result = agent_graph.invoke(
        {"user_message": "Book a dental cleaning Monday at 4 PM"},
        context=context,
    )

    assert result["final_response"] == (
        "Your dental cleaning appointment is booked for Monday at 4 PM. Let me know if you need anything else."
    )
    response_context = json.loads(response_inputs[0]["response_context"])
    assert response_context["workflow_stage"] == "appointment_booked"
    assert response_context["facts"]["booking_result"] == {
        "service": "dental cleaning",
        "date": "Monday",
        "time": "4 PM",
        "status": "confirmed",
    }
    assert response_context["facts"]["active_appointment"] == {
        "service": "dental cleaning",
        "date": "Monday",
        "time": "4 PM",
        "status": "confirmed",
    }


def test_response_generator_receives_off_domain_scope_for_general_questions() -> None:
    response_inputs: list[dict[str, str]] = []

    decision = PlannerDecision(
        intent="clarification",
        confidence=0.7,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(
        planner=RunnableLambda(lambda _: decision),
        response_generator=RunnableLambda(
            lambda response_input: (
                response_inputs.append(response_input)
                or "I can help with this business profile's services and appointments."
            )
        ),
    )

    agent_graph.invoke(
        {"user_message": "What does a giraffe say?"},
        context=context,
    )

    response_context = json.loads(response_inputs[0]["response_context"])
    assert response_context["workflow_stage"] == "planned"
    assert response_context["facts"]["query_scope"] == "off_domain"


def test_response_generator_receives_cross_business_scope() -> None:
    response_inputs: list[dict[str, str]] = []

    decision = PlannerDecision(
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
    context = AgentContext(
        planner=RunnableLambda(lambda _: decision),
        response_generator=RunnableLambda(
            lambda response_input: (
                response_inputs.append(response_input)
                or "This session is set to Dental Clinic. I can help with dental services."
            )
        ),
    )

    agent_graph.invoke(
        {"user_message": "Salon services"},
        context=context,
    )

    response_context = json.loads(response_inputs[0]["response_context"])
    assert response_context["business"]["label"] == "Dental Clinic"
    assert response_context["facts"]["query_scope"] == "cross_business_profile"


def test_current_date_question_uses_scheduling_reference_scope() -> None:
    decision = PlannerDecision(
        intent="clarification",
        confidence=0.78,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(planner=RunnableLambda(lambda _: decision))

    result = agent_graph.invoke(
        {"user_message": "What is the date?"},
        context=context,
    )

    today = date.today()
    current_date = f"{today:%A}, {today:%B} {today.day}, {today:%Y}"
    assert result["workflow_stage"] == "planned"
    assert result["query_scope"] == "scheduling_reference"
    assert result["final_response"] == (
        f"Today is {current_date}. I can use that to help schedule an appointment."
    )


def test_current_date_question_does_not_continue_unfinished_booking_flow() -> None:
    decision = PlannerDecision(
        intent="clarification",
        confidence=0.78,
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )
    context = AgentContext(planner=RunnableLambda(lambda _: decision))

    result = agent_graph.invoke(
        {
            "user_message": "What is the date?",
            "conversation_history": [
                {"role": "user", "content": "Book a dental cleaning"},
                {
                    "role": "assistant",
                    "content": "To book the appointment, please provide date and time.",
                },
            ],
            "workflow_stage": "booking_information_required",
            "detected_intent": "book_appointment",
            "extracted_slots": {"service": "dental cleaning"},
        },
        context=context,
    )

    today = date.today()
    current_date = f"{today:%A}, {today:%B} {today.day}, {today:%Y}"
    assert result["detected_intent"] == "clarification"
    assert result["workflow_stage"] == "planned"
    assert result["query_scope"] == "scheduling_reference"
    assert result["final_response"] == (
        f"Today is {current_date}. I can use that to help schedule an appointment."
    )
