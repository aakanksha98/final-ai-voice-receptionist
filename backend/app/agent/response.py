from __future__ import annotations

import json
import os
from datetime import date
from typing import TYPE_CHECKING, Any, Final

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime

from backend.app.agent.state import AgentState, AgentStateUpdate, SmallTalkTopic
from backend.app.business_profiles import supported_service_names


if TYPE_CHECKING:
    from backend.app.agent.context import AgentContext


DEFAULT_RESPONSE_MODEL = "gpt-5.4-mini"
ResponseRunnable = Runnable[dict[str, str], str]


SMALL_TALK_RESPONSES: Final[dict[SmallTalkTopic, str]] = {
    "greeting": (
        "Hi! I'm Aster, the AI receptionist for this business. "
        "How can I help today?"
    ),
    "assistant_identity": "I'm Aster, an AI reception assistant.",
    "capabilities": (
        "I can help with business questions, appointment booking, cancellation, "
        "rescheduling, or connecting you with a person."
    ),
    "courtesy": "You're welcome. Is there anything else I can help with?",
}


CONVERSATIONAL_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are Aster, a warm and professional AI receptionist.

The system has already handled intent classification, routing, business logic,
retrieval, state updates, and tool execution. Your only job is to turn the
provided structured response context into a natural user-facing reply.

Rules:
- Do not decide business logic, tool usage, appointment status, service
  eligibility, prices, policies, discounts, staff details, or availability.
- Use only facts present in the response context.
- Do not invent appointments, services, prices, policies, promotions, staff
  details, availability, or contact details.
- The business name is personalization only. Do not imply prices, policies,
  services, hours, or retrieved facts are specific to the custom business name.
  For factual business answers, state the fact directly without naming the
  custom business.
- For knowledge retrieval replies, retrieved_documents are the only evidence.
  Answer only when the retrieved documents directly support the user's exact
  question. If they do not, say the available business information does not
  cover that and offer appointment or service help.
- If facts.query_scope is off_domain, do not answer from general model
  knowledge. Explain that you can help with this business profile's services,
  policies, hours, pricing, and appointments.
- If facts.query_scope is cross_business_profile, do not answer for the other
  business type. Say this session is set to the selected business profile,
  list the supported services from context, and offer to help with those.
- If facts.query_scope is staff_personal_info, say you do not have information
  about individual staff members and offer business or appointment help.
- If facts.query_scope is sensitive_request, refuse briefly and redirect to
  appointments or business questions.
- If facts.query_scope is scheduling_reference, answer using facts.current_date
  and offer to use that for appointment scheduling.
- If information is unavailable, say so naturally and offer the next helpful
  step when appropriate.
- If fields are missing, ask only for the missing information.
- If a tool result is present, describe that result conversationally.
- Keep the reply concise, friendly, and suitable for voice.
- Do not use markdown, bullet points, JSON, or internal workflow labels.""",
        ),
        (
            "human",
            """Structured response context:
{response_context}

Write Aster's reply now.""",
        ),
    ]
)


def create_openai_response_generator(
    model: str | None = None,
) -> ResponseRunnable:
    selected_model = model or os.getenv("OPENAI_MODEL", DEFAULT_RESPONSE_MODEL)
    chat_model = ChatOpenAI(model=selected_model, max_retries=2)
    return CONVERSATIONAL_RESPONSE_PROMPT | chat_model | StrOutputParser()


def generate_response(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    response_context = _build_response_context(state, runtime)
    response_text = _generate_conversational_response(response_context, runtime).strip()
    if not response_text:
        raise ValueError("Response generation returned empty text")

    update: AgentStateUpdate = {"final_response": response_text}
    if state.get("input_status") == "valid":
        update["conversation_history"] = [
            {"role": "assistant", "content": response_text}
        ]
    return update


def _generate_conversational_response(
    response_context: dict[str, Any],
    runtime: Runtime[AgentContext],
) -> str:
    deterministic_response = _deterministic_context_response(response_context)
    if deterministic_response is not None:
        return deterministic_response

    if runtime.context is None or runtime.context.response_generator is None:
        return _fallback_response(response_context)

    return runtime.context.response_generator.invoke(
        {
            "response_context": json.dumps(
                response_context,
                ensure_ascii=False,
                sort_keys=True,
            )
        }
    )


def _deterministic_context_response(response_context: dict[str, Any]) -> str | None:
    if (
        response_context.get("workflow_stage") != "planned"
        or response_context.get("intent") != "clarification"
    ):
        return None

    user_message = str(response_context.get("user_message") or "")
    facts = response_context.get("facts", {})
    extracted_slots = facts.get("extracted_slots", {})
    service = extracted_slots.get("service") if isinstance(extracted_slots, dict) else None
    if not isinstance(service, str) or not service.strip():
        return None

    if _is_pending_booking_service_recall_question(user_message):
        return f"You were looking to book {service.strip()}."

    return None


def _build_response_context(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, Any]:
    agent_context = runtime.context
    business_profile = agent_context.business_profile if agent_context else None
    workflow_stage = state.get("workflow_stage")
    provided_business_name = (
        agent_context.business_name
        if agent_context and agent_context.business_name
        else None
    )
    business_name = provided_business_name or "this business"
    if (
        provided_business_name
        and (
            workflow_stage == "knowledge_retrieved"
            or state.get("query_scope") is not None
        )
    ):
        business_name = "this business"

    business = {
        "name": business_name,
        "type": business_profile.business_type if business_profile else None,
        "label": business_profile.label if business_profile else None,
        "supported_services": (
            supported_service_names(business_profile) if business_profile else []
        ),
    }

    return {
        "assistant": {
            "name": "Aster",
            "role": "AI receptionist",
        },
        "business": business,
        "user_message": state.get("normalized_message") or state.get("user_message", ""),
        "intent": state.get("detected_intent"),
        "workflow_stage": workflow_stage,
        "response_goal": _response_goal(state),
        "facts": {
            "query_scope": state.get("query_scope"),
            "retrieval_query": state.get("retrieval_query"),
            "small_talk_topic": state.get("small_talk_topic"),
            "extracted_slots": dict(state.get("extracted_slots", {})),
            "missing_fields": _missing_fields(state),
            "retrieved_documents": state.get("retrieved_documents", []),
            "booking_result": state.get("booking_result"),
            "schedule_violation": state.get("schedule_violation"),
            "business_hours": state.get("business_hours"),
            "cancellation_result": state.get("cancellation_result"),
            "reschedule_result": state.get("reschedule_result"),
            "escalation_result": state.get("escalation_result"),
            "active_appointment": state.get("active_appointment"),
            "unsupported_service": state.get("unsupported_service"),
            "supported_services": state.get("supported_services", []),
            "validation_error": state.get("validation_error"),
            "current_date": _current_date_label(),
        },
        "guardrails": {
            "business_logic_already_decided": True,
            "business_name_is_personalization_only": True,
            "do_not_execute_tools": True,
            "do_not_invent_business_facts": True,
            "do_not_override_workflow_stage": True,
        },
    }


def _response_goal(state: AgentState) -> str:
    workflow_stage = state.get("workflow_stage")
    intent = state.get("detected_intent")

    if workflow_stage == "rejected":
        return "Ask the user to enter a request so the receptionist can help."

    if workflow_stage == "knowledge_retrieved":
        if state.get("retrieved_documents"):
            return "Answer the business question using only the retrieved evidence."
        return (
            "Explain that the available business information does not answer the "
            "question and offer a helpful next step."
        )

    if workflow_stage == "planned":
        if intent == "small_talk":
            return "Respond naturally to the small-talk topic."
        return (
            "Ask a concise, natural clarification question related to the user's "
            "request and the receptionist domain."
        )

    goals = {
        "booking_information_required": (
            "Ask only for the missing appointment booking details."
        ),
        "appointment_booked": (
            "Confirm the appointment was booked using the booking result."
        ),
        "appointment_already_active": (
            "Explain that one active appointment already exists and offer to help "
            "cancel or reschedule it."
        ),
        "appointment_outside_business_hours": (
            "Explain that the requested appointment time is outside business "
            "hours, share the available business hours, and ask for another time."
        ),
        "unsupported_service_requested": (
            "Explain that the requested service is not supported for this demo "
            "profile and offer the supported services."
        ),
        "cancellation_information_required": (
            "Explain that an active appointment is needed before cancellation."
        ),
        "appointment_cancelled": (
            "Confirm the appointment was cancelled using the cancellation result."
        ),
        "reschedule_information_required": (
            "Ask only for the missing reschedule details."
        ),
        "appointment_rescheduled": (
            "Confirm the appointment was rescheduled using the reschedule result."
        ),
        "no_active_appointment": (
            "Explain that there is no active appointment in this conversation for "
            "the requested action."
        ),
        "human_escalation_queued": (
            "Acknowledge the concern or request and explain that a human support "
            "request was created."
        ),
    }
    return goals.get(
        workflow_stage or "",
        "Write a concise receptionist response based only on the provided facts.",
    )


def _missing_fields(state: AgentState) -> list[str]:
    for field_name in (
        "missing_booking_slots",
        "missing_cancellation_slots",
        "missing_reschedule_slots",
    ):
        fields = state.get(field_name)
        if fields:
            return list(fields)
    return []


def _fallback_response(response_context: dict[str, Any]) -> str:
    stage = response_context.get("workflow_stage")
    facts = response_context.get("facts", {})

    if stage == "rejected":
        return "Please say or enter a request so I can help."

    if stage == "knowledge_retrieved":
        documents = facts.get("retrieved_documents") or []
        if not documents:
            return "I could not find enough business information to answer that."
        return str(documents[0].get("content", "")).strip()

    if stage == "planned":
        intent = response_context.get("intent")
        if intent == "small_talk":
            return _small_talk_fallback(response_context)
        query_scope = facts.get("query_scope")
        if query_scope == "off_domain":
            return (
                "I'm here to help with this business profile's services, "
                "policies, hours, pricing, and appointments."
            )
        if query_scope == "cross_business_profile":
            return _cross_business_profile_fallback(response_context)
        if query_scope == "staff_personal_info":
            return (
                "I don't have information about individual staff members, but I "
                "can help with appointments or business questions."
            )
        if query_scope == "sensitive_request":
            return (
                "I can't help with credentials or internal system information. "
                "I can help with appointments or business questions."
            )
        if query_scope == "scheduling_reference":
            current_date = facts.get("current_date")
            return (
                f"Today is {current_date}. I can use that to help schedule an "
                "appointment."
            )
        return (
            "Could you clarify whether you need business information, "
            "appointment help, or a human specialist?"
        )

    fallback_builders = {
        "booking_information_required": _booking_information_fallback,
        "appointment_booked": _booking_confirmation_fallback,
        "appointment_already_active": _appointment_already_active_fallback,
        "appointment_outside_business_hours": _outside_business_hours_fallback,
        "unsupported_service_requested": _unsupported_service_fallback,
        "cancellation_information_required": (
            lambda _: "I need an active appointment in this conversation before I can cancel it."
        ),
        "appointment_cancelled": _cancellation_confirmation_fallback,
        "reschedule_information_required": _reschedule_information_fallback,
        "appointment_rescheduled": _reschedule_confirmation_fallback,
        "no_active_appointment": _no_active_appointment_fallback,
        "human_escalation_queued": _escalation_fallback,
    }
    builder = fallback_builders.get(str(stage))
    if builder is None:
        raise ValueError(f"No response strategy for workflow stage: {stage}")
    return builder(response_context)


def _small_talk_fallback(response_context: dict[str, Any]) -> str:
    facts = response_context.get("facts", {})
    topic = facts.get("small_talk_topic")
    if topic == "greeting":
        business_name = response_context.get("business", {}).get("name") or "this business"
        return (
            "Hi! I'm Aster, the AI receptionist for "
            f"{business_name}. How can I help today?"
        )
    return SMALL_TALK_RESPONSES[str(topic)]


def _booking_information_fallback(response_context: dict[str, Any]) -> str:
    facts = response_context.get("facts", {})
    missing_fields = _join_fields(facts.get("missing_fields", []))
    return f"To book the appointment, please provide {missing_fields}."


def _booking_confirmation_fallback(response_context: dict[str, Any]) -> str:
    result = response_context.get("facts", {}).get("booking_result")
    if result is None:
        raise ValueError("Booking result is required")
    return (
        f"Your {result['service']} appointment is booked for {result['date']} "
        f"at {result['time']}."
    )


def _appointment_already_active_fallback(response_context: dict[str, Any]) -> str:
    appointment = response_context.get("facts", {}).get("active_appointment")
    if appointment is None:
        raise ValueError("Active appointment is required")
    return (
        f"You already have an active {appointment['service']} appointment for "
        f"{appointment['date']} at {appointment['time']}. In this V1 demo, "
        "please cancel or reschedule it before booking another appointment."
    )


def _unsupported_service_fallback(response_context: dict[str, Any]) -> str:
    facts = response_context.get("facts", {})
    requested_service = facts.get("unsupported_service") or "that service"
    services = facts.get("supported_services", [])
    if not services:
        raise ValueError("Supported services are required")

    return (
        f"I can't book {requested_service} for this demo profile. "
        f"I can help with {_join_fields(services)}. Which service would you like?"
    )


def _cross_business_profile_fallback(response_context: dict[str, Any]) -> str:
    business = response_context.get("business", {})
    facts = response_context.get("facts", {})
    label = business.get("label") or "selected business profile"
    services = business.get("supported_services") or facts.get("supported_services", [])
    if not services:
        return f"This session is set to {label}. I can help with that profile."

    return (
        f"This session is set to {label}. I can help with "
        f"{_join_fields(services)}."
    )


def _outside_business_hours_fallback(response_context: dict[str, Any]) -> str:
    facts = response_context.get("facts", {})
    violation = facts.get("schedule_violation")
    business_hours = facts.get("business_hours")
    if violation:
        return (
            f"{violation} Please choose another time within business hours."
        )
    if business_hours:
        return (
            f"That appointment time is outside business hours. Available hours "
            f"are {business_hours}. Please choose another time."
        )
    return "That appointment time is outside business hours. Please choose another time."


def _cancellation_confirmation_fallback(response_context: dict[str, Any]) -> str:
    result = response_context.get("facts", {}).get("cancellation_result")
    if result is None:
        raise ValueError("Cancellation result is required")
    return (
        f"Done. I've cancelled your {result['service']} appointment for "
        f"{result['date']} at {result['time']}."
    )


def _reschedule_information_fallback(response_context: dict[str, Any]) -> str:
    facts = response_context.get("facts", {})
    missing_fields = _join_fields(facts.get("missing_fields", []))
    return f"To reschedule the appointment, please provide {missing_fields}."


def _reschedule_confirmation_fallback(response_context: dict[str, Any]) -> str:
    result = response_context.get("facts", {}).get("reschedule_result")
    if result is None:
        raise ValueError("Reschedule result is required")
    return (
        f"Your {result['service']} appointment has been rescheduled to "
        f"{result['date']} at {result['time']}."
    )


def _no_active_appointment_fallback(response_context: dict[str, Any]) -> str:
    intent = response_context.get("intent")
    action = "reschedule" if intent == "reschedule_appointment" else "cancel"
    return (
        "I don't have an active appointment in this conversation to "
        f"{action} yet."
    )


def _escalation_fallback(response_context: dict[str, Any]) -> str:
    result = response_context.get("facts", {}).get("escalation_result")
    if result is None:
        raise ValueError("Escalation result is required")
    return (
        "I have queued your request for a human specialist. "
        f"Your escalation ID is {result['escalation_id']}."
    )


def _join_fields(fields: list[str]) -> str:
    labels = {
        "service": "service",
        "date": "date",
        "time": "time",
    }
    readable_fields = [labels.get(field, field) for field in fields]
    if not readable_fields:
        raise ValueError("At least one missing field is required")
    if len(readable_fields) == 1:
        return readable_fields[0]
    if len(readable_fields) == 2:
        return " and ".join(readable_fields)
    return ", ".join(readable_fields[:-1]) + f", and {readable_fields[-1]}"


def _current_date_label() -> str:
    today = date.today()
    return f"{today:%A}, {today:%B} {today.day}, {today:%Y}"


def _is_pending_booking_service_recall_question(message: str) -> bool:
    normalized = " ".join(message.lower().split()).rstrip("?")
    recall_phrases = (
        "what service",
        "which service",
        "what appointment",
        "which appointment",
        "what was i booking",
        "what am i booking",
        "what were we booking",
        "what are we booking",
        "what was i looking for",
        "what am i looking for",
        "what were we looking for",
        "what did i ask",
    )
    return any(phrase in normalized for phrase in recall_phrases)
