from re import IGNORECASE, search
from typing import cast

from langchain_core.documents import Document
from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.planner import PlannerDecision, PlannerSlots
from backend.app.agent.scope import determine_query_scope
from backend.app.agent.state import (
    AgentState,
    AgentStateUpdate,
    ExtractedSlots,
    InputStatus,
    PlannerIntent,
    QueryScope,
    RetrievedDocument,
    WorkflowStage,
)
from backend.app.business_profiles import (
    BusinessProfile,
    match_supported_service,
    supported_service_names,
)


FOLLOW_UP_STAGES: dict[PlannerIntent, WorkflowStage] = {
    "book_appointment": "booking_information_required",
    "reschedule_appointment": "reschedule_information_required",
}


def validate_input(state: AgentState) -> AgentStateUpdate:
    normalized_message = state["user_message"].strip()
    input_status: InputStatus = "valid" if normalized_message else "invalid"

    return {
        "normalized_message": normalized_message,
        "input_status": input_status,
    }


def mark_ready_for_planning(state: AgentState) -> AgentStateUpdate:
    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before planning")

    update: AgentStateUpdate = {
        "workflow_stage": "ready_for_planning",
        "conversation_history": [
            {"role": "user", "content": normalized_message}
        ],
    }
    if state.get("conversation_history"):
        previous_stage = state.get("workflow_stage")
        if previous_stage == "rejected":
            previous_stage = state.get("previous_workflow_stage")

        update.update(_reset_turn_outputs())
        update["previous_workflow_stage"] = previous_stage

    return update


def reject_invalid_input(state: AgentState) -> AgentStateUpdate:
    update: AgentStateUpdate = {}
    if state.get("conversation_history"):
        update.update(_reset_turn_outputs())
        update["previous_workflow_stage"] = state.get("workflow_stage")

    update.update(
        {
            "workflow_stage": "rejected",
            "validation_error": "user_message must not be empty",
        }
    )
    return update


def _reset_turn_outputs() -> AgentStateUpdate:
    return {
        "validation_error": None,
        "final_response": None,
        "query_scope": None,
        "retrieval_query": None,
        "retrieved_documents": [],
        "missing_booking_slots": [],
        "booking_result": None,
        "unsupported_service": None,
        "supported_services": [],
        "schedule_violation": None,
        "business_hours": None,
        "missing_cancellation_slots": [],
        "cancellation_result": None,
        "missing_reschedule_slots": [],
        "reschedule_result": None,
        "escalation_result": None,
    }


def plan_request(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Planner context is required for valid input")

    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before planning")

    decision = runtime.context.planner.invoke(
        {
            "user_message": normalized_message,
            "conversation_history": _format_prior_conversation(state),
            "business_type": runtime.context.business_profile.label,
            "supported_services": ", ".join(
                supported_service_names(runtime.context.business_profile)
            ),
        }
    )
    decision = _apply_policy_overrides(decision, normalized_message)
    decision = _apply_booking_slot_overrides(
        decision,
        state,
        normalized_message,
        runtime.context.business_profile,
    )
    query_scope = _query_scope_for_decision(
        decision.intent,
        normalized_message,
        runtime.context.business_profile,
    )
    decision = _apply_escalation_scope_overrides(
        decision,
        normalized_message,
        query_scope,
    )
    decision = _apply_query_scope_overrides(decision, query_scope)
    current_slots = cast(
        ExtractedSlots,
        decision.slots.model_dump(exclude_none=True),
    )
    decision, query_scope, current_slots = _apply_pending_booking_context(
        decision,
        state,
        current_slots,
        normalized_message,
        runtime.context.business_profile,
        query_scope,
    )
    extracted_slots = _merge_follow_up_slots(state, decision.intent, current_slots)

    update: AgentStateUpdate = {
        "workflow_stage": "planned",
        "detected_intent": decision.intent,
        "planner_confidence": decision.confidence,
        "extracted_slots": extracted_slots,
        "small_talk_topic": decision.small_talk_topic,
    }
    if query_scope is not None:
        update["query_scope"] = query_scope

    return update


def retrieve_business_knowledge(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None or runtime.context.rag_retriever is None:
        raise RuntimeError("RAG retriever context is required for the rag intent")

    normalized_message = state.get("normalized_message")
    if not normalized_message:
        raise ValueError("A normalized message is required before retrieval")

    documents = runtime.context.rag_retriever.invoke(
        {
            "query": normalized_message,
            "business_type": runtime.context.business_profile.business_type,
        }
    )
    retrieved_documents = [
        serialized
        for document in documents
        if (serialized := _serialize_document(document)) is not None
    ]

    return {
        "workflow_stage": "knowledge_retrieved",
        "retrieval_query": normalized_message,
        "retrieved_documents": retrieved_documents,
    }


def _serialize_document(document: Document) -> RetrievedDocument | None:
    content = document.page_content.strip()
    if not content:
        return None

    metadata = document.metadata
    source = str(metadata.get("source") or "business_knowledge")
    retrieved_document: RetrievedDocument = {
        "content": content,
        "source": source,
    }

    category = metadata.get("category")
    if category:
        retrieved_document["category"] = str(category)

    business_type = metadata.get("business_type")
    if business_type:
        retrieved_document["business_type"] = str(business_type)

    similarity = metadata.get("similarity")
    if isinstance(similarity, (int, float)) and not isinstance(similarity, bool):
        retrieved_document["similarity"] = float(similarity)

    return retrieved_document


def _format_prior_conversation(state: AgentState) -> str:
    history = state.get("conversation_history", [])
    prior_turns = history[:-1]
    if not prior_turns:
        return "No prior conversation."

    return "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in prior_turns
    )


def _merge_follow_up_slots(
    state: AgentState,
    current_intent: PlannerIntent,
    current_slots: ExtractedSlots,
) -> ExtractedSlots:
    expected_stage = FOLLOW_UP_STAGES.get(current_intent)
    if (
        expected_stage is None
        or state.get("detected_intent") != current_intent
        or state.get("previous_workflow_stage") != expected_stage
    ):
        return current_slots

    return cast(
        ExtractedSlots,
        {**state.get("extracted_slots", {}), **current_slots},
    )


def _apply_pending_booking_context(
    decision: PlannerDecision,
    state: AgentState,
    current_slots: ExtractedSlots,
    normalized_message: str,
    business_profile: BusinessProfile,
    query_scope: QueryScope | None,
) -> tuple[PlannerDecision, QueryScope | None, ExtractedSlots]:
    pending_slots = _pending_booking_slots(state)
    if not pending_slots or decision.intent != "clarification":
        return decision, query_scope, current_slots

    merged_slots = cast(ExtractedSlots, {**pending_slots, **current_slots})
    if _is_pending_booking_recall_question(normalized_message):
        return decision, query_scope, merged_slots

    raw_query_scope = determine_query_scope(normalized_message, business_profile)
    if (
        raw_query_scope not in {"off_domain", "business_question"}
        or not _is_pending_booking_guidance_question(normalized_message)
    ):
        return decision, query_scope, merged_slots

    return (
        PlannerDecision(
            intent="book_appointment",
            confidence=max(decision.confidence, 0.95),
            small_talk_topic=None,
            slots=PlannerSlots(
                service=merged_slots.get("service"),
                date=merged_slots.get("date"),
                time=merged_slots.get("time"),
                escalation_reason=None,
            ),
        ),
        None,
        merged_slots,
    )


def _pending_booking_slots(state: AgentState) -> ExtractedSlots:
    extracted_slots = cast(ExtractedSlots, dict(state.get("extracted_slots", {})))
    if not extracted_slots:
        return {}

    previous_stage = state.get("previous_workflow_stage") or state.get(
        "workflow_stage"
    )
    if previous_stage == "booking_information_required":
        return extracted_slots

    if (
        state.get("detected_intent") == "clarification"
        and extracted_slots.get("service")
        and state.get("active_appointment") is None
    ):
        return extracted_slots

    return {}


def _apply_policy_overrides(
    decision: PlannerDecision,
    normalized_message: str,
) -> PlannerDecision:
    if _is_customer_dissatisfied(normalized_message):
        return PlannerDecision(
            intent="human_escalation",
            confidence=max(decision.confidence, 0.99),
            small_talk_topic=None,
            slots=PlannerSlots(
                service=None,
                date=None,
                time=None,
                escalation_reason=normalized_message,
            ),
        )

    return decision


def _apply_booking_slot_overrides(
    decision: PlannerDecision,
    state: AgentState,
    normalized_message: str,
    business_profile: BusinessProfile,
) -> PlannerDecision:
    if decision.intent not in {"clarification", "rag"}:
        return decision

    slot_updates = _extract_booking_slot_fragments(
        normalized_message,
        business_profile,
    )
    if not slot_updates:
        return decision

    previous_stage = state.get("previous_workflow_stage") or state.get(
        "workflow_stage"
    )
    is_booking_follow_up = (
        state.get("detected_intent") == "book_appointment"
        and previous_stage == "booking_information_required"
    )
    is_standalone_slot_fragment = (
        ("date" in slot_updates or "time" in slot_updates)
        and not _looks_like_explicit_question(normalized_message)
    )
    if not is_booking_follow_up and not is_standalone_slot_fragment:
        return decision

    base_slots: ExtractedSlots = {}
    if is_booking_follow_up or state.get("extracted_slots"):
        base_slots.update(cast(ExtractedSlots, state.get("extracted_slots", {})))

    merged_slots = {
        **base_slots,
        **decision.slots.model_dump(exclude_none=True),
        **slot_updates,
    }
    return PlannerDecision(
        intent="book_appointment",
        confidence=max(decision.confidence, 0.95),
        small_talk_topic=None,
        slots=PlannerSlots(
            service=merged_slots.get("service"),
            date=merged_slots.get("date"),
            time=merged_slots.get("time"),
            escalation_reason=None,
        ),
    )


def _extract_booking_slot_fragments(
    message: str,
    business_profile: BusinessProfile,
) -> ExtractedSlots:
    slots: ExtractedSlots = {}
    matched_service = match_supported_service(business_profile, message)
    if matched_service is not None:
        slots["service"] = matched_service

    date_fragment = _extract_date_fragment(message)
    if date_fragment is not None:
        slots["date"] = date_fragment

    time_fragment = _extract_time_fragment(message)
    if time_fragment is not None:
        slots["time"] = time_fragment

    return slots


def _extract_date_fragment(message: str) -> str | None:
    relative_match = search(
        r"\b(yesterday|today|tomorrow|last\s+(?:week|month|year|night|"
        r"mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
        r"fri(?:day)?|sat(?:urday)?|sun(?:day)?))\b",
        message,
        flags=IGNORECASE,
    )
    if relative_match is not None:
        return relative_match.group(1)

    numeric_match = search(
        r"\b(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}(?:/\d{2,4})?)\b",
        message,
    )
    if numeric_match is not None:
        return numeric_match.group(0)

    month_match = search(
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\.?\s+\d{1,2}(?:st|nd|rd|th)?"
        r"(?:,?\s+\d{2,4})?\b",
        message,
        flags=IGNORECASE,
    )
    if month_match is not None:
        return month_match.group(0)

    weekday_match = search(
        r"\b(mon(?:day)?|tue(?:sday)?|wed(?:nesday)?|thu(?:rsday)?|"
        r"fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
        message,
        flags=IGNORECASE,
    )
    if weekday_match is not None:
        return weekday_match.group(1)

    return None


def _extract_time_fragment(message: str) -> str | None:
    meridiem_match = search(
        r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
        message,
        flags=IGNORECASE,
    )
    if meridiem_match is not None:
        return meridiem_match.group(0)

    military_match = search(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", message)
    if military_match is not None:
        return military_match.group(0)

    return None


def _query_scope_for_decision(
    intent: PlannerIntent,
    normalized_message: str,
    business_profile: BusinessProfile,
) -> QueryScope | None:
    if intent in {"rag", "human_escalation"}:
        return determine_query_scope(normalized_message, business_profile)

    if intent != "clarification":
        return None

    query_scope = determine_query_scope(normalized_message, business_profile)
    if query_scope == "business_question":
        return None
    if query_scope == "off_domain" and not _looks_like_explicit_question(
        normalized_message
    ):
        return None
    return query_scope


def _apply_escalation_scope_overrides(
    decision: PlannerDecision,
    normalized_message: str,
    query_scope: QueryScope | None,
) -> PlannerDecision:
    if decision.intent != "human_escalation":
        return decision

    if _is_allowed_escalation(normalized_message, query_scope):
        return decision

    return PlannerDecision(
        intent="clarification",
        confidence=max(decision.confidence, 0.95),
        small_talk_topic=None,
        slots=PlannerSlots(
            service=None,
            date=None,
            time=None,
            escalation_reason=None,
        ),
    )


def _apply_query_scope_overrides(
    decision: PlannerDecision,
    query_scope: QueryScope | None,
) -> PlannerDecision:
    if decision.intent != "rag" or query_scope in {None, "business_question"}:
        return decision

    return PlannerDecision(
        intent="clarification",
        confidence=max(decision.confidence, 0.95),
        small_talk_topic=None,
        slots=decision.slots,
    )


def _is_customer_dissatisfied(message: str) -> bool:
    normalized = message.lower()
    dissatisfaction_phrases = (
        "not satisfied",
        "unsatisfied",
        "unhappy",
        "not happy",
        "bad experience",
        "poor service",
        "complaint",
        "complain",
        "not acceptable",
        "unacceptable",
        "not resolved",
        "unresolved",
        "bad resolution",
        "your resolution",
    )
    return any(phrase in normalized for phrase in dissatisfaction_phrases)


def _is_allowed_escalation(
    message: str,
    query_scope: QueryScope | None,
) -> bool:
    if query_scope == "business_question":
        return True

    if query_scope in {
        "cross_business_profile",
        "staff_personal_info",
        "sensitive_request",
    }:
        return False

    if _looks_like_unrelated_support_request(message):
        return False

    return _is_customer_dissatisfied(message) or _is_explicit_handoff_request(message)


def _is_explicit_handoff_request(message: str) -> bool:
    normalized = message.lower()
    handoff_phrases = (
        "speak to a person",
        "speak with a person",
        "talk to a person",
        "talk with a person",
        "speak to someone",
        "speak with someone",
        "talk to someone",
        "talk with someone",
        "human",
        "real person",
        "representative",
        "manager",
        "front desk",
        "receptionist",
    )
    return any(phrase in normalized for phrase in handoff_phrases)


def _looks_like_unrelated_support_request(message: str) -> bool:
    normalized = message.lower()
    support_terms = (
        "issue",
        "problem",
        "trouble",
        "troubleshoot",
        "setup",
        "set up",
        "not working",
        "facing",
        "help with",
        "help regarding",
        "not satisfied",
        "unsatisfied",
        "unhappy",
        "not happy",
        "complaint",
        "complain",
    )
    unrelated_terms = (
        "keyboard",
        "mouse",
        "computer",
        "laptop",
        "pc",
        "printer",
        "router",
        "wifi",
        "wi-fi",
        "internet",
        "software",
        "hardware",
        "phone",
        "mobile",
        "app",
        "grand kid",
        "grandkid",
        "grandchild",
        "child",
        "son",
        "daughter",
        "family",
        "homework",
    )
    return any(term in normalized for term in support_terms) and any(
        term in normalized for term in unrelated_terms
    )


def _looks_like_explicit_question(message: str) -> bool:
    normalized = message.strip().lower()
    question_starters = (
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "do ",
        "does ",
        "can ",
        "could ",
        "is ",
        "are ",
    )
    return normalized.endswith("?") or normalized.startswith(question_starters)


def _is_pending_booking_recall_question(message: str) -> bool:
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
        "what did we discuss",
    )
    return any(phrase in normalized for phrase in recall_phrases)


def _is_pending_booking_guidance_question(message: str) -> bool:
    normalized = " ".join(message.lower().split()).rstrip("?")
    guidance_phrases = (
        "what do you suggest",
        "what would you suggest",
        "what do you recommend",
        "what would you recommend",
        "what should i do",
        "what should i choose",
        "what next",
        "what now",
        "any suggestion",
        "any suggestions",
        "suggest something",
        "recommend something",
        "help me choose",
    )
    return any(phrase in normalized for phrase in guidance_phrases)
