from operator import add
from typing import Annotated, Literal, NotRequired, TypedDict


InputStatus = Literal["valid", "invalid"]
SmallTalkTopic = Literal[
    "greeting",
    "assistant_identity",
    "capabilities",
    "courtesy",
]
QueryScope = Literal[
    "business_question",
    "scheduling_reference",
    "off_domain",
    "cross_business_profile",
    "staff_personal_info",
    "sensitive_request",
]
PlannerIntent = Literal[
    "small_talk",
    "rag",
    "book_appointment",
    "cancel_appointment",
    "reschedule_appointment",
    "human_escalation",
    "clarification",
]
BookingSlot = Literal["service", "date", "time"]
BookingStatus = Literal["confirmed"]
CancellationSlot = Literal["active_appointment"]
CancellationStatus = Literal["cancelled"]
RescheduleSlot = Literal["date", "time"]
RescheduleStatus = Literal["rescheduled"]
AppointmentStatus = Literal["confirmed", "cancelled", "rescheduled"]
EscalationStatus = Literal["queued"]
ConversationRole = Literal["user", "assistant"]
WorkflowStage = Literal[
    "ready_for_planning",
    "planned",
    "knowledge_retrieved",
    "booking_information_required",
    "appointment_booked",
    "appointment_already_active",
    "appointment_outside_business_hours",
    "unsupported_service_requested",
    "cancellation_information_required",
    "appointment_cancelled",
    "reschedule_information_required",
    "appointment_rescheduled",
    "no_active_appointment",
    "human_escalation_queued",
    "rejected",
]


class ExtractedSlots(TypedDict, total=False):
    service: str
    date: str
    time: str
    escalation_reason: str


class RetrievedDocument(TypedDict):
    content: str
    source: str
    category: NotRequired[str]
    business_type: NotRequired[str]
    similarity: NotRequired[float]


class ActiveAppointment(TypedDict):
    service: str
    date: str
    time: str
    status: AppointmentStatus


class BookingResult(TypedDict):
    status: BookingStatus
    service: str
    date: str
    time: str


class CancellationResult(TypedDict):
    status: CancellationStatus
    service: str
    date: str
    time: str


class RescheduleResult(TypedDict):
    status: RescheduleStatus
    service: str
    date: str
    time: str


class EscalationResult(TypedDict):
    escalation_id: str
    status: EscalationStatus
    reason: str | None


class ConversationTurn(TypedDict):
    role: ConversationRole
    content: str


class AgentState(TypedDict):
    user_message: str
    conversation_history: Annotated[list[ConversationTurn], add]
    normalized_message: NotRequired[str]
    input_status: NotRequired[InputStatus]
    workflow_stage: NotRequired[WorkflowStage]
    validation_error: NotRequired[str | None]
    detected_intent: NotRequired[PlannerIntent]
    planner_confidence: NotRequired[float]
    extracted_slots: NotRequired[ExtractedSlots]
    small_talk_topic: NotRequired[SmallTalkTopic | None]
    query_scope: NotRequired[QueryScope | None]
    final_response: NotRequired[str | None]
    retrieval_query: NotRequired[str | None]
    retrieved_documents: NotRequired[list[RetrievedDocument]]
    missing_booking_slots: NotRequired[list[BookingSlot]]
    booking_result: NotRequired[BookingResult | None]
    unsupported_service: NotRequired[str | None]
    supported_services: NotRequired[list[str]]
    schedule_violation: NotRequired[str | None]
    business_hours: NotRequired[str | None]
    active_appointment: NotRequired[ActiveAppointment | None]
    missing_cancellation_slots: NotRequired[list[CancellationSlot]]
    cancellation_result: NotRequired[CancellationResult | None]
    missing_reschedule_slots: NotRequired[list[RescheduleSlot]]
    reschedule_result: NotRequired[RescheduleResult | None]
    escalation_result: NotRequired[EscalationResult | None]
    previous_workflow_stage: NotRequired[WorkflowStage | None]


class AgentStateUpdate(TypedDict, total=False):
    normalized_message: str
    conversation_history: list[ConversationTurn]
    input_status: InputStatus
    workflow_stage: WorkflowStage
    validation_error: str | None
    detected_intent: PlannerIntent
    planner_confidence: float
    extracted_slots: ExtractedSlots
    small_talk_topic: SmallTalkTopic | None
    query_scope: QueryScope | None
    final_response: str | None
    retrieval_query: str | None
    retrieved_documents: list[RetrievedDocument]
    missing_booking_slots: list[BookingSlot]
    booking_result: BookingResult | None
    unsupported_service: str | None
    supported_services: list[str]
    schedule_violation: str | None
    business_hours: str | None
    active_appointment: ActiveAppointment | None
    missing_cancellation_slots: list[CancellationSlot]
    cancellation_result: CancellationResult | None
    missing_reschedule_slots: list[RescheduleSlot]
    reschedule_result: RescheduleResult | None
    escalation_result: EscalationResult | None
    previous_workflow_stage: WorkflowStage | None
