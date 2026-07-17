from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    ActiveAppointment,
    AgentState,
    AgentStateUpdate,
    RescheduleResult,
    RescheduleSlot,
)
from backend.app.agent.booking import validate_appointment_schedule
from backend.app.tools.rescheduling import (
    RescheduleConfirmation,
    RescheduleRequest,
)


REQUIRED_RESCHEDULE_SLOTS: tuple[RescheduleSlot, ...] = (
    "date",
    "time",
)
ACTIVE_APPOINTMENT_STATUSES = {"confirmed", "rescheduled"}


def execute_reschedule(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the reschedule intent")

    appointment = _current_active_appointment(state)
    if appointment is None:
        return {
            "workflow_stage": "no_active_appointment",
            "missing_reschedule_slots": [],
            "reschedule_result": None,
        }

    extracted_slots = state.get("extracted_slots", {})
    missing_slots = [
        slot
        for slot in REQUIRED_RESCHEDULE_SLOTS
        if not _clean_slot_value(extracted_slots.get(slot))
    ]
    if missing_slots:
        return {
            "workflow_stage": "reschedule_information_required",
            "missing_reschedule_slots": missing_slots,
            "reschedule_result": None,
        }

    schedule_violation = validate_appointment_schedule(
        runtime.context.business_profile,
        _clean_slot_value(extracted_slots.get("date")),
        _clean_slot_value(extracted_slots.get("time")),
    )
    if schedule_violation is not None:
        return {
            "workflow_stage": "appointment_outside_business_hours",
            "missing_reschedule_slots": [],
            "reschedule_result": None,
            "schedule_violation": schedule_violation,
            "business_hours": runtime.context.business_profile.booking_hours.description,
        }

    if runtime.context.reschedule_tool is None:
        raise RuntimeError(
            "Reschedule tool context is required for complete reschedule details"
        )

    request = RescheduleRequest(
        service=appointment["service"],
        new_date=extracted_slots["date"],
        new_time=extracted_slots["time"],
    )
    raw_result = runtime.context.reschedule_tool.invoke(request.model_dump())
    confirmation = RescheduleConfirmation.model_validate(raw_result)
    rescheduled_appointment = cast(ActiveAppointment, confirmation.model_dump())

    return {
        "workflow_stage": "appointment_rescheduled",
        "missing_reschedule_slots": [],
        "reschedule_result": cast(
            RescheduleResult,
            confirmation.model_dump(),
        ),
        "active_appointment": rescheduled_appointment,
    }


def _clean_slot_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _current_active_appointment(state: AgentState) -> ActiveAppointment | None:
    appointment = state.get("active_appointment")
    if appointment is None:
        return None

    if appointment.get("status") in ACTIVE_APPOINTMENT_STATUSES:
        return appointment

    return None
