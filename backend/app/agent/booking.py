from datetime import date, datetime
from re import search
from typing import cast

from langgraph.runtime import Runtime

from backend.app.agent.context import AgentContext
from backend.app.agent.state import (
    ActiveAppointment,
    AgentState,
    AgentStateUpdate,
    BookingResult,
    BookingSlot,
    ExtractedSlots,
)
from backend.app.business_profiles import (
    BusinessProfile,
    match_known_service,
    match_supported_service,
    supported_service_names,
)
from backend.app.tools.booking import BookingConfirmation, BookingRequest


REQUIRED_BOOKING_SLOTS: tuple[BookingSlot, ...] = ("service", "date", "time")
ACTIVE_APPOINTMENT_STATUSES = {"confirmed", "rescheduled"}


def execute_booking(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> AgentStateUpdate:
    if runtime.context is None:
        raise RuntimeError("Agent context is required for the booking intent")

    if _current_active_appointment(state) is not None:
        return {
            "workflow_stage": "appointment_already_active",
            "missing_booking_slots": [],
            "booking_result": None,
        }

    extracted_slots = cast(ExtractedSlots, dict(state.get("extracted_slots", {})))
    normalized_message = state.get("normalized_message") or state.get(
        "user_message",
        "",
    )
    if not _clean_slot_value(extracted_slots.get("service")):
        matched_message_service = match_supported_service(
            runtime.context.business_profile,
            normalized_message,
        )
        if matched_message_service is not None:
            extracted_slots["service"] = matched_message_service

    missing_slots = [
        slot
        for slot in REQUIRED_BOOKING_SLOTS
        if not _clean_slot_value(extracted_slots.get(slot))
    ]
    unsupported_service = _unsupported_service_from_message(
        runtime.context.business_profile,
        extracted_slots,
        normalized_message,
    )
    if unsupported_service is not None:
        return {
            "workflow_stage": "unsupported_service_requested",
            "missing_booking_slots": [],
            "booking_result": None,
            "unsupported_service": unsupported_service,
            "supported_services": supported_service_names(
                runtime.context.business_profile
            ),
        }

    schedule_violation = validate_appointment_schedule(
        runtime.context.business_profile,
        _clean_slot_value(extracted_slots.get("date")),
        _clean_slot_value(extracted_slots.get("time")),
    )
    if schedule_violation is not None:
        return {
            "workflow_stage": "appointment_outside_business_hours",
            "missing_booking_slots": [],
            "booking_result": None,
            "schedule_violation": schedule_violation,
            "business_hours": runtime.context.business_profile.booking_hours.description,
        }

    if missing_slots:
        return {
            "workflow_stage": "booking_information_required",
            "missing_booking_slots": missing_slots,
            "booking_result": None,
        }

    requested_service = extracted_slots["service"]
    matched_service = match_supported_service(
        runtime.context.business_profile,
        requested_service,
    )
    if matched_service is None:
        return {
            "workflow_stage": "unsupported_service_requested",
            "missing_booking_slots": [],
            "booking_result": None,
            "unsupported_service": requested_service,
            "supported_services": supported_service_names(
                runtime.context.business_profile
            ),
        }

    if runtime.context.booking_tool is None:
        raise RuntimeError(
            "Booking tool context is required for complete booking details"
        )

    canonical_slots = cast(
        ExtractedSlots,
        {**extracted_slots, "service": matched_service},
    )
    request = BookingRequest.model_validate(
        {
            slot: canonical_slots[slot]
            for slot in REQUIRED_BOOKING_SLOTS
        }
    )
    raw_result = runtime.context.booking_tool.invoke(request.model_dump())
    confirmation = BookingConfirmation.model_validate(raw_result)
    appointment = cast(ActiveAppointment, confirmation.model_dump())

    return {
        "workflow_stage": "appointment_booked",
        "extracted_slots": canonical_slots,
        "missing_booking_slots": [],
        "booking_result": cast(BookingResult, confirmation.model_dump()),
        "active_appointment": appointment,
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


def _unsupported_service_from_message(
    profile: BusinessProfile,
    extracted_slots: ExtractedSlots,
    normalized_message: str,
) -> str | None:
    requested_service = _clean_slot_value(extracted_slots.get("service"))
    if requested_service:
        if match_supported_service(profile, requested_service) is None:
            return requested_service
        return None

    known_service = match_known_service(normalized_message)
    if known_service is None:
        return None

    service_business_type, service_name = known_service
    if service_business_type == profile.business_type:
        return None

    return service_name


WEEKDAY_INDEXES = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

MONTH_INDEXES = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


def validate_appointment_schedule(
    profile: BusinessProfile,
    requested_date: str,
    requested_time: str,
) -> str | None:
    parsed_date, date_error = _parse_booking_date(requested_date)
    time_minutes = _parse_time_minutes(requested_time)
    time_error = _time_validation_error(requested_time, time_minutes)
    date_violation = _date_violation(
        requested_date,
        requested_time,
        parsed_date,
        date_error,
        time_minutes,
    )
    weekday = parsed_date.weekday() if parsed_date is not None else _parse_weekday(requested_date)
    if date_violation is not None:
        time_violation = time_error or _time_outside_any_business_window(
            profile,
            requested_time,
            time_minutes,
        )
        details = [date_violation]
        if time_violation is not None:
            details.append(time_violation)
        else:
            details.append(f"Available booking hours are {profile.booking_hours.description}.")
        return " ".join(details)

    if date_error is not None:
        return (
            f"I could not validate {requested_date} as a real appointment date. "
            "Please choose a valid future date."
        )

    if time_error is not None:
        return time_error

    if weekday is None or time_minutes is None:
        return _time_outside_any_business_window(profile, requested_time, time_minutes)

    for window_weekday, start_minutes, end_minutes in profile.booking_hours.weekly_windows:
        if window_weekday == weekday and start_minutes <= time_minutes < end_minutes:
            return None

    return (
        f"{requested_date} at {requested_time} is outside business hours. "
        f"Available booking hours are {profile.booking_hours.description}."
    )


def _parse_weekday(value: str) -> int | None:
    normalized = value.strip().lower()
    for token, weekday in WEEKDAY_INDEXES.items():
        if search(rf"\b{token}\b", normalized):
            return weekday

    return None


def _parse_booking_date(value: str) -> tuple[date | None, str | None]:
    normalized = value.strip().lower()
    if not normalized:
        return None, None

    today = date.today()
    if search(r"\btoday\b", normalized):
        return today, None

    if search(r"\bday after tomorrow\b", normalized):
        return date.fromordinal(today.toordinal() + 2), None

    if search(r"\btomorrow\b", normalized):
        return date.fromordinal(today.toordinal() + 1), None

    if search(r"\byesterday\b", normalized):
        return date.fromordinal(today.toordinal() - 1), None

    numeric_date = _parse_numeric_date(normalized, today)
    if numeric_date[0] is not None or numeric_date[1] is not None:
        return numeric_date

    named_date = _parse_named_month_date(normalized, today)
    if named_date[0] is not None or named_date[1] is not None:
        return named_date

    return None, None


def _parse_numeric_date(
    value: str,
    today: date,
) -> tuple[date | None, str | None]:
    iso_match = search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", value)
    if iso_match is not None:
        year = int(iso_match.group(1))
        month = int(iso_match.group(2))
        day = int(iso_match.group(3))
        return _safe_date(year, month, day)

    slash_match = search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", value)
    if slash_match is None:
        return None, None

    month = int(slash_match.group(1))
    day = int(slash_match.group(2))
    raw_year = slash_match.group(3)
    year = today.year if raw_year is None else _normalize_year(raw_year)
    return _safe_date(year, month, day)


def _parse_named_month_date(
    value: str,
    today: date,
) -> tuple[date | None, str | None]:
    match = search(
        r"\b("
        + "|".join(MONTH_INDEXES)
        + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{2,4}))?\b",
        value,
    )
    if match is None:
        return None, None

    month = MONTH_INDEXES[match.group(1)]
    day = int(match.group(2))
    raw_year = match.group(3)
    year = today.year if raw_year is None else _normalize_year(raw_year)
    return _safe_date(year, month, day)


def _normalize_year(raw_year: str) -> int:
    year = int(raw_year)
    if year < 100:
        return 2000 + year
    return year


def _safe_date(year: int, month: int, day: int) -> tuple[date | None, str | None]:
    try:
        return date(year, month, day), None
    except ValueError:
        return None, "invalid_date"


def _date_violation(
    requested_date: str,
    requested_time: str,
    parsed_date: date | None,
    date_error: str | None,
    time_minutes: int | None,
) -> str | None:
    if date_error is not None:
        return None

    normalized_date = requested_date.strip().lower()
    if not normalized_date:
        return None

    if _looks_like_past_relative_date(normalized_date):
        return f"{requested_date} is in the past. Please choose a future date."

    if parsed_date is None:
        if _parse_weekday(normalized_date) is None:
            return (
                f"I could not validate {requested_date} as a specific appointment "
                "date. Please choose a specific future date."
            )
        return None

    today = date.today()
    if parsed_date < today:
        return f"{requested_date} is in the past. Please choose a future date."

    if parsed_date == today and time_minutes is not None and _time_has_passed_today(
        time_minutes
    ):
        return (
            f"{requested_date} at {requested_time} has already passed. "
            "Please choose a future appointment time."
        )

    return None


def _looks_like_past_relative_date(value: str) -> bool:
    return bool(
        search(r"\byesterday\b", value)
        or search(r"\blast\s+(week|month|year|night)\b", value)
        or search(
            r"\blast\s+("
            + "|".join(WEEKDAY_INDEXES)
            + r")\b",
            value,
        )
    )


def _time_has_passed_today(time_minutes: int) -> bool:
    now = datetime.now()
    return time_minutes <= now.hour * 60 + now.minute


def _parse_time_minutes(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized == "noon":
        return 12 * 60
    if normalized == "midnight":
        return 0

    match = search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", normalized)
    if match is not None:
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        meridiem = match.group(3)
        if hour < 1 or hour > 12 or minute > 59:
            return None

        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0

        return hour * 60 + minute

    match = search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", normalized)
    if match is None:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour == 0 or hour > 12:
        return hour * 60 + minute

    return None


def _time_validation_error(
    requested_time: str,
    time_minutes: int | None,
) -> str | None:
    cleaned_time = requested_time.strip()
    if not cleaned_time or time_minutes is not None:
        return None

    return (
        f"I could not validate {cleaned_time} as an appointment time. "
        "Please provide a specific time, such as 2 PM."
    )


def _time_outside_any_business_window(
    profile: BusinessProfile,
    requested_time: str,
    time_minutes: int | None,
) -> str | None:
    if not requested_time.strip() or time_minutes is None:
        return None

    for _, start_minutes, end_minutes in profile.booking_hours.weekly_windows:
        if start_minutes <= time_minutes < end_minutes:
            return None

    return (
        f"{requested_time} is outside business hours. "
        f"Available booking hours are {profile.booking_hours.description}."
    )
