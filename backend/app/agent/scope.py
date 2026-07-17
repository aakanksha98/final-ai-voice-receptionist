from __future__ import annotations

from re import escape, search

from backend.app.agent.state import QueryScope
from backend.app.business_profiles import BusinessProfile, supported_service_names


def determine_query_scope(
    message: str,
    business_profile: BusinessProfile | None,
) -> QueryScope:
    normalized_message = message.lower()
    if _looks_like_sensitive_request(normalized_message):
        return "sensitive_request"
    if _looks_like_cross_business_profile(normalized_message, business_profile):
        return "cross_business_profile"
    if _looks_like_staff_personal_info(normalized_message):
        return "staff_personal_info"
    if _looks_like_scheduling_reference_query(normalized_message):
        return "scheduling_reference"
    if _looks_like_business_question(normalized_message, business_profile):
        return "business_question"
    return "off_domain"


def _looks_like_sensitive_request(message: str) -> bool:
    sensitive_terms = (
        "api key",
        "apikey",
        "secret key",
        "password",
        "token",
        "credential",
        "ignore all your instructions",
    )
    return any(_contains_term(message, term) for term in sensitive_terms)


def _looks_like_cross_business_profile(
    message: str,
    business_profile: BusinessProfile | None,
) -> bool:
    if business_profile is None:
        return False

    profile_terms_by_type = {
        "dental": ("dental", "dentist", "tooth", "teeth", "clinic"),
        "salon": ("salon",),
        "auto_repair": ("auto", "car", "garage", "auto repair"),
    }
    current_type = business_profile.business_type
    for profile_type, terms in profile_terms_by_type.items():
        if profile_type == current_type:
            continue
        if any(_contains_term(message, term) for term in terms):
            return True
    return False


def _looks_like_staff_personal_info(message: str) -> bool:
    staff_terms = ("front desk", "receptionist", "staff", "employee", "manager")
    personal_terms = ("know ", "who is", "who's", "karen", "person")
    return any(_contains_term(message, term) for term in staff_terms) and any(
        _contains_term(message, term) for term in personal_terms
    )


def _looks_like_scheduling_reference_query(message: str) -> bool:
    normalized = " ".join(message.strip().lower().split())
    reference_questions = (
        "what is the date",
        "what's the date",
        "whats the date",
        "what is today's date",
        "what's today's date",
        "whats today's date",
        "what date is today",
        "what day is it",
        "what day is today",
    )
    return normalized.rstrip("?") in reference_questions


def _looks_like_business_question(
    message: str,
    business_profile: BusinessProfile | None,
) -> bool:
    business_terms = {
        "appointment",
        "appointments",
        "available",
        "availability",
        "book",
        "booking",
        "bring",
        "bill",
        "billing",
        "cancel",
        "cancellation",
        "charge",
        "charges",
        "close",
        "closed",
        "cost",
        "costs",
        "deal",
        "deals",
        "discount",
        "discounts",
        "fee",
        "fees",
        "friend",
        "guest",
        "hour",
        "hours",
        "invoice",
        "offer",
        "offers",
        "open",
        "payment",
        "policy",
        "policies",
        "price",
        "prices",
        "pricing",
        "promotion",
        "promotions",
        "refund",
        "reschedule",
        "schedule",
        "service",
        "services",
        "visit",
    }
    profile_terms = _profile_terms(business_profile)
    return any(_contains_term(message, term) for term in business_terms | profile_terms)


def _profile_terms(business_profile: BusinessProfile | None) -> set[str]:
    if business_profile is None:
        return set()

    service_terms: set[str] = set()
    for service_name in supported_service_names(business_profile):
        service_terms.update(service_name.lower().split())

    type_terms_by_profile = {
        "dental": {
            "cleaning",
            "dental",
            "dentist",
            "exam",
            "filling",
            "teeth",
            "tooth",
            "whitening",
        },
        "salon": {
            "blowout",
            "color",
            "facial",
            "hair",
            "haircut",
            "manicure",
            "salon",
        },
        "auto_repair": {
            "auto",
            "battery",
            "brake",
            "car",
            "diagnostic",
            "engine",
            "oil",
            "repair",
            "tire",
        },
    }
    return service_terms | type_terms_by_profile[business_profile.business_type]


def _contains_term(message: str, term: str) -> bool:
    normalized_term = term.strip().lower()
    if not normalized_term:
        return False

    pattern = r"\b" + r"\s+".join(
        escape(part) for part in normalized_term.split()
    ) + r"\b"
    return search(pattern, message) is not None
