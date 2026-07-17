from __future__ import annotations

from dataclasses import dataclass
from re import sub
from typing import Literal


BusinessType = Literal["dental", "salon", "auto_repair"]


@dataclass(frozen=True)
class ServiceCatalogItem:
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class BusinessHours:
    description: str
    weekly_windows: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class BusinessProfile:
    business_type: BusinessType
    label: str
    example_business_name: str
    services: tuple[ServiceCatalogItem, ...]
    booking_hours: BusinessHours


BUSINESS_NAME_MAX_LENGTH = 80
DEFAULT_BUSINESS_TYPE: BusinessType = "dental"


BUSINESS_PROFILES: dict[BusinessType, BusinessProfile] = {
    "dental": BusinessProfile(
        business_type="dental",
        label="Dental Clinic",
        example_business_name="BrightSmile Dental",
        services=(
            ServiceCatalogItem("dental cleaning", ("cleaning", "teeth cleaning")),
            ServiceCatalogItem("dental exam", ("checkup", "check-up", "exam")),
            ServiceCatalogItem("teeth whitening", ("whitening",)),
            ServiceCatalogItem("filling", ("cavity filling", "dental filling")),
            ServiceCatalogItem(
                "emergency dental visit",
                ("emergency visit", "tooth pain visit"),
            ),
        ),
        booking_hours=BusinessHours(
            description=(
                "Monday through Friday from 8 AM to 5 PM, and Saturday "
                "from 9 AM to 1 PM"
            ),
            weekly_windows=(
                (0, 8 * 60, 17 * 60),
                (1, 8 * 60, 17 * 60),
                (2, 8 * 60, 17 * 60),
                (3, 8 * 60, 17 * 60),
                (4, 8 * 60, 17 * 60),
                (5, 9 * 60, 13 * 60),
            ),
        ),
    ),
    "salon": BusinessProfile(
        business_type="salon",
        label="Salon",
        example_business_name="Luxe Hair Studio",
        services=(
            ServiceCatalogItem("haircut", ("hair cut", "trim")),
            ServiceCatalogItem("blowout", ("blow dry", "blow-dry")),
            ServiceCatalogItem("hair color", ("color", "root touch up", "root touch-up")),
            ServiceCatalogItem("manicure", ("nails", "basic manicure")),
            ServiceCatalogItem("facial", ("skin facial",)),
        ),
        booking_hours=BusinessHours(
            description="Tuesday through Saturday from 10 AM to 7 PM",
            weekly_windows=(
                (1, 10 * 60, 19 * 60),
                (2, 10 * 60, 19 * 60),
                (3, 10 * 60, 19 * 60),
                (4, 10 * 60, 19 * 60),
                (5, 10 * 60, 19 * 60),
            ),
        ),
    ),
    "auto_repair": BusinessProfile(
        business_type="auto_repair",
        label="Auto Repair Shop",
        example_business_name="TurboFix Garage",
        services=(
            ServiceCatalogItem("oil change", ("oil service",)),
            ServiceCatalogItem("brake inspection", ("brake check", "brakes")),
            ServiceCatalogItem("tire rotation", ("rotate tires", "tyre rotation")),
            ServiceCatalogItem("battery diagnostic", ("battery check",)),
            ServiceCatalogItem("engine diagnostic", ("check engine light", "diagnostic")),
        ),
        booking_hours=BusinessHours(
            description=(
                "Monday through Friday from 7:30 AM to 6 PM, and Saturday "
                "from 8 AM to 2 PM"
            ),
            weekly_windows=(
                (0, 7 * 60 + 30, 18 * 60),
                (1, 7 * 60 + 30, 18 * 60),
                (2, 7 * 60 + 30, 18 * 60),
                (3, 7 * 60 + 30, 18 * 60),
                (4, 7 * 60 + 30, 18 * 60),
                (5, 8 * 60, 14 * 60),
            ),
        ),
    ),
}


def get_business_profile(business_type: BusinessType) -> BusinessProfile:
    try:
        return BUSINESS_PROFILES[business_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported business type: {business_type}") from exc


def default_business_profile() -> BusinessProfile:
    return get_business_profile(DEFAULT_BUSINESS_TYPE)


def is_supported_business_type(value: str) -> bool:
    return value in BUSINESS_PROFILES


def sanitize_business_name(
    business_name: str,
) -> str:
    cleaned_name = sub(r"[\x00-\x1F\x7F]", " ", business_name)
    cleaned_name = sub(r"\s+", " ", cleaned_name).strip()

    if not cleaned_name:
        raise ValueError("business_name is required")

    return cleaned_name[:BUSINESS_NAME_MAX_LENGTH]


def supported_service_names(profile: BusinessProfile) -> list[str]:
    return [service.name for service in profile.services]


def match_supported_service(
    profile: BusinessProfile,
    requested_service: str,
) -> str | None:
    requested = _normalize_service(requested_service)
    if not requested:
        return None

    for service in profile.services:
        candidates = (service.name, *service.aliases)
        normalized_candidates = [_normalize_service(candidate) for candidate in candidates]
        if any(
            requested == candidate
            or requested in candidate
            or candidate in requested
            for candidate in normalized_candidates
        ):
            return service.name

    return None


def match_known_service(
    requested_service: str,
) -> tuple[BusinessType, str] | None:
    for profile in BUSINESS_PROFILES.values():
        matched_service = match_supported_service(profile, requested_service)
        if matched_service is not None:
            return profile.business_type, matched_service

    return None


def _normalize_service(value: str) -> str:
    cleaned = sub(r"[^a-z0-9]+", " ", value.lower())
    return sub(r"\s+", " ", cleaned).strip()
