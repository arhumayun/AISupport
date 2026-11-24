"""Synthetic authentication and personal-data authorization tests."""

from pathlib import Path

import pytest

from ai_support.agents.profiles import ProfileAuthorizationError, SyntheticProfileStore
from ai_support.security.auth import AuthenticationError, AuthService


@pytest.fixture
def profiles(tmp_path: Path) -> SyntheticProfileStore:
    store = SyntheticProfileStore(
        tmp_path / "profiles.db", Path("data/app/synthetic_profiles.json")
    )
    store.initialize()
    return store


def test_signed_session_binds_server_verified_audience(profiles: SyntheticProfileStore) -> None:
    auth = AuthService(
        profiles, "a-test-secret-that-is-long-enough", "demo-code", clock=lambda: 1000
    )
    token, principal, ttl = auth.login("s100001", "demo-code")

    assert principal.user_id == "S100001"
    assert principal.audience == "student"
    assert ttl == 3600
    assert auth.verify(token) == principal

    with pytest.raises(AuthenticationError):
        auth.login("E100001", "wrong")
    with pytest.raises(AuthenticationError):
        auth.verify(token[:-1] + ("A" if token[-1] != "A" else "B"))


def test_expired_and_malformed_sessions_are_rejected(profiles: SyntheticProfileStore) -> None:
    now = [1000]
    auth = AuthService(
        profiles,
        "a-test-secret-that-is-long-enough",
        "demo-code",
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    token, _, _ = auth.login("S100001", "demo-code")
    now[0] = 1006

    with pytest.raises(AuthenticationError, match="expired"):
        auth.verify(token)
    with pytest.raises(AuthenticationError, match="malformed"):
        auth.verify("not-a-token")


def test_profile_lookup_uses_authenticated_identity_only(profiles: SyntheticProfileStore) -> None:
    student = profiles.student_profile("S100001")
    progress = profiles.course_progress("S100001")
    staff = profiles.staff_profile("E100001")

    assert student.name == "Amina Rahman"
    assert progress.progress_percentage == 50.0
    assert staff.name == "Dr Layla Karim"
    with pytest.raises(ProfileAuthorizationError):
        profiles.student_profile("E100001")
    with pytest.raises(ProfileAuthorizationError):
        profiles.staff_profile("S100001")
