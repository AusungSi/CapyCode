from __future__ import annotations

from typing import TYPE_CHECKING

from .models import EscalationAction, EscalationDecision
from .profile import ProfileRegistry

if TYPE_CHECKING:
    from capycode.core.state import SessionState


class EscalationPolicy:
    def __init__(self, registry: ProfileRegistry, *, max_retries: int = 1) -> None:
        self.registry = registry
        self.max_retries = max_retries

    def decide(self, state: SessionState, profile_id: str, *, failed: bool) -> EscalationDecision:
        if not failed:
            return EscalationDecision(action=EscalationAction.RETRY, reason="step completed")
        if state.retry_count < self.max_retries:
            return EscalationDecision(action=EscalationAction.RETRY, reason="retry budget remains")
        current = self.registry.get(profile_id)
        profiles = [
            p
            for p in self.registry.all()
            if p.profile_id != profile_id
            and (current is None or p.capability == current.capability)
        ]
        if profiles:
            next_profile = max(profiles, key=lambda p: (p.max_output_tokens, p.profile_id))
            return EscalationDecision(
                action=EscalationAction.SWITCH_PROFILE,
                reason="retry budget exhausted; switching profile",
                next_profile_id=next_profile.profile_id,
            )
        return EscalationDecision(
            action=EscalationAction.RETRY,
            reason="no same-capability escalation profile; retaining current profile",
        )
