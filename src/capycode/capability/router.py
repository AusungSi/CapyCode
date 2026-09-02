from __future__ import annotations

from typing import TYPE_CHECKING

from .measurements import ProfiledRoutingArtifact
from .models import CapabilityDecision, RouteDecision
from .profile import Profile, ProfileRegistry

if TYPE_CHECKING:
    from capycode.core.state import SessionState


class ProfileRouter:
    def __init__(
        self,
        registry: ProfileRegistry,
        *,
        default_profile_id: str | None = None,
        profiled_routing: ProfiledRoutingArtifact | None = None,
    ) -> None:
        self.registry = registry
        self.default_profile_id = default_profile_id
        self.profiled_routing = profiled_routing

    def select(
        self,
        decision: CapabilityDecision,
        state: SessionState,
        *,
        preferred_profile_id: str | None = None,
    ) -> RouteDecision:
        candidates = self.registry.for_capability(decision.capability)
        if preferred_profile_id:
            preferred = self.registry.get(preferred_profile_id)
            if preferred is not None and preferred.capability == decision.capability:
                candidates = [preferred]
        if not candidates and self.default_profile_id:
            default = self.registry.get(self.default_profile_id)
            if default is not None:
                candidates = [default]
        if not candidates:
            raise ValueError(f"no profile configured for capability: {decision.capability.value}")
        selected = (
            self.profiled_routing.selection_for(decision.capability)
            if self.profiled_routing
            else None
        )
        profile = next(
            (
                item
                for item in candidates
                if selected is not None and item.profile_id == selected.profile_id
            ),
            None,
        )
        if profile is None:
            profile = min(candidates, key=lambda item: (item.max_output_tokens, item.profile_id))
        reason = "; ".join(item.signal for item in decision.evidence)
        if selected is not None and profile.profile_id == selected.profile_id:
            reason += (
                f"; measured profile: n={selected.samples}, success={selected.success_rate:.1%}, "
                f"expected_cost={selected.expected_cost_per_success:.6f}"
            )
        else:
            reason += "; no eligible measured profile; deterministic budget fallback"
        return RouteDecision(
            capability=decision.capability,
            profile_id=profile.profile_id,
            model_ref=profile.model_ref,
            reason=reason,
            escalation_level=state.capability_failures.get(decision.capability.value, 0),
        )

    def profile(self, decision: RouteDecision) -> Profile:
        profile = self.registry.get(decision.profile_id)
        if profile is None:
            raise ValueError(f"unknown profile: {decision.profile_id}")
        return profile
