from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Capability(StrEnum):
    RETRIEVAL = "repository_retrieval"
    UNDERSTANDING = "code_understanding"
    PLANNING = "task_planning"
    EDITING = "code_editing"
    DIAGNOSIS = "failure_diagnosis"
    VERIFICATION = "verification"


class CapabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: str = Field(min_length=1)
    weight: float = Field(gt=0, le=1)


class CapabilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: Capability
    confidence: float = Field(ge=0, le=1)
    evidence: tuple[CapabilityEvidence, ...] = ()
    alternatives: tuple[Capability, ...] = ()


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability: Capability
    profile_id: str
    model_ref: str
    reason: str = Field(min_length=1)
    escalation_level: int = Field(default=0, ge=0)


class EscalationAction(StrEnum):
    RETRY = "retry"
    SWITCH_PROFILE = "switch_profile"
    FAIL = "fail"


class EscalationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: EscalationAction
    reason: str = Field(min_length=1)
    next_profile_id: str | None = None
