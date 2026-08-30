from __future__ import annotations

from typing import TYPE_CHECKING

from .models import Capability, CapabilityDecision, CapabilityEvidence

if TYPE_CHECKING:
    from capycode.core.state import SessionState


class CapabilityDetector:
    """Explainable state-based detector for the current coding step."""

    def detect(self, state: SessionState) -> CapabilityDecision:
        if state.last_tests_passed is False or state.last_error:
            return self._decision(Capability.DIAGNOSIS, "previous test or tool failure")
        if state.modified_files and state.last_tests_passed is not True:
            return self._decision(Capability.VERIFICATION, "workspace has unverified modifications")
        if state.modified_files and state.last_tests_passed is True:
            return self._decision(Capability.VERIFICATION, "modified files have a test result")
        # Once retrieval/understanding has produced context for a coding task,
        # move into the edit phase explicitly. Without this transition a long
        # SWE-bench statement can remain in the planning profile indefinitely,
        # whose tools intentionally exclude write operations.
        if state.current_capability in {
            Capability.UNDERSTANDING.value,
            Capability.PLANNING.value,
        } and self._is_coding_task(state.task):
            return self._decision(
                Capability.EDITING, "context is available; implementation can proceed"
            )
        if state.relevant_files:
            if len(state.relevant_files) >= 2 or len(state.task.split()) > 18:
                return self._decision(Capability.PLANNING, "multiple files or a complex task")
            return self._decision(Capability.UNDERSTANDING, "relevant files are available")
        return self._decision(Capability.RETRIEVAL, "no relevant files have been inspected")

    @staticmethod
    def _is_coding_task(task: str) -> bool:
        words = task.lower()
        return any(
            marker in words
            for marker in (
                "fix",
                "bug",
                "implement",
                "change",
                "update",
                "add ",
                "remove",
                "refactor",
                "修复",
                "修改",
                "实现",
            )
        )

    @staticmethod
    def _decision(capability: Capability, signal: str) -> CapabilityDecision:
        return CapabilityDecision(
            capability=capability,
            confidence=0.9,
            evidence=(CapabilityEvidence(signal=signal, weight=0.9),),
        )
