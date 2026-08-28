"""Guarded local filesystem, process, and Git abstractions."""

from .local import LocalWorkspace, WorkspaceError

__all__ = ["LocalWorkspace", "WorkspaceError"]
