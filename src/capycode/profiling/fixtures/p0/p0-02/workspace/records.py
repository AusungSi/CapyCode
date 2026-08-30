from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    name: str
    score: int
    active: bool


def parse_record(line: str) -> Record:
    """Parse ``name|score|active`` where active is exactly true or false, ignoring case."""
    parts = [part.strip() for part in line.split("|")]
    if len(parts) != 3 or not parts[0]:
        raise ValueError("invalid record")
    name, score, active = parts
    if active.lower() not in {"true", "false"}:
        raise ValueError("invalid active flag")
    return Record(name=name, score=int(score), active=bool(active))
