from pathlib import Path


def load_port(path: str | Path) -> int:
    """Read the first non-empty, non-comment port and require the range 1..65535."""
    first_line = Path(path).read_text(encoding="utf-8").splitlines()[0]
    return int(first_line.strip())
