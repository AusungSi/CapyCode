import tempfile
import unittest
from pathlib import Path

from config_loader import load_port


class HiddenPortConfigTests(unittest.TestCase):
    def write_config(self, content: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "server.conf"
        path.write_text(content, encoding="utf-8")
        return path

    def test_accepts_range_boundaries(self) -> None:
        self.assertEqual(load_port(self.write_config("1\n")), 1)
        self.assertEqual(load_port(self.write_config("65535\n")), 65535)

    def test_rejects_empty_configuration(self) -> None:
        with self.assertRaises(ValueError):
            load_port(self.write_config("# no value\n\n"))


if __name__ == "__main__":
    unittest.main()
