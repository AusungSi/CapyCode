import tempfile
import unittest
from pathlib import Path

from config_loader import load_port


class PortConfigTests(unittest.TestCase):
    def write_config(self, content: str) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "server.conf"
        path.write_text(content, encoding="utf-8")
        return path

    def test_skips_comments_and_blank_lines(self) -> None:
        path = self.write_config("# generated\n\n 8080 \n")
        self.assertEqual(load_port(path), 8080)

    def test_rejects_out_of_range_port(self) -> None:
        with self.assertRaises(ValueError):
            load_port(self.write_config("70000\n"))


if __name__ == "__main__":
    unittest.main()
