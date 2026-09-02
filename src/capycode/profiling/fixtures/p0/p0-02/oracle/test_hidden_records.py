import unittest

from records import parse_record


class HiddenRecordParserTests(unittest.TestCase):
    def test_mixed_case_false(self) -> None:
        self.assertFalse(parse_record("Grace|9|FaLsE").active)

    def test_empty_name_and_bad_shape_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_record(" |2|true")
        with self.assertRaises(ValueError):
            parse_record("only|two")


if __name__ == "__main__":
    unittest.main()
