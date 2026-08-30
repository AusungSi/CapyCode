import unittest

from records import Record, parse_record


class RecordParserTests(unittest.TestCase):
    def test_false_is_not_treated_as_truthy_text(self) -> None:
        self.assertEqual(parse_record("Ada | 42 | false"), Record("Ada", 42, False))

    def test_true_and_whitespace(self) -> None:
        self.assertEqual(parse_record(" Lin | 7 | TRUE "), Record("Lin", 7, True))

    def test_invalid_flag_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_record("Sam|5|yes")


if __name__ == "__main__":
    unittest.main()
