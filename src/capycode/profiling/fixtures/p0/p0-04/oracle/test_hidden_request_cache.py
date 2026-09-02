import unittest

from request_cache import RequestCache


class HiddenRequestCacheTests(unittest.TestCase):
    def test_clear_only_affects_its_instance(self) -> None:
        first = RequestCache()
        second = RequestCache()
        first.put("key", "first")
        second.put("key", "second")
        first.clear()
        self.assertIsNone(first.get("key"))
        self.assertEqual(second.get("key"), "second")


if __name__ == "__main__":
    unittest.main()
