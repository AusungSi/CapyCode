import unittest

from request_cache import RequestCache


class RequestCacheTests(unittest.TestCase):
    def test_instances_are_isolated(self) -> None:
        first = RequestCache()
        second = RequestCache()
        first.put("request-id", "one")
        self.assertIsNone(second.get("request-id"))

    def test_put_and_get(self) -> None:
        cache = RequestCache()
        cache.put("key", "value")
        self.assertEqual(cache.get("key"), "value")


if __name__ == "__main__":
    unittest.main()
