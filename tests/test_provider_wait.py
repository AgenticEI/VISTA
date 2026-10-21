"""A provider outage cannot unwind the in-flight agent decision."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai.exceptions import ModelHTTPError

from rpent.planner.provider_wait import PersistentRequest, retry_reason


class ProviderWaitTest(unittest.TestCase):
    def test_quota_and_server_errors_keep_the_original_request_until_recovery(self):
        with tempfile.TemporaryDirectory() as scratch:
            messages, settings, parameters = [], {}, object()
            calls = []
            wait_states = []
            path = Path(scratch) / "provider-wait.json"

            async def request(*args):
                calls.append(args)
                if len(calls) <= 5:
                    raise ModelHTTPError(403, "test", {"error": "insufficient_user_quota"})
                if len(calls) == 6:
                    raise ModelHTTPError(503, "test", {"error": "temporary"})
                return "same-scene-result"

            async def sleep(_interval):
                wait_states.append(json.loads(path.read_text())["state"])

            with patch("rpent.planner.provider_wait.asyncio.sleep", sleep):
                result = asyncio.run(PersistentRequest(request, path)(messages, settings, parameters))

            self.assertEqual(result, "same-scene-result")
            self.assertEqual(len(calls), 7)
            self.assertTrue(all(actual[0] is messages and actual[1] is settings and actual[2] is parameters for actual in calls))
            self.assertEqual(wait_states, ["waiting"] * 6)
            state = json.loads(path.read_text())
            self.assertEqual(state["state"], "recovered")
            self.assertEqual(state["attempts"], 6)

    def test_deterministic_auth_and_request_errors_are_reported(self):
        self.assertEqual(retry_reason(ModelHTTPError(402, "test", {"error": "budget_paused"})), "payment_required")
        self.assertIsNone(retry_reason(ModelHTTPError(401, "test", {"error": "bad_key"})))
        self.assertIsNone(retry_reason(ModelHTTPError(403, "test", {"error": "forbidden"})))
        self.assertIsNone(retry_reason(ModelHTTPError(400, "test", {"error": "bad_schema"})))


if __name__ == "__main__":
    unittest.main()
