import random

import numpy as np
import pytest
import torch

from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.robots.components.pi05_vla_server import Pi05PolicySessions


def draw():
    return np.array([random.random(), np.random.random(), torch.rand(()).item()], dtype=np.float32)


def test_private_streams_match_serial_and_interleaved_requests():
    serial = Pi05PolicySessions()
    expected = [serial.run("a", 77, i, draw) for i in range(4)]
    interleaved = Pi05PolicySessions()
    for index in range(4):
        actual = interleaved.run("a", 77, index, draw)
        np.testing.assert_array_equal(expected[index], actual)
        interleaved.run("b", 999, index, draw)
    different = interleaved.run("c", 78, 0, draw)
    assert not np.array_equal(expected[0], different)


def test_retry_does_not_resample_and_failure_does_not_advance():
    sessions = Pi05PolicySessions()
    first = sessions.run("a", 22, 0, draw)
    np.testing.assert_array_equal(first, sessions.run("a", 22, 0, lambda: pytest.fail("resampled")))

    def broken():
        draw()
        raise RuntimeError("temporary inference failure")

    with pytest.raises(RuntimeError, match="temporary"):
        sessions.run("a", 22, 1, broken)
    recovered = sessions.run("a", 22, 1, draw)
    reference = Pi05PolicySessions()
    reference.run("ref", 22, 0, draw)
    np.testing.assert_array_equal(recovered, reference.run("ref", 22, 1, draw))
    assert sessions.release("a") == {"released": True}
    assert sessions.release("a") == {"released": False}
    with pytest.raises(ValueError, match="missing or expired"):
        sessions.run("a", 22, 2, draw)


def test_isolated_inference_preserves_service_rng():
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    before = draw()
    sessions = Pi05PolicySessions()
    sessions.run("one", 334, 0, draw)
    after = draw()
    random.seed(45)
    np.random.seed(45)
    torch.manual_seed(45)
    np.testing.assert_array_equal(before, draw())
    np.testing.assert_array_equal(after, draw())


def test_session_identity_and_order_are_checked():
    sessions = Pi05PolicySessions(max_sessions=1)
    sessions.run("a", 1, 0, draw)
    with pytest.raises(ValueError, match="cannot change"):
        sessions.run("a", 2, 1, draw)
    with pytest.raises(ValueError, match="consecutive"):
        sessions.run("a", 1, 2, draw)
    with pytest.raises(RuntimeError, match="capacity"):
        sessions.run("b", 2, 0, draw)
    sessions.sessions["a"].touched -= 4000
    sessions.run("b", 2, 0, draw)
    assert "a" not in sessions.sessions


def test_client_retries_same_id_after_lost_response_and_releases(monkeypatch):
    monkeypatch.setenv("RPENT_POLICY_RNG", "56")
    monkeypatch.setenv("RPENT_VLA_SESSION_ID", "episode-one")
    monkeypatch.setenv("RPENT_VLA_SESSION_PROTOCOL", "pi05")

    class Transport:
        def __init__(self):
            self.sessions = Pi05PolicySessions()
            self.calls = []
            self.lose_response = True

        def call(self, method, args=(), **kwargs):
            self.calls.append((method, args))
            if method == "vla.release_session":
                return self.sessions.release(*args)
            _, options = args
            result = self.sessions.run(options["session_id"], options["policy_seed"], options["request_id"], draw)
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("response lost after inference")
            return result

    transport = Transport()
    client = Pi05VLAClient(transport)
    obs = {"states": np.array([1, 2])}
    with pytest.raises(TimeoutError):
        client.predict(obs)
    with pytest.raises(RuntimeError, match="identical input"):
        client.predict({"states": np.array([4, 5])})
    recovered = client.predict(obs)
    expected = Pi05PolicySessions().run("ref", 56, 0, draw)
    np.testing.assert_array_equal(recovered, expected)
    assert [call[1][1]["request_id"] for call in transport.calls] == [0, 0]
    client.close()
    client.close()
    assert transport.sessions.sessions == {}
    with pytest.raises(RuntimeError, match="closed"):
        client.predict(obs)


def test_ordinary_client_keeps_original_options(monkeypatch):
    for name in ("RPENT_POLICY_RNG", "RPENT_VLA_SESSION_ID", "RPENT_VLA_SESSION_PROTOCOL"):
        monkeypatch.delenv(name, raising=False)

    class Transport:
        def call(self, method, args, **kwargs):
            assert method == "vla.predict"
            assert args[1] == {"mode": "eval"}
            return np.zeros((1, 5, 7), dtype=np.float32)

    client = Pi05VLAClient(Transport())
    actions, _ = client.predict_action_batch({"main_images": np.zeros((2, 2, 3)), "states": np.zeros(8)})
    assert actions.shape == (5, 7)
    client.close()
