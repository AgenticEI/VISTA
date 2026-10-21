"""RPent reference API image-history policy, expressed as an ordinary LIBERO package."""

from copy import deepcopy


class Policy:
    """Omit old camera images behind a monotonic watermark.

    The history budget used to be re-evaluated from scratch on every request,
    so images flipped to the placeholder text at a slightly different position
    each turn and provider-side prompt caching never saw a stable prefix.
    Now an omitted image stays omitted: the watermark only moves forward, the
    placeholder string never changes, and consecutive requests share one
    byte-stable prefix. The watermark advances only when retained image bytes
    exceed the four-MiB budget, and never past the newest two images so the
    model keeps its current view.
    """

    _PLACEHOLDER = "[earlier camera image omitted to bound request size]"
    _MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024
    _MIN_RECENT_IMAGES = 2

    def __init__(self):
        # Item position (message, part, item) before which every image is
        # omitted. Only moves forward within an episode, so a position's
        # rendered form changes at most once: image -> placeholder, forever.
        self._omit_before = (0, 0, 0)

    def prepare_request(self, request, state):
        located = []
        for mi, message in enumerate(request.messages):
            for pi, part in enumerate(message.get("parts", [])):
                content = part.get("content")
                if part.get("part_kind") != "user-prompt" or not isinstance(content, list):
                    continue
                for ii, item in enumerate(content):
                    if (isinstance(item, dict) and item.get("kind") == "binary"
                            and item.get("media_type", "").startswith("image/")):
                        located.append((mi, pi, ii, len(item["data"])))
        if not located:
            return request

        # The newest images always survive so the model keeps its current view.
        protected = {(mi, pi, ii) for mi, pi, ii, _ in located[-self._MIN_RECENT_IMAGES:]}

        # Advance the watermark only while retained bytes exceed the budget.
        boundary = self._omit_before
        retained = sum(size for mi, pi, ii, size in located
                       if (mi, pi, ii) >= boundary)
        if retained > self._MAX_HISTORY_IMAGE_BYTES:
            for mi, pi, ii, size in located:  # oldest -> newest
                if retained <= self._MAX_HISTORY_IMAGE_BYTES:
                    break
                position = (mi, pi, ii)
                if position < boundary or position in protected:
                    continue
                boundary = (mi, pi, ii + 1)
                retained -= size
            self._omit_before = max(self._omit_before, boundary)

        drop = [(mi, pi, ii) for mi, pi, ii, _ in located
                if (mi, pi, ii) < self._omit_before]
        if not drop:
            return request
        request.messages = deepcopy(request.messages)
        for mi, pi, ii in drop:
            request.messages[mi]["parts"][pi]["content"][ii] = self._PLACEHOLDER
        return request

    def process_tool_result(self, call, raw_result, state):
        return raw_result

    def decide(self, request, state):
        return None
